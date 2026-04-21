import torch
from ultralytics.models.sam import sam3
from ultralytics.models.sam import SAM3SemanticPredictor
import glob
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
#import syntheticexamplestuff 
from gmphd import Gmphd,GmphdComponent
#Helper functions
def extract_measurements(results):
    """
    Parse the SAM3 results into a list of measurements dicts

    Each measurement will hold:
        z: 2x1 [cx,cy] centroid in pixel space
        conf: a scalar detection confidence in [0,1]

    Confidence is not folded into R here. It is passed through to the PHD likelihood so the update step can weight each gaussian accordingly.
    """

    measurements = []
    for result in results:
        if result.masks is None:
            continue
        masks = result.masks.data.cpu().numpy()
        confs = (result.boxes.conf.cpu().numpy() 
                if result.boxes is not None else np.ones(len(masks)))

        for mask,conf in zip(masks,confs):
            if conf < MIN_DETECTION_CONF:
                continue
            ys,xs = np.where(mask > MASK_PIXEL_THRESHOLD)
            if len(xs) ==0:
                continue
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            measurements.append({"z":np.array([cx,cy]),"conf":float(conf)})
    return measurements


def build_birth_gmm(measurements, birth_weight, P_birth):
    """Spawn a birth component at each measurement location each frame"""
    birth_gmm = []
    for m in measurements:
        z=m["z"]
        #born with zero velocity since we don't know it yet
        loc = np.array([z[0],z[1],0.0,0.0])
        birth_gmm.append(GmphdComponent(
                weight=birth_weight,
                mean=loc,
                cov = P_birth.copy()
        ))
    return birth_gmm

import matplotlib.patches as patches
import matplotlib.cm as cm

def visualise_single_frame(img, results, measurements, gmm_components, targets, frame_idx=0):
    """
    Overlay SAM3 masks, measurement centroids, and GM-PHD estimates on the image.
    
    - Coloured translucent masks — one per SAM3 detection
    - Yellow crosses — raw measurement centroids
    - Cyan circles + covariance ellipses — GM-PHD extracted targets
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)

    colours = cm.Set1(np.linspace(0, 1, 9))

    # ── SAM3 masks ────────────────────────────────────────────────
    for i, result in enumerate(results):
        if result.masks is None:
            continue
        masks = result.masks.data.cpu().numpy()   # (N, H, W)
        confs = (result.boxes.conf.cpu().numpy()
                 if result.boxes is not None else np.ones(len(masks)))

        img_w, img_h = img.size
        for j, (mask, conf) in enumerate(zip(masks, confs)):
            # Resize mask to image dimensions
            from PIL import Image as PILImage
            mask_img = PILImage.fromarray((mask * 255).astype(np.uint8)).resize(
                (img_w, img_h), PILImage.NEAREST)
            mask_np = np.array(mask_img) / 255.0

            colour = colours[j % len(colours)][:3]
            overlay = np.zeros((*mask_np.shape, 4))
            overlay[mask_np > 0.5] = [*colour, 0.4]
            ax.imshow(overlay)

            # Mask label
            ys, xs = np.where(mask_np > 0.5)
            if len(xs):
                ax.text(np.mean(xs), np.mean(ys) - 10,
                        f"det {j+1}  conf {conf:.2f}",
                        color="white", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc=colour, alpha=0.7))

    # ── Raw measurement centroids ─────────────────────────────────
    for k, m in enumerate(measurements):
        z = m["z"]
        ax.plot(z[0], z[1], "x", color="yellow",
                markersize=12, markeredgewidth=2,
                label="SAM3 centroid" if k == 0 else "")
        ax.text(z[0] + 5, z[1] - 8, f"z{k+1}", color="yellow", fontsize=8)

    # ── GM-PHD estimates ──────────────────────────────────────────
    active = [c for c in gmm_components if c.weight > 0.05]
    for k, comp in enumerate(active):
        pos  = comp.mean.flatten()[:2]
        cov2 = comp.cov[:2, :2]

        # Covariance ellipse (2-sigma)
        eigvals, eigvecs = np.linalg.eigh(cov2)
        eigvals = np.maximum(eigvals, 0)
        angle   = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        w, h    = 2 * 2.0 * np.sqrt(eigvals)   # 2-sigma axes
        ellipse = patches.Ellipse(
            xy=(pos[0], pos[1]), width=w, height=h, angle=angle,
            edgecolor="cyan", facecolor="cyan", alpha=0.15, linewidth=1.5)
        ax.add_patch(ellipse)

        ax.plot(pos[0], pos[1], "o", color="cyan",
                markersize=8, markerfacecolor="none", markeredgewidth=2,
                label="PHD estimate" if k == 0 else "")
        ax.text(pos[0] + 5, pos[1] + 10,
                f"w={comp.weight:.3f}", color="cyan", fontsize=8)
        
        print(f"Target {k} located at state {comp.mean}")

    # ── PHD mass readout ──────────────────────────────────────────
    total_mass = sum(c.weight for c in gmm_components)
    ax.set_title(
        f"Frame {frame_idx}  |  {len(measurements)} detections  |  "
        f"PHD mass = {total_mass:.2f}  |  {len(targets)} extracted targets",
        fontsize=11)

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc="upper right", fontsize=9, framealpha=0.6)

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"frame_{frame_idx:04d}_overlay.png", dpi=150)
    plt.close(fig)
    #plt.show()
    print(f"Saved frame_{frame_idx:04d}_overlay.png")

#-----------------------------------------------------------------------------
#Importing Ground Truths
def load_mot_ground_truth(filepath):
    """
    Load MOT-style ground truth file.

    Returns:
        dict: frame_idx -> list of dicts:
              {"id": int, "state": np.array([cx, cy])}
    """
    gt_by_frame = {}

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split(",")

            frame = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])

            # Convert to center coordinates
            cx = x + w / 2.0
            cy = y + h / 2.0

            state = np.array([cx, cy])

            if frame not in gt_by_frame:
                gt_by_frame[frame] = []

            gt_by_frame[frame].append({
                "id": track_id,
                "state": state
            })

    return gt_by_frame

#Grabbing Ground truth states 
def get_gt_states(gt_by_frame, frame_idx):
    """
    Returns list of [x,y] states for a frame.
    """
    if frame_idx not in gt_by_frame:
        return []

    return [obj["state"] for obj in gt_by_frame[frame_idx]]
#------------------------------------------------------------------------------
#Ospa Evaluation 

def ospa_distance(X, Y, c=100.0, p=2):
    """
    Compute OSPA distance between two sets of states.

    X: list of ground truth vectors (each shape [n])
    Y: list of estimated vectors (each shape [n])
    c: cutoff distance
    p: order parameter

    Returns:
        ospa distance (scalar)
    """

    m = len(X)
    n = len(Y)

    if m == 0 and n == 0:
        return 0.0
    if m == 0:
        return ( (n * (c ** p)) / n ) ** (1/p)
    if n == 0:
        return ( (m * (c ** p)) / m ) ** (1/p)

    # Cost matrix
    cost = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            d = np.linalg.norm(X[i][:2] - Y[j][:2])  # compare position only
            cost[i, j] = min(c, d) ** p

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    assignment_cost = cost[row_ind, col_ind].sum()

    # Cardinality penalty
    card_penalty = abs(m - n) * (c ** p)

    ospa = ((assignment_cost + card_penalty) / max(m, n)) ** (1/p)
    return ospa

def cardinality_error(X, Y):
    """
    Absolute difference in number of targets.
    
    X: ground truth list
    Y: estimated list
    """
    return abs(len(X) - len(Y))
def cardinality_bias(X, Y):
    """
    Positive = overestimation
    Negative = underestimation
    """
    return len(Y) - len(X)

def nis_bounds(dof, alpha=0.05):
    lower = chi2.ppf(alpha / 2, dof)
    upper = chi2.ppf(1 - alpha / 2, dof)
    return lower, upper

def compute_nis(measurement, components, H, R):
    z = np.array(measurement).reshape(-1,1)
    total_weight = sum(c.weight for c in components)
    if total_weight == 0:
        return None
    weighted_nis = 0.0
    for c in components:
        z_pred = (H@c.mean).reshape(-1,1)
        S = H @ c.cov @ H.T + R
        innov = z - z_pred
        nis  = (innov.T @ np.linalg.inv(S) @ innov).item()
        weighted_nis += (c.weight / total_weight) * nis
    return weighted_nis


def plot_nis(frame_ids, nis_vals, dof=2):
    lower, upper = nis_bounds(dof)

    plt.figure()
    plt.plot(frame_ids, nis_vals, label="NIS")

    plt.axhline(lower, linestyle="--", label="Lower bound (95%)")
    plt.axhline(upper, linestyle="--", label="Upper bound (95%)")

    plt.xlabel("Frame")
    plt.ylabel("NIS")
    plt.title("NIS Consistency Test")
    plt.legend()
    plt.grid()

    plt.show()

def plot_metrics(frame_ids, ospa_vals, card_vals):
    plt.figure()
    plt.plot(frame_ids, ospa_vals)
    plt.xlabel("Frame")
    plt.ylabel("OSPA")
    plt.title("OSPA over Time")
    plt.grid()

    plt.figure()
    plt.plot(frame_ids, card_vals)
    plt.xlabel("Frame")
    plt.ylabel("Cardinality Error")
    plt.title("Cardinality Error over Time")
    plt.grid()

    plt.show()



def plot_mc_results(frame_ids, ospa_mean, ospa_std,
                    cbias_mean, cbias_std,
                    all_ospa, all_card_bias):
    """
    Plot Monte Carlo aggregated results.
      - Top:    OSPA mean ± 1 std, with per-run traces in the background
      - Bottom: Cardinality bias mean ± 1 std, with per-run traces
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("GM-PHD Monte Carlo Analysis", fontsize=14, fontweight="bold")

    # ── OSPA ──────────────────────────────────────────────────────────────────
    ax = axes[0]

    # Per-run traces (light, in background)
    for run in all_ospa:
        ax.plot(frame_ids, run, color="steelblue", alpha=0.15, linewidth=0.8)

    # Mean ± std band
    ax.fill_between(frame_ids,
                    ospa_mean - ospa_std,
                    ospa_mean + ospa_std,
                    color="steelblue", alpha=0.3, label="Mean ± 1σ")
    ax.plot(frame_ids, ospa_mean, color="steelblue", linewidth=2, label="Mean OSPA")

    ax.set_ylabel("OSPA Distance")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(f"OSPA  (N={all_ospa.shape[0]} runs)")

    # ── Cardinality Bias ──────────────────────────────────────────────────────
    ax = axes[1]

    for run in all_card_bias:
        ax.plot(frame_ids, run, color="darkorange", alpha=0.15, linewidth=0.8)

    ax.fill_between(frame_ids,
                    cbias_mean - cbias_std,
                    cbias_mean + cbias_std,
                    color="darkorange", alpha=0.3, label="Mean ± 1σ")
    ax.plot(frame_ids, cbias_mean, color="darkorange", linewidth=2, label="Mean CardBias")
    ax.axhline(0, color="black", linewidth=1, linestyle="--", label="Zero bias")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Cardinality Bias (est − gt)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Cardinality Bias  (positive = over-counting, negative = under-counting)")

    plt.tight_layout()
    plt.savefig("mc_results.png", dpi=150)
    plt.show()
# if __name__ == "__main__":
#     # Initialize predictor
#     overrides = dict(conf=0.25, task="segment", mode="track", model="sam3.pt", half=True, save=True)
#     predictor = SAM3SemanticPredictor(overrides=overrides)


#     #print("Please input filepath for video analysis")
#     #"~/data/M3OT/1/rgb/train/1-01/img1/000001.PNG"
#     #filepath = input()

#     import os
#     #image_paths = glob.glob(os.path.expanduser("~/data/M3OT/1/rgb/train/1-01/img1/000001.PNG"))
#     # image_paths = glob.glob(os.path.expanduser("~/data/M3OT/1/ir/train/1-01T/img1/*.PNG"))
#     image_paths = glob.glob(os.path.expanduser("~/data/M3OT/2/ir/test/2-03T/img1/*.PNG"))
#     #image_paths = sorted(glob.glob(os.path.expanduser("~/data/M3OT/1/rgb/train/1-01/img1/*.PNG")))[:100]
#     # image_paths = sorted(
#     # glob.glob(os.path.expanduser("~/data/M3OT/1/ir/train/1-01T/img1/*.PNG"))
#     # )[:300]

#     # gt_by_frame = load_mot_ground_truth(os.path.expanduser("~/data/M3OT/1/ir/train/1-01T/gt/gt.txt"))

    
#     # image_paths = sorted(
#     #     glob.glob(os.path.expanduser("~/data/M3OT/1/rgb/train/1-01/img1/*.PNG"))
#     # )[:300]

#     # gt_by_frame = load_mot_ground_truth(os.path.expanduser("~/data/M3OT/1/rgb/train/1-01/gt/gt.txt"))
#     image_paths = sorted(
#         glob.glob(os.path.expanduser("~/data/M3OT/2/ir/test/2-03T/img1/*.PNG"))
#     )[:300]

#     gt_by_frame = load_mot_ground_truth(os.path.expanduser("~/data/M3OT/2/ir/test/2-03T/gt/gt.txt"))
#     print(f"Found {len(image_paths)} images")

#     print(f"Found {len(image_paths)} images")
#     #print(f"Found {len(image_paths)} images: {image_paths}")
#     #image_paths = glob.glob("~/data/M3OT/1/rgb/train/1-01/img1/000001.PNG")
#     image_area = 1920 * 1080
#     # Detection Thresholds
#     MASK_PIXEL_THRESHOLD = 0.5 # Only want to really accept things that we are some level of confident in. 50% feels reasonable
#     MIN_DETECTION_CONF   = 0.35

#     #Motion Model: CV and White noise accelertation
#     dt = 1
#     #Define the state transition matrix to be x = [x,y,vx,vy]
#     F = np.array([[1,0,dt,0],
#                 [0,1,0,dt],
#                 [0,0,1,0],
#                 [0,0,0,1]], dtype=np.float64)


#     q = 0.9
#     Q = q * np.array([
#         [dt**3/3, 0,        dt**2/2, 0],
#         [0,        dt**3/3, 0,        dt**2/2],
#         [dt**2/2, 0,        dt,        0],
#         [0,        dt**2/2, 0,        dt]
#     ],dtype=np.float64)

#     #Measurement Model: Observing [x,y] only
#     H = np.hstack((np.eye(2), np.zeros((2,2))))
#     r = 5.0
#     R = r*np.eye(2)

#     #GM-PHD parameters
#     birth_prob              = 0.1
#     survival_prob           = 0.975
#     detect_prob             = 0.9 #0.95
#     clutter_total = 5 #Defines clutter per frame
#     clutter_intensity = clutter_total/image_area
#     bias                    = 2 #Defines the bias towards false positives over missed detections

#     m0 = np.array([0.0,0.0,0.0,0.0]) #x,y,Vx,Vy
#     P0 = 10000 * np.eye(4)


#     #birth_gmm = [{"weight": birth_prob, "mean": m0.copy(), "cov": P0.copy()}]
#     birth_gmm = [GmphdComponent(weight=birth_prob, mean=m0.copy(), cov=P0.copy())]
#     gmphd_filter = Gmphd(birth_gmm,birth_prob,survival_prob,detect_prob, F, Q, H, R, clutter_intensity)
#     ospa_history = []
#     cardinality_history = []
#     cardinalty_bias_history = []
#     frame_id = []
#     nis_history = []
#     state_err = []
#     lower, upper = nis_bounds(dof=2)
#     print(lower, upper)

#     for frame_idx, path in enumerate(image_paths):
#         try: 
#             #Open image
#             with Image.open(path) as img:
#                 predictor.set_image(img)
#                 results = predictor(text=["vehicles"])
#                 # ── SAM3 debug ────────────────────────────────────────────────────
#                 print(f"Number of results: {len(results)}")
#                 for i, r in enumerate(results):
#                     print(f"  Result {i}:")
#                     print(f"    masks:  {r.masks.data.shape if r.masks is not None else 'None'}")
#                     print(f"    boxes:  {r.boxes.data.shape if r.boxes is not None else 'None'}")
#                     print(f"    confs:  {r.boxes.conf.cpu().numpy() if r.boxes is not None else 'None'}")

#                 measurements = extract_measurements(results)

#                 print(f"Measurements extracted: {len(measurements)}")
#                 for m in measurements:
#                     print(f"  z={m['z']}, conf={m['conf']:.3f}")

#                 gmphd_filter.birthgmm = build_birth_gmm(measurements, birth_prob, P0)
#                 gmphd_filter.update(measurements)
#                 gmphd_filter.prune_targets()
#                 targets = gmphd_filter.extractstate()
#                 print(f"PHD mass: {sum(c.weight for c in gmphd_filter.gmm):.3f}, targets extracted: {len(targets)}")
#                 #visualise_single_frame(img, results, measurements, gmphd_filter.gmm, targets, frame_idx=0)
#                 visualise_single_frame(img, results, measurements, gmphd_filter.gmm, targets, frame_idx=frame_idx)
#                 print(f"[{path}] {len(measurements)} measurements → {len(targets)} targets")

#                 #OSPA Analysis 
#                 #gt_states = get_gt_states(gt_by_frame, frame_idx + 1)  
#                 frame_number = int(os.path.splitext(os.path.basename(path))[0])
#                 gt_states = get_gt_states(gt_by_frame, frame_number)

#                 est_states = [t[:2] for t in targets]

#                 ospa = ospa_distance(gt_states, est_states)
#                 ospa_history.append(ospa)
#                 card_err = cardinality_error(gt_states, est_states)
#                 card_bias = cardinality_bias(gt_states,est_states)
#                 cardinality_history.append(card_err)
#                 frame_id.append(frame_idx)
#                 cardinalty_bias_history.append(card_bias)
#                 print(f"Frame {frame_idx}: OSPA={ospa:.2f}, CardErr={card_err}, CardBias={card_bias}")


#                 # #NEES approximation 
#                 nis_vals = []

#                 for m in measurements:
#                     nis = compute_nis(m["z"], gmphd_filter.gmm, H, R)
#                     if nis is not None:
#                         nis_vals.append(nis)

#                 avg_nis = np.mean(nis_vals) if nis_vals else 0

#                 nis_history.append(avg_nis)



#         except Exception as e:
#             print(f"Error processing {path}: {e}")
#             import traceback
#             traceback.print_exc()
#             print(f"Error processing {path}: {e}")
#             # Still need to append placeholders to keep arrays aligned
#             ospa_history.append(np.nan)
#             cardinality_history.append(np.nan)
#             cardinalty_bias_history.append(np.nan)
#             nis_history.append(np.nan)
#             frame_id.append(frame_idx)
#     ospa_history = np.array(ospa_history)
#     cardinality_history = np.array(cardinality_history)
#     frame_id = np.array(frame_id)
#     nis_history = np.array(nis_history)

#     plot_nis(frame_id,nis_history)

#     plot_metrics(frame_id,ospa_history,cardinality_history)

if __name__ == "__main__":
    import os
    import numpy as np

    # ── Monte Carlo configuration ─────────────────────────────────────────────
    N_RUNS = 20

    # Parameter sampling ranges (uniform)
    MC_PARAMS = {
        "survival_prob":   (0.90, 0.99),
        "detect_prob":     (0.75, 0.99),
        "clutter_total":   (1,    15),       # integer-ish, will be cast to int
    }

    # ── Fixed parameters (unchanged across runs) ──────────────────────────────
    overrides = dict(conf=0.25, task="segment", mode="track", model="sam3.pt", half=True, save=True)
    predictor = SAM3SemanticPredictor(overrides=overrides)

    # image_paths = sorted(
    #     glob.glob(os.path.expanduser("~/data/M3OT/1/rgb/train/1-01/img1/*.PNG"))
    # )[:300]

    # gt_by_frame = load_mot_ground_truth(os.path.expanduser("~/data/M3OT/1/rgb/train/1-01/gt/gt.txt"))

    
    image_paths = sorted(
        glob.glob(os.path.expanduser("~/data/M3OT/2/ir/test/2-03T/img1/*.PNG"))
    )[:300]

    gt_by_frame = load_mot_ground_truth(os.path.expanduser("~/data/M3OT/2/ir/test/2-03T/gt/gt.txt"))
    print(f"Found {len(image_paths)} images")

    image_area = 1920 * 1080

    MASK_PIXEL_THRESHOLD = 0.5
    MIN_DETECTION_CONF   = 0.35

    dt = 1
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0,  dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]], dtype=np.float64)

    q = 0.9
    Q = q * np.array([
        [dt**3/3, 0,        dt**2/2, 0      ],
        [0,        dt**3/3, 0,        dt**2/2],
        [dt**2/2, 0,        dt,        0     ],
        [0,        dt**2/2, 0,        dt     ],
    ], dtype=np.float64)

    H = np.hstack((np.eye(2), np.zeros((2, 2))))
    r = 5.0
    R = r * np.eye(2)

    birth_prob = 0.1
    m0 = np.array([0.0, 0.0, 0.0, 0.0])
    P0 = 10000 * np.eye(4)

    # ── Storage across runs ───────────────────────────────────────────────────
    # Shape: (N_RUNS, n_frames)
    all_ospa      = []
    all_card_bias = []
    sampled_params = []

    rng = np.random.default_rng(seed=42)

    # ── Monte Carlo loop ──────────────────────────────────────────────────────
    for run_idx in range(N_RUNS):
        print(f"\n{'='*60}")
        print(f"  Monte Carlo Run {run_idx + 1} / {N_RUNS}")

        # Sample parameters
        survival_prob     = rng.uniform(*MC_PARAMS["survival_prob"])
        detect_prob       = rng.uniform(*MC_PARAMS["detect_prob"])
        clutter_total     = int(rng.uniform(*MC_PARAMS["clutter_total"]))
        clutter_intensity = clutter_total / image_area

        params = dict(survival_prob=survival_prob,
                      detect_prob=detect_prob,
                      clutter_total=clutter_total)
        sampled_params.append(params)
        print(f"  Params: {params}")

        # Re-initialise filter with sampled params
        birth_gmm = [GmphdComponent(weight=birth_prob, mean=m0.copy(), cov=P0.copy())]
        gmphd_filter = Gmphd(birth_gmm, birth_prob, survival_prob, detect_prob,
                             F, Q, H, R, clutter_intensity)

        run_ospa      = []
        run_card_bias = []

        for frame_idx, path in enumerate(image_paths):
            try:
                with Image.open(path) as img:
                    predictor.set_image(img)
                    results = predictor(text=["vehicles"])

                    measurements = extract_measurements(results)

                    gmphd_filter.birthgmm = build_birth_gmm(measurements, birth_prob, P0)
                    gmphd_filter.update(measurements)
                    gmphd_filter.prune_targets()
                    targets = gmphd_filter.extractstate()

                    frame_number = int(os.path.splitext(os.path.basename(path))[0])
                    gt_states    = get_gt_states(gt_by_frame, frame_number)
                    est_states   = [t[:2] for t in targets]

                    ospa      = ospa_distance(gt_states, est_states)
                    card_bias = cardinality_bias(gt_states, est_states)

                    run_ospa.append(ospa)
                    run_card_bias.append(card_bias)

                    if frame_idx % 50 == 0:
                        print(f"    Frame {frame_idx:3d}: OSPA={ospa:.2f}, CardBias={card_bias:+.1f}")

            except Exception as e:
                print(f"  Error at frame {frame_idx} ({path}): {e}")
                run_ospa.append(np.nan)
                run_card_bias.append(np.nan)

        all_ospa.append(run_ospa)
        all_card_bias.append(run_card_bias)

    # ── Aggregate results ─────────────────────────────────────────────────────
    all_ospa      = np.array(all_ospa)       # (N_RUNS, n_frames)
    all_card_bias = np.array(all_card_bias)

    frame_ids = np.arange(len(image_paths))

    ospa_mean  = np.nanmean(all_ospa, axis=0)
    ospa_std   = np.nanstd(all_ospa, axis=0)
    cbias_mean = np.nanmean(all_card_bias, axis=0)
    cbias_std  = np.nanstd(all_card_bias, axis=0)

    # ── Summary statistics (per-run scalar) ───────────────────────────────────
    print("\n" + "="*60)
    print("  Monte Carlo Summary")
    print("="*60)
    for i, p in enumerate(sampled_params):
        run_ospa_mean  = np.nanmean(all_ospa[i])
        run_cbias_mean = np.nanmean(all_card_bias[i])
        print(f"  Run {i+1:2d} | Pd={p['detect_prob']:.3f}  Ps={p['survival_prob']:.3f}  "
              f"Clutter={p['clutter_total']:2d} | "
              f"OSPA={run_ospa_mean:.2f}  CardBias={run_cbias_mean:+.2f}")

    print(f"\n  Overall OSPA:      {np.nanmean(ospa_mean):.2f} ± {np.nanmean(ospa_std):.2f}")
    print(f"  Overall CardBias:  {np.nanmean(cbias_mean):+.2f} ± {np.nanmean(cbias_std):.2f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_mc_results(frame_ids, ospa_mean, ospa_std, cbias_mean, cbias_std,
                    all_ospa, all_card_bias)