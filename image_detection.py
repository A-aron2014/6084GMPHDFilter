"""
GM-PHD Multi-Target Tracker
///////////////////////////////////////////////////////////////////////////////
Usage
-----
# Quick single-pass inference (debug / visualisation)
python image_detection.py run --data ~/data/M3OT/2/ir/test/2-03T --n-frames 300 --visualise
 
# Full analytics: OSPA, NIS, cardinality plots
python image_detection.py analyze --data ~/data/M3OT/2/ir/test/2-03T --n-frames 300
 
# Monte Carlo parameter sensitivity sweep
python image_detection.py montecarlo --data ~/data/M3OT/2/ir/test/2-03T --n-frames 300 --n-runs 20 --seed 42 Be careful with this one it is a resource sink. I would drop frame nums to ~100 or runs to ~10
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass, field
import glob
from gmphd import Gmphd,GmphdComponent
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import numpy as np
import os
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
import torch
from typing import Optional
from ultralytics.models.sam import sam3
from ultralytics.models.sam import SAM3SemanticPredictor

#Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gmphd.main")

# Configuration dataclasses
@dataclass
class MotionModelConfig:
    """Constant-velocity motion model with white-noise acceleration."""
    dt: float = 1.0
    process_noise_q: float = 0.9
    measurement_noise_r: float = 5.0
 
    @property
    def F(self) -> np.ndarray:
        dt = self.dt
        return np.array(
            [[1, 0, dt, 0],
             [0, 1,  0, dt],
             [0, 0,  1,  0],
             [0, 0,  0,  1]],
            dtype=np.float64,
        )
 
    @property
    def Q(self) -> np.ndarray:
        dt, q = self.dt, self.process_noise_q
        return q * np.array(
            [[dt**3/3, 0,        dt**2/2, 0      ],
             [0,        dt**3/3, 0,        dt**2/2],
             [dt**2/2, 0,        dt,        0     ],
             [0,        dt**2/2, 0,        dt     ]],
            dtype=np.float64,
        )
 
    @property
    def H(self) -> np.ndarray:
        return np.hstack((np.eye(2), np.zeros((2, 2))))
 
    @property
    def R(self) -> np.ndarray:
        return self.measurement_noise_r * np.eye(2)
 
@dataclass
class FilterConfig:
    """GM-PHD filter and detection parameters."""
    birth_prob: float = 0.1
    survival_prob: float = 0.975
    detect_prob: float = 0.9
    clutter_total: int = 5
    image_width: int = 1920
    image_height: int = 1080
    mask_pixel_threshold: float = 0.5
    min_detection_conf: float = 0.35
    initial_state_cov_scale: float = 10_000.0
 
    @property
    def image_area(self) -> int:
        return self.image_width * self.image_height
 
    @property
    def clutter_intensity(self) -> float:
        return self.clutter_total / self.image_area
 
    @property
    def m0(self) -> np.ndarray:
        return np.zeros(4, dtype=np.float64)
 
    @property
    def P0(self) -> np.ndarray:
        return self.initial_state_cov_scale * np.eye(4)
 
@dataclass
class MonteCarloConfig:
    """Ranges for Monte Carlo parameter sampling."""
    n_runs: int = 20
    seed: int = 42
    survival_prob_range: tuple = (0.90, 0.99)
    detect_prob_range: tuple   = (0.75, 0.99)
    clutter_total_range: tuple = (1, 15)
 
 
@dataclass
class DataConfig:
    """Dataset paths and SAM3 predictor settings."""
    data_root: str = "~/data/M3OT/2/ir/test/2-03T"
    n_frames: Optional[int] = 300
    text_prompt: list = field(default_factory=lambda: ["vehicles"])
    model_weights: str = "sam3.pt"
 
    @property
    def image_paths(self) -> list:
        paths = sorted(
            glob.glob(os.path.expanduser(f"{self.data_root}/img1/*.PNG"))
        )
        return paths[: self.n_frames] if self.n_frames else paths
 
    @property
    def gt_path(self) -> str:
        return os.path.expanduser(f"{self.data_root}/gt/gt.txt")
    
# Measurement extraction
def extract_measurements(results, min_conf: float, mask_threshold: float) -> list:
    """
    Parse SAM3 results into measurement dicts.
 
    Each measurement holds:
        z:    np.array([cx, cy]) — centroid in pixel space
        conf: float              — detection confidence in [0, 1]
    """
    measurements = []
    for result in results:
        if result.masks is None:
            continue
        masks = result.masks.data.cpu().numpy()
        confs = (
            result.boxes.conf.cpu().numpy()
            if result.boxes is not None
            else np.ones(len(masks))
        )
        for mask, conf in zip(masks, confs):
            if conf < min_conf:
                continue
            ys, xs = np.where(mask > mask_threshold)
            if len(xs) == 0:
                continue
            measurements.append({
                "z":    np.array([float(np.mean(xs)), float(np.mean(ys))]),
                "conf": float(conf),
            })
    return measurements
 
 
def build_birth_gmm(measurements: list, birth_weight: float, P_birth: np.ndarray) -> list:
    """Spawn a birth component at each measurement location."""
    return [
        GmphdComponent(
            weight=birth_weight,
            mean=np.array([m["z"][0], m["z"][1], 0.0, 0.0]),
            cov=P_birth.copy(),
        )
        for m in measurements
    ]

# Ground truth helpers
def load_mot_ground_truth(filepath: str) -> dict:
    """
    Load a MOT-style ground truth file.
 
    Returns
    -------
    dict: frame_number -> list of {"id": int, "state": np.array([cx, cy])}
    """
    gt_by_frame: dict = {}
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            frame    = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            cx, cy = x + w / 2.0, y + h / 2.0
            gt_by_frame.setdefault(frame, []).append(
                {"id": track_id, "state": np.array([cx, cy])}
            )
    return gt_by_frame
 
 
def get_gt_states(gt_by_frame: dict, frame_idx: int) -> list:
    return [obj["state"] for obj in gt_by_frame.get(frame_idx, [])]
 
 
# Metric Generation
def ospa_distance(X: list, Y: list, c: float = 100.0, p: int = 2) -> float:
    """Optimal Sub-Pattern Assignment distance between two state sets."""
    m, n = len(X), len(Y)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return c
 
    cost = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            cost[i, j] = min(c, np.linalg.norm(X[i][:2] - Y[j][:2])) ** p
 
    row_ind, col_ind    = linear_sum_assignment(cost)
    assignment_cost     = cost[row_ind, col_ind].sum()
    card_penalty        = abs(m - n) * (c ** p)
    return ((assignment_cost + card_penalty) / max(m, n)) ** (1 / p)
 
 
def cardinality_error(X: list, Y: list) -> int:
    return abs(len(X) - len(Y))
 
 
def cardinality_bias(X: list, Y: list) -> int:
    """Positive = over-counting, negative = under-counting."""
    return len(Y) - len(X)
 
 
def nis_bounds(dof: int, alpha: float = 0.05) -> tuple:
    return chi2.ppf(alpha / 2, dof), chi2.ppf(1 - alpha / 2, dof)
 
 
def compute_nis(measurement, components: list, H: np.ndarray, R: np.ndarray) -> Optional[float]:
    """Weighted Normalised Innovation Squared across all GMM components."""
    total_weight = sum(c.weight for c in components)
    if total_weight == 0:
        return None
    z = np.array(measurement).reshape(-1, 1)
    weighted_nis = 0.0
    for c in components:
        z_pred = (H @ c.mean).reshape(-1, 1)
        S      = H @ c.cov @ H.T + R
        innov  = z - z_pred
        weighted_nis += (c.weight / total_weight) * (innov.T @ np.linalg.inv(S) @ innov).item()
    return weighted_nis
 
 

# Visualisation
def visualise_single_frame(
    img: Image.Image,
    results,
    measurements: list,
    gmm_components: list,
    targets: list,
    frame_idx: int = 0,
    save_dir: str = ".",
) -> None:
    """Overlay SAM3 masks, measurement centroids, and GM-PHD estimates."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    colours = cm.Set1(np.linspace(0, 1, 9))
 
    # SAM3 masks
    for result in results:
        if result.masks is None:
            continue
        masks = result.masks.data.cpu().numpy()
        confs = (result.boxes.conf.cpu().numpy()
                 if result.boxes is not None else np.ones(len(masks)))
        img_w, img_h = img.size
        for j, (mask, conf) in enumerate(zip(masks, confs)):
            mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(
                (img_w, img_h), Image.NEAREST)
            mask_np = np.array(mask_img) / 255.0
            colour  = colours[j % len(colours)][:3]
            overlay = np.zeros((*mask_np.shape, 4))
            overlay[mask_np > 0.5] = [*colour, 0.4]
            ax.imshow(overlay)
            ys, xs = np.where(mask_np > 0.5)
            if len(xs):
                ax.text(np.mean(xs), np.mean(ys) - 10,
                        f"det {j+1}  conf {conf:.2f}",
                        color="white", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc=colour, alpha=0.7))
 
    # Raw measurement centroids
    for k, m in enumerate(measurements):
        z = m["z"]
        ax.plot(z[0], z[1], "x", color="yellow", markersize=12, markeredgewidth=2,
                label="SAM3 centroid" if k == 0 else "")
        ax.text(z[0] + 5, z[1] - 8, f"z{k+1}", color="yellow", fontsize=8)
 
    # GM-PHD estimates + 2-sigma covariance ellipses
    for k, comp in enumerate([c for c in gmm_components if c.weight > 0.05]):
        pos  = comp.mean.flatten()[:2]
        cov2 = comp.cov[:2, :2]
        eigvals, eigvecs = np.linalg.eigh(cov2)
        eigvals = np.maximum(eigvals, 0)
        angle   = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        w, h    = 2 * 2.0 * np.sqrt(eigvals)
        ax.add_patch(patches.Ellipse(
            xy=(pos[0], pos[1]), width=w, height=h, angle=angle,
            edgecolor="cyan", facecolor="cyan", alpha=0.15, linewidth=1.5))
        ax.plot(pos[0], pos[1], "o", color="cyan", markersize=8,
                markerfacecolor="none", markeredgewidth=2,
                label="PHD estimate" if k == 0 else "")
        ax.text(pos[0] + 5, pos[1] + 10, f"w={comp.weight:.3f}", color="cyan", fontsize=8)
        log.debug("Target %d at state %s", k, comp.mean)
 
    total_mass = sum(c.weight for c in gmm_components)
    ax.set_title(
        f"Frame {frame_idx}  |  {len(measurements)} detections  |  "
        f"PHD mass = {total_mass:.2f}  |  {len(targets)} extracted targets",
        fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(),
              loc="upper right", fontsize=9, framealpha=0.6)
    ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"frame_{frame_idx:04d}_overlay.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out_path)
 
 
def plot_nis(frame_ids: np.ndarray, nis_vals: np.ndarray, dof: int = 2) -> None:
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
    plt.tight_layout()
    plt.savefig("nis_plot.png", dpi=150)
    plt.show()
 
 
def plot_metrics(frame_ids: np.ndarray, ospa_vals: np.ndarray, card_vals: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(frame_ids, ospa_vals)
    axes[0].set_ylabel("OSPA")
    axes[0].set_title("OSPA over Time")
    axes[0].grid()
    axes[1].plot(frame_ids, card_vals)
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Cardinality Error")
    axes[1].set_title("Cardinality Error over Time")
    axes[1].grid()
    plt.tight_layout()
    plt.savefig("metrics_plot.png", dpi=150)
    plt.show()
 
 
def plot_mc_results(
    frame_ids: np.ndarray,
    ospa_mean: np.ndarray, ospa_std: np.ndarray,
    cbias_mean: np.ndarray, cbias_std: np.ndarray,
    all_ospa: np.ndarray, all_card_bias: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("GM-PHD Monte Carlo Analysis", fontsize=14, fontweight="bold")
 
    ax = axes[0]
    for run in all_ospa:
        ax.plot(frame_ids, run, color="steelblue", alpha=0.15, linewidth=0.8)
    ax.fill_between(frame_ids, ospa_mean - ospa_std, ospa_mean + ospa_std,
                    color="steelblue", alpha=0.3, label="Mean ± 1σ")
    ax.plot(frame_ids, ospa_mean, color="steelblue", linewidth=2, label="Mean OSPA")
    ax.set_ylabel("OSPA Distance")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(f"OSPA  (N={all_ospa.shape[0]} runs)")
 
    ax = axes[1]
    for run in all_card_bias:
        ax.plot(frame_ids, run, color="darkorange", alpha=0.15, linewidth=0.8)
    ax.fill_between(frame_ids, cbias_mean - cbias_std, cbias_mean + cbias_std,
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

# Filter + predictor
def build_filter(filter_cfg: FilterConfig, motion_cfg: MotionModelConfig) -> Gmphd:
    birth_gmm = [GmphdComponent(
        weight=filter_cfg.birth_prob,
        mean=filter_cfg.m0.copy(),
        cov=filter_cfg.P0.copy(),
    )]
    return Gmphd(
        birth_gmm,
        filter_cfg.birth_prob,
        filter_cfg.survival_prob,
        filter_cfg.detect_prob,
        motion_cfg.F,
        motion_cfg.Q,
        motion_cfg.H,
        motion_cfg.R,
        filter_cfg.clutter_intensity,
    )
 
 
def build_predictor(data_cfg: DataConfig) -> SAM3SemanticPredictor:
    overrides = dict(conf=0.25, task="segment", mode="track",
                     model=data_cfg.model_weights, half=True, save=True)
    return SAM3SemanticPredictor(overrides=overrides)
 
 

# Core per-frame processing
def process_frame(
    path: str,
    frame_idx: int,
    predictor: SAM3SemanticPredictor,
    gmphd_filter: Gmphd,
    filter_cfg: FilterConfig,
    motion_cfg: MotionModelConfig,
    gt_by_frame: dict,
    data_cfg: DataConfig,
    *,
    visualise: bool = False,
    save_dir: str = ".",
) -> dict:
    """
    Run SAM3 + GM-PHD on a single frame.
    Returns per-frame metrics dict (nan-filled on exception).
    """
    result = dict(ospa=np.nan, card_err=np.nan, card_bias=np.nan, nis=np.nan)
    try:
        with Image.open(path) as img:
            predictor.set_image(img)
            detections   = predictor(text=data_cfg.text_prompt)
            measurements = extract_measurements(
                detections,
                min_conf=filter_cfg.min_detection_conf,
                mask_threshold=filter_cfg.mask_pixel_threshold,
            )
 
            gmphd_filter.birthgmm = build_birth_gmm(
                measurements, filter_cfg.birth_prob, filter_cfg.P0)
            gmphd_filter.update(measurements)
            gmphd_filter.prune_targets()
            targets = gmphd_filter.extractstate()
 
            frame_number = int(os.path.splitext(os.path.basename(path))[0])
            gt_states    = get_gt_states(gt_by_frame, frame_number)
            est_states   = [t[:2] for t in targets]
 
            result["ospa"]      = ospa_distance(gt_states, est_states)
            result["card_err"]  = cardinality_error(gt_states, est_states)
            result["card_bias"] = cardinality_bias(gt_states, est_states)
 
            nis_vals = [
                nis for m in measurements
                if (nis := compute_nis(m["z"], gmphd_filter.gmm, motion_cfg.H, motion_cfg.R))
                is not None
            ]
            result["nis"] = float(np.mean(nis_vals)) if nis_vals else 0.0
 
            if visualise:
                visualise_single_frame(
                    img, detections, measurements,
                    gmphd_filter.gmm, targets,
                    frame_idx=frame_idx,
                    save_dir=save_dir,
                )
    except Exception:
        log.exception("Error at frame %d (%s)", frame_idx, path)
    return result
 
 
# Progam Modes
def mode_run(args: argparse.Namespace) -> None:
    """Single-pass inference — debug and spot-checking."""
    data_cfg   = DataConfig(data_root=args.data, n_frames=args.n_frames,
                            model_weights=args.weights)
    filter_cfg = FilterConfig()
    motion_cfg = MotionModelConfig()
 
    image_paths = data_cfg.image_paths
    gt_by_frame = load_mot_ground_truth(data_cfg.gt_path)
    predictor   = build_predictor(data_cfg)
    gmphd       = build_filter(filter_cfg, motion_cfg)
 
    log.info("Mode: RUN | %d frames | visualise=%s", len(image_paths), args.visualise)
 
    for frame_idx, path in enumerate(image_paths):
        metrics = process_frame(
            path, frame_idx, predictor, gmphd,
            filter_cfg, motion_cfg, gt_by_frame, data_cfg,
            visualise=args.visualise, save_dir=args.save_dir,
        )
        log.info("[%3d] OSPA=%.2f  CardBias=%+.1f  NIS=%.3f",
                 frame_idx, metrics["ospa"], metrics["card_bias"], metrics["nis"])
 
 
def mode_analyze(args: argparse.Namespace) -> None:
    """Full analytics: OSPA, NIS, and cardinality plots over all frames."""
    data_cfg   = DataConfig(data_root=args.data, n_frames=args.n_frames,
                            model_weights=args.weights)
    filter_cfg = FilterConfig()
    motion_cfg = MotionModelConfig()
 
    image_paths = data_cfg.image_paths
    gt_by_frame = load_mot_ground_truth(data_cfg.gt_path)
    predictor   = build_predictor(data_cfg)
    gmphd       = build_filter(filter_cfg, motion_cfg)
 
    lower_nis, upper_nis = nis_bounds(dof=2)
    log.info("Mode: ANALYZE | %d frames | NIS bounds [%.3f, %.3f]",
             len(image_paths), lower_nis, upper_nis)
 
    ospa_history: list = []
    card_history: list = []
    nis_history:  list = []
    frame_ids:    list = []
 
    for frame_idx, path in enumerate(image_paths):
        metrics = process_frame(
            path, frame_idx, predictor, gmphd,
            filter_cfg, motion_cfg, gt_by_frame, data_cfg,
        )
        ospa_history.append(metrics["ospa"])
        card_history.append(metrics["card_err"])
        nis_history.append(metrics["nis"])
        frame_ids.append(frame_idx)
        if frame_idx % 50 == 0:
            log.info("[%3d] OSPA=%.2f  CardErr=%.1f  NIS=%.3f",
                     frame_idx, metrics["ospa"], metrics["card_err"], metrics["nis"])
 
    frame_ids    = np.array(frame_ids)
    ospa_history = np.array(ospa_history)
    card_history = np.array(card_history)
    nis_history  = np.array(nis_history)
 
    log.info("Summary — mean OSPA=%.2f  mean CardErr=%.2f  mean NIS=%.3f",
             np.nanmean(ospa_history), np.nanmean(card_history), np.nanmean(nis_history))
 
    plot_nis(frame_ids, nis_history)
    plot_metrics(frame_ids, ospa_history, card_history)
 
 
def mode_montecarlo(args: argparse.Namespace) -> None:
    """Monte Carlo sensitivity sweep over survival_prob, detect_prob, clutter."""
    data_cfg   = DataConfig(data_root=args.data, n_frames=args.n_frames,
                            model_weights=args.weights)
    motion_cfg = MotionModelConfig()
    mc_cfg     = MonteCarloConfig(n_runs=args.n_runs, seed=args.seed)
 
    image_paths = data_cfg.image_paths
    gt_by_frame = load_mot_ground_truth(data_cfg.gt_path)
    predictor   = build_predictor(data_cfg)
 
    log.info("Mode: MONTE CARLO | %d runs × %d frames | seed=%d",
             mc_cfg.n_runs, len(image_paths), mc_cfg.seed)
 
    rng            = np.random.default_rng(seed=mc_cfg.seed)
    all_ospa:      list = []
    all_card_bias: list = []
    sampled_params: list = []
 
    for run_idx in range(mc_cfg.n_runs):
        survival_prob = float(rng.uniform(*mc_cfg.survival_prob_range))
        detect_prob   = float(rng.uniform(*mc_cfg.detect_prob_range))
        clutter_total = int(rng.integers(*mc_cfg.clutter_total_range))
 
        params = dict(survival_prob=survival_prob, detect_prob=detect_prob,
                      clutter_total=clutter_total)
        sampled_params.append(params)
        log.info("Run %2d/%d — Ps=%.3f  Pd=%.3f  Clutter=%d",
                 run_idx + 1, mc_cfg.n_runs, survival_prob, detect_prob, clutter_total)
 
        filter_cfg = FilterConfig(survival_prob=survival_prob,
                                  detect_prob=detect_prob,
                                  clutter_total=clutter_total)
        gmphd = build_filter(filter_cfg, motion_cfg)
 
        run_ospa:      list = []
        run_card_bias: list = []
 
        for frame_idx, path in enumerate(image_paths):
            metrics = process_frame(
                path, frame_idx, predictor, gmphd,
                filter_cfg, motion_cfg, gt_by_frame, data_cfg,
            )
            run_ospa.append(metrics["ospa"])
            run_card_bias.append(metrics["card_bias"])
            if frame_idx % 50 == 0:
                log.debug("  [%3d] OSPA=%.2f  CardBias=%+.1f",
                          frame_idx, metrics["ospa"], metrics["card_bias"])
 
        all_ospa.append(run_ospa)
        all_card_bias.append(run_card_bias)
 
    ospa_arr  = np.array(all_ospa)
    cbias_arr = np.array(all_card_bias)
 
    ospa_mean  = np.nanmean(ospa_arr,  axis=0)
    ospa_std   = np.nanstd(ospa_arr,   axis=0)
    cbias_mean = np.nanmean(cbias_arr, axis=0)
    cbias_std  = np.nanstd(cbias_arr,  axis=0)
 
    log.info("=" * 70)
    log.info("Monte Carlo Summary")
    log.info("=" * 70)
    for i, p in enumerate(sampled_params):
        log.info("Run %2d | Pd=%.3f  Ps=%.3f  Clutter=%2d | OSPA=%.2f  CardBias=%+.2f",
                 i + 1, p["detect_prob"], p["survival_prob"], p["clutter_total"],
                 np.nanmean(ospa_arr[i]), np.nanmean(cbias_arr[i]))
    log.info("Overall — OSPA: %.2f ± %.2f   CardBias: %+.2f ± %.2f",
             np.nanmean(ospa_mean), np.nanmean(ospa_std),
             np.nanmean(cbias_mean), np.nanmean(cbias_std))
 
    plot_mc_results(np.arange(len(image_paths)),
                    ospa_mean, ospa_std, cbias_mean, cbias_std,
                    ospa_arr, cbias_arr)
    
# =============================================================================
# CLI
# =============================================================================
 
def _shared_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data", default="~/data/M3OT/2/ir/test/2-03T",
                   help="Dataset split root (expects img1/ and gt/ subdirs)")
    p.add_argument("--n-frames", type=int, default=300, metavar="N",
                   help="Max frames to process (0 = all)")
    p.add_argument("--weights", default="sam3.pt",
                   help="Path to SAM3 model weights")
    return p
 
 
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gmphd",
        description="GM-PHD Multi-Target Tracker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_shared(p):
        p.add_argument("--data", default="~/data/M3OT/2/ir/test/2-03T")
        p.add_argument("--n-frames", type=int, default=300, metavar="N")
        p.add_argument("--weights", default="sam3.pt")

    p_run = sub.add_parser("run", help="Single-pass inference (debug / quick check)")
    add_shared(p_run)
    p_run.add_argument("--visualise", action="store_true")
    p_run.add_argument("--save-dir", default=".")
    p_run.set_defaults(func=mode_run)

    p_analyze = sub.add_parser("analyze", help="Full analytics: OSPA, NIS, cardinality")
    add_shared(p_analyze)
    p_analyze.set_defaults(func=mode_analyze)

    p_mc = sub.add_parser("montecarlo", help="Monte Carlo parameter sensitivity sweep")
    add_shared(p_mc)
    p_mc.add_argument("--n-runs", type=int, default=20, metavar="N")
    p_mc.add_argument("--seed", type=int, default=42)
    p_mc.set_defaults(func=mode_montecarlo)

    return parser
 
 
def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    if hasattr(args, "n_frames") and args.n_frames == 0:
        args.n_frames = None
    args.func(args)
 
 
if __name__ == "__main__":
    main()



