import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import chi2
from scipy.optimize import linear_sum_assignment

# ─────────────────────────────────────────────────────────────────
# NEES — requires ground truth states
# ─────────────────────────────────────────────────────────────────

def compute_nees(x_true, x_est, P_est):
    """
    Normalized Estimation Error Squared for a single timestep.
    x_true, x_est: (n,) state vectors
    P_est:         (n,n) estimated covariance
    Returns scalar — should be chi2(n) distributed if filter is consistent.
    """
    err = x_true - x_est
    return float(err @ np.linalg.inv(P_est) @ err)


# ─────────────────────────────────────────────────────────────────
# NIS — no ground truth needed, just measurements + filter state
# ─────────────────────────────────────────────────────────────────

def compute_nis(z, z_pred, S):
    """
    Normalized Innovation Squared for a single measurement.
    z:      (m,) actual measurement
    z_pred: (m,) predicted measurement H @ x_pred
    S:      (m,m) innovation covariance H @ P @ H.T + R
    Returns scalar — should be chi2(m) distributed.
    """
    nu = z - z_pred
    return float(nu @ np.linalg.inv(S) @ nu)


# ─────────────────────────────────────────────────────────────────
# OSPA — the right metric for RFS / GM-PHD
# ─────────────────────────────────────────────────────────────────

def compute_ospa(true_states, est_states, c=100.0, p=2):
    """
    Optimal Subpattern Assignment metric.

    true_states: list of (2,) or (4,) arrays — ground truth positions
    est_states:  list of (2,) or (4,) arrays — estimated positions
    c:           cutoff distance (pixels). Errors larger than c are capped.
    p:           order (1 = absolute, 2 = squared — more sensitive to large errors)

    Returns (ospa, loc_component, card_component)
      ospa            — total OSPA distance
      loc_component   — location error on matched pairs
      card_component  — penalty for cardinality mismatch
    """
    n = len(true_states)
    m = len(est_states)

    if n == 0 and m == 0:
        return 0.0, 0.0, 0.0
    if n == 0 or m == 0:
        return c, 0.0, c    # pure cardinality error

    # Use only position components for distance (first 2 dims)
    X = np.array([s[:2] for s in true_states])   # (n, 2)
    Y = np.array([s[:2] for s in est_states])     # (m, 2)

    # Build cost matrix — min(dist, c)^p
    cost = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            d = np.linalg.norm(X[i] - Y[j])
            cost[i, j] = min(d, c) ** p

    # Optimal assignment on the smaller set
    k = min(n, m)
    row_ind, col_ind = linear_sum_assignment(cost[:k, :k] if n >= m
                                             else cost[:k, :k].T)
    matched_cost = cost[row_ind, col_ind].sum() if n >= m \
                   else cost.T[row_ind, col_ind].sum()

    # Cardinality penalty: unmatched objects each contribute c^p
    card_penalty = (abs(n - m)) * (c ** p)

    ospa_p = (1.0 / max(n, m)) * (matched_cost + card_penalty)
    ospa   = ospa_p ** (1.0 / p)

    loc_component  = ((1.0 / max(n, m)) * matched_cost) ** (1.0 / p)
    card_component = ((1.0 / max(n, m)) * card_penalty) ** (1.0 / p)

    return ospa, loc_component, card_component


# ─────────────────────────────────────────────────────────────────
# Cardinality tracker
# ─────────────────────────────────────────────────────────────────

def cardinality_estimate(gmm_components):
    """
    Expected number of targets = sum of all component weights.
    Round to nearest int for a hard count estimate.
    """
    total_weight = sum(c.weight for c in gmm_components)
    return total_weight, round(total_weight)


# ─────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────

def plot_consistency(nees_history, nis_history, state_dim=4, meas_dim=2):
    """
    Plot NEES and NIS over time with chi-squared 95% confidence bands.
    Values inside the band → filter is consistent.
    """
    T     = max(len(nees_history), len(nis_history))
    alpha = 0.05
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # NEES
    if nees_history:
        ax1.plot(nees_history, label="NEES", color="steelblue", linewidth=1.2)
        lo = chi2.ppf(alpha / 2, df=state_dim)
        hi = chi2.ppf(1 - alpha / 2, df=state_dim)
        ax1.axhline(lo, color="red", linestyle="--", linewidth=0.8, label=f"95% band [{lo:.1f}, {hi:.1f}]")
        ax1.axhline(hi, color="red", linestyle="--", linewidth=0.8)
        ax1.fill_between(range(len(nees_history)), lo, hi, alpha=0.08, color="red")
        ax1.set_ylabel("NEES")
        ax1.legend(fontsize=8)
        ax1.set_title("NEES — filter consistency (needs ground truth)")

    # NIS
    if nis_history:
        ax2.plot(nis_history, label="NIS", color="darkorange", linewidth=1.2)
        lo = chi2.ppf(alpha / 2, df=meas_dim)
        hi = chi2.ppf(1 - alpha / 2, df=meas_dim)
        ax2.axhline(lo, color="red", linestyle="--", linewidth=0.8, label=f"95% band [{lo:.1f}, {hi:.1f}]")
        ax2.axhline(hi, color="red", linestyle="--", linewidth=0.8)
        ax2.fill_between(range(len(nis_history)), lo, hi, alpha=0.08, color="red")
        ax2.set_ylabel("NIS")
        ax2.set_xlabel("Frame")
        ax2.legend(fontsize=8)
        ax2.set_title("NIS — measurement consistency (no ground truth needed)")

    plt.tight_layout()
    plt.savefig("consistency_plots.png", dpi=150)
    plt.show()


def plot_ospa(ospa_history, loc_history, card_history):
    """Plot OSPA total and its two components over time."""
    frames = range(len(ospa_history))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frames, ospa_history,  label="OSPA total",     color="purple",  linewidth=1.5)
    ax.plot(frames, loc_history,   label="Location error", color="steelblue", linestyle="--", linewidth=1.0)
    ax.plot(frames, card_history,  label="Cardinality err", color="coral",   linestyle=":",  linewidth=1.0)
    ax.set_xlabel("Frame")
    ax.set_ylabel("OSPA distance (pixels)")
    ax.set_title("OSPA over time")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("ospa_plot.png", dpi=150)
    plt.show()


def draw_covariance_ellipse(ax, mean, cov_2x2, n_std=2.0, color="cyan", alpha=0.3):
    """
    Draw a covariance ellipse on a matplotlib axis.
    mean:    (2,) position
    cov_2x2: (2,2) position submatrix of the state covariance
    n_std:   number of standard deviations to draw (2 = 95% for 2D Gaussian)
    """
    eigvals, eigvecs = np.linalg.eigh(cov_2x2)
    eigvals = np.maximum(eigvals, 0)   # numerical guard
    angle   = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    w, h    = 2 * n_std * np.sqrt(eigvals)

    ellipse = patches.Ellipse(
        xy=(mean[0], mean[1]), width=w, height=h,
        angle=angle, linewidth=1.2,
        edgecolor=color, facecolor=color, alpha=alpha
    )
    ax.add_patch(ellipse)
    ax.plot(mean[0], mean[1], "+", color=color, markersize=6, linewidth=1.2)


def visualise_frame(img, measurements, gmm_components, frame_idx,
                    true_states=None, save_dir="eval_frames"):
    """
    Overlay on a single frame:
      - SAM3 detected centroids (yellow crosses)
      - GM-PHD extracted state estimates (cyan ellipses + crosses)
      - Ground truth positions if provided (green circles)
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)
    ax.set_title(f"Frame {frame_idx}  |  {len(measurements)} detections  |  "
                 f"PHD mass = {sum(c.weight for c in gmm_components):.2f} targets")

    # SAM3 measurements
    for m in measurements:
        z = m["z"]
        ax.plot(z[0], z[1], "x", color="yellow", markersize=8, linewidth=1.5,
                label="SAM3 detection" if m is measurements[0] else "")

    # GM-PHD estimates — only components above weight threshold
    estimate_plotted = False
    for comp in gmm_components:
        if comp.weight < 0.5:
            continue
        pos  = comp.loc.flatten()[:2]
        cov2 = comp.cov[:2, :2]
        draw_covariance_ellipse(ax, pos, cov2, n_std=2.0,
                                color="cyan", alpha=0.25)
        label = "PHD estimate" if not estimate_plotted else ""
        ax.plot(pos[0], pos[1], "+", color="cyan", markersize=10, linewidth=2,
                label=label)
        estimate_plotted = True

    # Ground truth (optional)
    if true_states is not None:
        for i, gt in enumerate(true_states):
            label = "Ground truth" if i == 0 else ""
            ax.plot(gt[0], gt[1], "o", color="lime", markersize=8,
                    markerfacecolor="none", linewidth=1.5, label=label)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc="upper right", fontsize=8, framealpha=0.6)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/frame_{frame_idx:04d}.png", dpi=120)
    plt.close(fig)