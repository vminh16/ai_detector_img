"""
src/visualization/spatial.py
==================
Group 4 — Visualization & debugging helpers for Normalized Spatial Stats.

All numeric logic is delegated to ``spatial_features.py`` to guarantee
identical math between feature extraction and visualization.

Dependencies: numpy, matplotlib, src.feature_extraction.spatial.
"""
from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.feature_extraction.spatial import (
    # Constants
    CROP_SIZE,
    FEATURE_KEYS,
    MIN_GRAD_EDGE,
    MAX_GRAD_FLAT,
    NOISE_FLOOR,
    SIGMA_GUARD,
    # Private helpers (reused to guarantee identical math)
    _validate_ycrcb_array,
    _convolve_valid,
    _compute_residual_sq3,
    _compute_gradient_valid,
    _compute_flat_edge_masks,
    _extract_spatial_snr_ratio,   # signature: (R_Y, M_edge, M_flat)
    _extract_cross_noise_ratio,   # signature: (R_Y, R_Cb)
    _compute_skew_kurt_safe,
    # Public API
    extract_spatial_features,
    extract_spatial_audit,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# 1. Gradient magnitude and edge/flat masks
# ────────────────────────────────────────────────────────────────────────

def plot_gradient_and_masks(
    ycbcr_npy: np.ndarray,
) -> plt.Figure:
    """Display gradient magnitude ``G``, ``M_edge``, and ``M_flat`` masks.

    Annotates ``T_edge``, ``T_flat``, and pixel counts for each mask.

    Parameters
    ----------
    ycbcr_npy : np.ndarray
        ``(256, 256, 3)`` YCrCb array.

    Returns
    -------
    matplotlib.figure.Figure
    """
    Y, _Cr, _Cb = _validate_ycrcb_array(ycbcr_npy)

    G = _compute_gradient_valid(Y)  # (254, 254)

    T_edge = max(float(np.percentile(G, 90)), MIN_GRAD_EDGE)
    T_flat = min(float(np.percentile(G, 30)), MAX_GRAD_FLAT)

    M_edge = G >= T_edge
    M_flat = G <= T_flat

    edge_count = int(np.sum(M_edge))
    flat_count = int(np.sum(M_flat))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Gradient magnitude
    im0 = axes[0].imshow(G, cmap="hot")
    axes[0].set_title(f"Gradient |G| ({G.shape[0]}×{G.shape[1]})")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Edge mask
    axes[1].imshow(M_edge.astype(np.uint8), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        f"M_edge (G ≥ {T_edge:.1f})\n"
        f"count = {edge_count}"
    )

    # Flat mask
    axes[2].imshow(M_flat.astype(np.uint8), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(
        f"M_flat (G ≤ {T_flat:.1f})\n"
        f"count = {flat_count}"
    )

    fig.suptitle(
        f"Sobel Routing — T_edge={T_edge:.1f}, T_flat={T_flat:.1f}",
        y=1.02,
    )
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────
# 2. Residual maps (Y, Cr, Cb)
# ────────────────────────────────────────────────────────────────────────

def plot_residual_maps(
    ycbcr_npy: np.ndarray,
) -> plt.Figure:
    """Display SQUARE3x3 residual maps for Y, Cr, Cb channels.

    Annotates mean and std of each residual.

    Parameters
    ----------
    ycbcr_npy : np.ndarray
        ``(256, 256, 3)`` YCrCb array.

    Returns
    -------
    matplotlib.figure.Figure
    """
    Y, Cr, Cb = _validate_ycrcb_array(ycbcr_npy)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (channel, name) in zip(
        axes, [(Y, "Y"), (Cr, "Cr"), (Cb, "Cb")]
    ):
        R = _compute_residual_sq3(channel)
        r_mean = float(np.mean(R))
        r_std = float(np.std(R))

        vmax = max(abs(float(R.min())), abs(float(R.max())), 1e-6)
        im = ax.imshow(R, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(
            f"R_{name} ({R.shape[0]}×{R.shape[1]})\n"
            f"mean={r_mean:.3e}  std={r_std:.3e}"
        )
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("SQUARE3x3 Residual Maps (valid conv)", y=1.02)
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────
# 3. Spatial SNR components
# ────────────────────────────────────────────────────────────────────────

def plot_spatial_snr_components(
    ycbcr_npy: np.ndarray,
    *,
    bins: int = 80,
) -> plt.Figure:
    """Display |R_Y| distribution on edge vs flat zones.

    Annotates ``V_edge``, ``V_flat``, and ``spatial_snr_ratio``.

    Parameters
    ----------
    ycbcr_npy : np.ndarray
        ``(256, 256, 3)`` YCrCb array.
    bins : int
        Number of histogram bins.

    Returns
    -------
    matplotlib.figure.Figure
    """
    Y, _Cr, _Cb = _validate_ycrcb_array(ycbcr_npy)

    G = _compute_gradient_valid(Y)
    R_Y = _compute_residual_sq3(Y)
    abs_R_Y = np.abs(R_Y)

    T_edge = max(float(np.percentile(G, 90)), MIN_GRAD_EDGE)
    T_flat = min(float(np.percentile(G, 30)), MAX_GRAD_FLAT)

    M_edge = G >= T_edge
    M_flat = G <= T_flat

    snr = _extract_spatial_snr_ratio(R_Y, M_edge, M_flat)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Edge zone histogram
    if np.any(M_edge):
        edge_vals = abs_R_Y[M_edge]
        V_edge = float(np.mean(edge_vals))
        axes[0].hist(
            edge_vals.ravel(), bins=bins, color="coral",
            edgecolor="none", alpha=0.8,
        )
        axes[0].axvline(V_edge, color="red", linewidth=1.5, linestyle="--")
        axes[0].set_title(
            f"Edge zone |R_Y|  (n={int(np.sum(M_edge))})\n"
            f"V_edge = {V_edge:.4e}"
        )
    else:
        axes[0].set_title("Edge zone — EMPTY (mask has 0 pixels)")
    axes[0].set_xlabel("|R_Y|")
    axes[0].set_ylabel("Count")

    # Flat zone histogram
    if np.any(M_flat):
        flat_vals = abs_R_Y[M_flat]
        V_flat = float(np.mean(flat_vals))
        axes[1].hist(
            flat_vals.ravel(), bins=bins, color="steelblue",
            edgecolor="none", alpha=0.8,
        )
        axes[1].axvline(V_flat, color="navy", linewidth=1.5, linestyle="--")
        axes[1].set_title(
            f"Flat zone |R_Y|  (n={int(np.sum(M_flat))})\n"
            f"V_flat = {V_flat:.4e}"
        )
    else:
        axes[1].set_title("Flat zone — EMPTY (mask has 0 pixels)")
    axes[1].set_xlabel("|R_Y|")
    axes[1].set_ylabel("Count")

    snr_str = f"{snr:.6f}" if np.isfinite(snr) else "NaN"
    fig.suptitle(
        f"Spatial SNR Components — ratio = {snr_str}", y=1.02,
    )
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────
# 4. Cross-noise ratio components
# ────────────────────────────────────────────────────────────────────────

def plot_cross_noise_ratio_components(
    ycbcr_npy: np.ndarray,
    *,
    bins: int = 80,
) -> plt.Figure:
    """Display histograms of |R_Y| and |R_Cb| with noise annotations.

    Parameters
    ----------
    ycbcr_npy : np.ndarray
        ``(256, 256, 3)`` YCrCb array.
    bins : int
        Number of histogram bins.

    Returns
    -------
    matplotlib.figure.Figure
    """
    Y, _Cr, Cb = _validate_ycrcb_array(ycbcr_npy)

    R_Y = _compute_residual_sq3(Y)
    R_Cb = _compute_residual_sq3(Cb)

    Noise_Y = float(np.mean(np.abs(R_Y)))
    Noise_Cb = float(np.mean(np.abs(R_Cb)))
    ratio = _extract_cross_noise_ratio(R_Y, R_Cb)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # |R_Y| histogram
    axes[0].hist(
        np.abs(R_Y).ravel(), bins=bins, color="steelblue",
        edgecolor="none", alpha=0.8,
    )
    axes[0].axvline(Noise_Y, color="red", linewidth=1.5, linestyle="--")
    axes[0].set_title(f"|R_Y|  —  Noise_Y = {Noise_Y:.4e}")
    axes[0].set_xlabel("|R_Y|")
    axes[0].set_ylabel("Count")

    # |R_Cb| histogram
    axes[1].hist(
        np.abs(R_Cb).ravel(), bins=bins, color="olivedrab",
        edgecolor="none", alpha=0.8,
    )
    axes[1].axvline(Noise_Cb, color="red", linewidth=1.5, linestyle="--")
    axes[1].set_title(f"|R_Cb|  —  Noise_Cb = {Noise_Cb:.4e}")
    axes[1].set_xlabel("|R_Cb|")
    axes[1].set_ylabel("Count")

    ratio_str = f"{ratio:.6f}" if np.isfinite(ratio) else "NaN"
    fig.suptitle(
        f"Cross Noise Ratio — Noise_Y / Noise_Cb = {ratio_str}\n"
        f"(Red flag: may confound with chroma 4:2:0 history)",
        y=1.05,
    )
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────
# 5. Residual distribution stats (skew / kurtosis)
# ────────────────────────────────────────────────────────────────────────

def plot_residual_distribution_stats(
    ycbcr_npy: np.ndarray,
    *,
    bins: int = 100,
) -> plt.Figure:
    """Histograms of Y/Cr/Cb residuals with skew/kurt annotations.

    Parameters
    ----------
    ycbcr_npy : np.ndarray
        ``(256, 256, 3)`` YCrCb array.
    bins : int
        Number of histogram bins.

    Returns
    -------
    matplotlib.figure.Figure
    """
    Y, Cr, Cb = _validate_ycrcb_array(ycbcr_npy)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (channel, name) in zip(
        axes, [(Y, "Y"), (Cr, "Cr"), (Cb, "Cb")]
    ):
        R = _compute_residual_sq3(channel)
        skew, kurt = _compute_skew_kurt_safe(R)

        ax.hist(R.ravel(), bins=bins, color="steelblue", edgecolor="none")
        ax.axvline(0, color="red", linewidth=0.8, linestyle="--")

        skew_str = f"{skew:.4f}" if np.isfinite(skew) else "NaN"
        kurt_str = f"{kurt:.4f}" if np.isfinite(kurt) else "NaN"
        ax.set_title(
            f"R_{name} residual\n"
            f"skew={skew_str}  kurt={kurt_str}"
        )
        ax.set_xlabel("Residual value")
        ax.set_ylabel("Count")

    fig.suptitle(
        "Residual Distribution Stats (empirical — not guaranteed Gaussian)",
        y=1.02,
    )
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────
# 6. NaN case inspector
# ────────────────────────────────────────────────────────────────────────

def plot_nan_cases(
    ycbcr_npy: np.ndarray,
) -> plt.Figure:
    """Visualise which features are NaN and why.

    Top row: edge/flat mask with pixel counts (spatial_snr_ratio guard).
    Bottom-left: Noise_Cb bar (cross_noise_ratio guard).
    Bottom-right: per-channel sigma bars (skew/kurt guard).

    Parameters
    ----------
    ycbcr_npy : np.ndarray
        ``(256, 256, 3)`` YCrCb array.

    Returns
    -------
    matplotlib.figure.Figure
    """
    audit = extract_spatial_audit(ycbcr_npy)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # ── Top-left: edge mask status ───────────────────────────────────
    edge_count = audit["edge_pixel_count"]
    flat_count = audit["flat_pixel_count"]
    bars = axes[0, 0].bar(
        ["M_edge", "M_flat"],
        [edge_count, flat_count],
        color=["coral", "steelblue"],
    )
    for bar, count in zip(bars, [edge_count, flat_count]):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center", va="bottom", fontsize=10,
        )
    snr_nan = audit["spatial_snr_is_nan"]
    axes[0, 0].set_title(
        f"Edge / Flat pixel counts\n"
        f"spatial_snr_ratio = {'NaN' if snr_nan else 'finite'}"
    )
    axes[0, 0].set_ylabel("Pixel count")

    # ── Top-right: thresholds vs percentiles ─────────────────────────
    Y, _Cr, _Cb = _validate_ycrcb_array(ycbcr_npy)
    G = _compute_gradient_valid(Y)
    p90 = float(np.percentile(G, 90))
    p30 = float(np.percentile(G, 30))
    axes[0, 1].bar(
        ["P90(G)", "MIN_GRAD_EDGE", "P30(G)", "MAX_GRAD_FLAT"],
        [p90, MIN_GRAD_EDGE, p30, MAX_GRAD_FLAT],
        color=["coral", "darkred", "steelblue", "navy"],
    )
    axes[0, 1].set_title(
        f"Thresholds — T_edge={audit['T_edge']:.1f}, T_flat={audit['T_flat']:.1f}"
    )
    axes[0, 1].set_ylabel("Gradient value")

    # ── Bottom-left: cross noise guard ───────────────────────────────
    noise_y = audit["Noise_Y"]
    noise_cb = audit["Noise_Cb"]
    bars2 = axes[1, 0].bar(
        ["Noise_Y", "Noise_Cb"],
        [noise_y, noise_cb],
        color=["steelblue", "olivedrab"],
    )
    axes[1, 0].axhline(1e-6, color="red", linewidth=1.0, linestyle="--",
                        label="guard = 1e-6")
    cn_nan = audit["cross_noise_is_nan"]
    axes[1, 0].set_title(
        f"Cross Noise Guard\n"
        f"cross_noise_ratio = {'NaN' if cn_nan else 'finite'}"
    )
    axes[1, 0].set_ylabel("Mean |residual|")
    axes[1, 0].legend(fontsize=8)

    # ── Bottom-right: per-channel sigma (skew/kurt guard) ────────────
    sigmas = [audit["sigma_y"], audit["sigma_cr"], audit["sigma_cb"]]
    colors = ["steelblue", "coral", "olivedrab"]
    bars3 = axes[1, 1].bar(["σ_Y", "σ_Cr", "σ_Cb"], sigmas, color=colors)
    axes[1, 1].axhline(
        SIGMA_GUARD, color="red", linewidth=1.0, linestyle="--",
        label=f"guard = {SIGMA_GUARD}",
    )
    axes[1, 1].set_title(
        f"Residual σ per channel\n"
        f"skew/kurt NaN count = {audit['skew_nan_count']}"
    )
    axes[1, 1].set_ylabel("Standard deviation")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("NaN Guard Inspector", y=1.02)
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────
# 7. Feature summary for notebook display
# ────────────────────────────────────────────────────────────────────────

def summarize_spatial_features(
    ycbcr_npy: np.ndarray,
) -> dict[str, Any]:
    """Return a combined feature + audit summary for notebook display.

    Calls both ``extract_spatial_features`` and ``extract_spatial_audit``
    and merges results (audit includes features, but the clean public API
    values take precedence).

    Parameters
    ----------
    ycbcr_npy : np.ndarray
        ``(256, 256, 3)`` YCrCb array.

    Returns
    -------
    dict
        All 8 features plus audit diagnostics.
    """
    feats = extract_spatial_features(ycbcr_npy)
    audit = extract_spatial_audit(ycbcr_npy)
    audit.update(feats)
    return audit


# ────────────────────────────────────────────────────────────────────────
# 8. Side-by-side comparison of two images
# ────────────────────────────────────────────────────────────────────────

def compare_spatial_features(
    ycbcr_a: np.ndarray,
    ycbcr_b: np.ndarray,
    labels: tuple[str, str] = ("A", "B"),
) -> plt.Figure:
    """Side-by-side comparison of spatial features for two images.

    Row 0: edge/flat masks.
    Row 1: spatial_snr_ratio + cross_noise_ratio bar comparison.
    Row 2: skew/kurt bar comparison per channel.

    Parameters
    ----------
    ycbcr_a, ycbcr_b : np.ndarray
        ``(256, 256, 3)`` YCrCb arrays.
    labels : pair of str
        Display labels for the two images.

    Returns
    -------
    matplotlib.figure.Figure
    """
    audit_a = extract_spatial_audit(ycbcr_a)
    audit_b = extract_spatial_audit(ycbcr_b)
    feats_a = extract_spatial_features(ycbcr_a)
    feats_b = extract_spatial_features(ycbcr_b)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    # ── Row 0: edge/flat masks for each image ────────────────────────
    for col, (ycbcr, lbl, audit) in enumerate([
        (ycbcr_a, labels[0], audit_a),
        (ycbcr_b, labels[1], audit_b),
    ]):
        Y, _, _ = _validate_ycrcb_array(ycbcr)
        G = _compute_gradient_valid(Y)
        M_edge = G >= audit["T_edge"]
        M_flat = G <= audit["T_flat"]

        # Composite: 0=neither, 1=flat, 2=edge
        composite = np.zeros_like(G, dtype=np.uint8)
        composite[M_flat] = 1
        composite[M_edge] = 2
        axes[0, col].imshow(composite, cmap="RdYlBu", vmin=0, vmax=2)
        axes[0, col].set_title(
            f"{lbl} — edge={audit['edge_pixel_count']}, "
            f"flat={audit['flat_pixel_count']}"
        )

    # ── Row 1: ratio features bar comparison ─────────────────────────
    ratio_keys = ["spatial_snr_ratio", "cross_noise_ratio"]
    x = np.arange(len(ratio_keys))
    w = 0.35
    vals_a = [feats_a[k] for k in ratio_keys]
    vals_b = [feats_b[k] for k in ratio_keys]

    # Replace NaN with 0 for display, mark with hatching
    def _safe_bar(val: float) -> float:
        return 0.0 if np.isnan(val) else val

    bars_a = axes[1, 0].bar(
        x - w / 2,
        [_safe_bar(v) for v in vals_a],
        w, label=labels[0], color="steelblue",
    )
    bars_b = axes[1, 0].bar(
        x + w / 2,
        [_safe_bar(v) for v in vals_b],
        w, label=labels[1], color="coral",
    )
    # Annotate NaN cases
    for i, (va, vb) in enumerate(zip(vals_a, vals_b)):
        if np.isnan(va):
            axes[1, 0].text(i - w / 2, 0, "NaN", ha="center", va="bottom",
                            fontsize=8, color="red")
        if np.isnan(vb):
            axes[1, 0].text(i + w / 2, 0, "NaN", ha="center", va="bottom",
                            fontsize=8, color="red")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(["snr_ratio", "cross_noise"])
    axes[1, 0].set_title("Ratio Features")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].axis("off")  # empty cell

    # ── Row 2: skew/kurt comparison ──────────────────────────────────
    channels = ["y", "cr", "cb"]
    skew_a = [feats_a[f"skew_noise_{c}"] for c in channels]
    skew_b = [feats_b[f"skew_noise_{c}"] for c in channels]
    kurt_a = [feats_a[f"kurt_noise_{c}"] for c in channels]
    kurt_b = [feats_b[f"kurt_noise_{c}"] for c in channels]

    x3 = np.arange(len(channels))

    # Skewness
    axes[2, 0].bar(
        x3 - w / 2,
        [_safe_bar(v) for v in skew_a],
        w, label=labels[0], color="steelblue",
    )
    axes[2, 0].bar(
        x3 + w / 2,
        [_safe_bar(v) for v in skew_b],
        w, label=labels[1], color="coral",
    )
    for i, (va, vb) in enumerate(zip(skew_a, skew_b)):
        if np.isnan(va):
            axes[2, 0].text(i - w / 2, 0, "NaN", ha="center", va="bottom",
                            fontsize=8, color="red")
        if np.isnan(vb):
            axes[2, 0].text(i + w / 2, 0, "NaN", ha="center", va="bottom",
                            fontsize=8, color="red")
    axes[2, 0].set_xticks(x3)
    axes[2, 0].set_xticklabels(channels)
    axes[2, 0].set_title("Skewness")
    axes[2, 0].legend(fontsize=8)

    # Kurtosis
    axes[2, 1].bar(
        x3 - w / 2,
        [_safe_bar(v) for v in kurt_a],
        w, label=labels[0], color="steelblue",
    )
    axes[2, 1].bar(
        x3 + w / 2,
        [_safe_bar(v) for v in kurt_b],
        w, label=labels[1], color="coral",
    )
    for i, (va, vb) in enumerate(zip(kurt_a, kurt_b)):
        if np.isnan(va):
            axes[2, 1].text(i - w / 2, 0, "NaN", ha="center", va="bottom",
                            fontsize=8, color="red")
        if np.isnan(vb):
            axes[2, 1].text(i + w / 2, 0, "NaN", ha="center", va="bottom",
                            fontsize=8, color="red")
    axes[2, 1].set_xticks(x3)
    axes[2, 1].set_xticklabels(channels)
    axes[2, 1].set_title("Excess Kurtosis")
    axes[2, 1].legend(fontsize=8)

    fig.suptitle("Spatial Feature Comparison", y=1.02)
    fig.tight_layout()
    return fig
