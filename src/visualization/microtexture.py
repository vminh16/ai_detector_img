"""
src/visualization/microtexture.py
=======================
Group 3 — Visualization & debugging helpers for Micro-Texture Residual
and Chroma LBP features.

All numeric logic is delegated to ``microtexture_features.py`` to
guarantee identical math between feature extraction and visualization.

Dependencies: numpy, matplotlib, src.feature_extraction.microtexture.
"""
from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.feature_extraction.microtexture import (
    # Constants
    CROP_SIZE,
    FEATURE_KEYS,
    LBP_BINS,
    LBP_RADIUS,
    SRM_ENERGY_THRESHOLD,
    # Private helpers (reused to guarantee identical math)
    _convolve_valid,
    _compute_srm_pair,
    _extract_lbp_channel,
    _lbp_codes_radius1_points8,
    _lbp_histogram_stats,
    _LBP_LUT_59,
    _SRM_KERNELS,
    _validate_ycrcb_array,
    # Public API
    extract_microtexture_features,
    extract_microtexture_audit,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# 1. Cr channel visualization
# ────────────────────────────────────────────────────────────────────────

def plot_cr_channel(
    ycbcr_npy: np.ndarray,
    *,
    ax: tuple[plt.Axes, plt.Axes] | None = None,
) -> plt.Figure:
    """Display the raw Cr channel and its centred version (Cr − 128).

    Parameters
    ----------
    ycbcr_npy : np.ndarray
        ``(256, 256, 3)`` YCrCb array.
    ax : pair of Axes, optional
        If given, plot into these axes instead of creating a new figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _Y, Cr, _Cb = _validate_ycrcb_array(ycbcr_npy)
    Cr_centred = Cr - 128.0

    if ax is None:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    else:
        ax0, ax1 = ax
        fig = ax0.figure

    im0 = ax0.imshow(Cr, cmap="gray")
    ax0.set_title("Cr (raw)")
    plt.colorbar(im0, ax=ax0, fraction=0.046)

    vmax = max(abs(float(Cr_centred.min())), abs(float(Cr_centred.max())), 1.0)
    im1 = ax1.imshow(Cr_centred, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax1.set_title("Cr' = Cr − 128")
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────
# 2. SRM residual maps
# ────────────────────────────────────────────────────────────────────────

def plot_srm_residuals(
    ycbcr_npy: np.ndarray,
) -> plt.Figure:
    """Display the 3 SRM residual maps with MAR/Energy annotations.

    Parameters
    ----------
    ycbcr_npy : np.ndarray
        ``(256, 256, 3)`` YCrCb array.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _Y, Cr, _Cb = _validate_ycrcb_array(ycbcr_npy)
    Cr_centred = Cr - 128.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, kernel) in zip(axes, _SRM_KERNELS):
        residual = _convolve_valid(Cr_centred, kernel)
        mar, energy = _compute_srm_pair(residual)

        vmax = max(abs(float(residual.min())), abs(float(residual.max())), 1e-6)
        im = ax.imshow(residual, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(
            f"{name}  ({residual.shape[0]}×{residual.shape[1]})\n"
            f"MAR={mar:.4e}  E={energy:.4e}"
        )
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("SRM Residual Maps (Cr' = Cr − 128, valid conv)", y=1.02)
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────
# 3. SRM residual histograms
# ────────────────────────────────────────────────────────────────────────

def plot_srm_histograms(
    ycbcr_npy: np.ndarray,
    *,
    bins: int = 100,
) -> plt.Figure:
    """Histogram of SRM residual values for the 3 kernels.

    Useful for inspecting zero-mean symmetry and tail behaviour.

    Parameters
    ----------
    ycbcr_npy : np.ndarray
        ``(256, 256, 3)`` YCrCb array.
    bins : int
        Number of histogram bins (default 100).

    Returns
    -------
    matplotlib.figure.Figure
    """
    _Y, Cr, _Cb = _validate_ycrcb_array(ycbcr_npy)
    Cr_centred = Cr - 128.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, kernel) in zip(axes, _SRM_KERNELS):
        residual = _convolve_valid(Cr_centred, kernel)
        mar, energy = _compute_srm_pair(residual)

        ax.hist(residual.ravel(), bins=bins, color="steelblue", edgecolor="none")
        ax.axvline(0, color="red", linewidth=0.8, linestyle="--")
        ax.set_title(f"{name}\nMAR={mar:.3e}  E={energy:.3e}")
        ax.set_xlabel("Residual value")
        ax.set_ylabel("Count")

    fig.suptitle("SRM Residual Distributions", y=1.02)
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────
# 4. LBP code maps
# ────────────────────────────────────────────────────────────────────────

def plot_lbp_maps(
    ycbcr_npy: np.ndarray,
) -> plt.Figure:
    """Display raw LBP code maps and uniform-mapped bin maps for Cr/Cb.

    Parameters
    ----------
    ycbcr_npy : np.ndarray
        ``(256, 256, 3)`` YCrCb array.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _Y, Cr, Cb = _validate_ycrcb_array(ycbcr_npy)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for row, (channel, label) in enumerate([(Cr, "Cr"), (Cb, "Cb")]):
        C_int = np.rint(channel).astype(np.int32)
        ptp = int(C_int.max() - C_int.min())

        if ptp == 0:
            # Sentinel case: no texture — show blank maps
            core_h = CROP_SIZE - 2 * LBP_RADIUS
            core_w = CROP_SIZE - 2 * LBP_RADIUS
            raw_codes = np.zeros((core_h, core_w), dtype=np.uint8)
            mapped = np.zeros((core_h, core_w), dtype=np.int32)
        else:
            raw_codes = _lbp_codes_radius1_points8(C_int)
            mapped = _LBP_LUT_59[raw_codes]

        im0 = axes[row, 0].imshow(raw_codes, cmap="viridis")
        axes[row, 0].set_title(f"{label} — Raw LBP codes [0–255]")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        im1 = axes[row, 1].imshow(mapped, cmap="viridis", vmin=0, vmax=58)
        axes[row, 1].set_title(f"{label} — Uniform-mapped bins [0–58]")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────
# 5. LBP histograms
# ────────────────────────────────────────────────────────────────────────

def plot_lbp_histograms(
    ycbcr_npy: np.ndarray,
) -> plt.Figure:
    """59-bin LBP histograms for Cr and Cb with annotations.

    The non-uniform bin (index 58) is highlighted in red.  Non-uniformity
    ratio and entropy are annotated on each subplot.

    Parameters
    ----------
    ycbcr_npy : np.ndarray
        ``(256, 256, 3)`` YCrCb array.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _Y, Cr, Cb = _validate_ycrcb_array(ycbcr_npy)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (channel, label) in zip(axes, [(Cr, "Cr"), (Cb, "Cb")]):
        C_int = np.rint(channel).astype(np.int32)
        ptp = int(C_int.max() - C_int.min())

        if ptp == 0:
            nu_ratio, entropy = 0.0, 0.0
            prob = np.zeros(LBP_BINS, dtype=np.float64)
        else:
            codes = _lbp_codes_radius1_points8(C_int)
            nu_ratio, entropy = _lbp_histogram_stats(codes)
            mapped = _LBP_LUT_59[codes.ravel()]
            counts = np.bincount(mapped, minlength=LBP_BINS).astype(np.float64)
            total = counts.sum()
            prob = counts / total if total > 0 else counts

        colors = ["steelblue"] * (LBP_BINS - 1) + ["crimson"]
        ax.bar(range(LBP_BINS), prob, color=colors, edgecolor="none")
        ax.set_xlabel("Bin index (58 = non-uniform)")
        ax.set_ylabel("Probability")
        ax.set_title(
            f"{label} LBP Histogram (59 bins)\n"
            f"Non-uniform ratio = {nu_ratio:.4f}  |  "
            f"Entropy = {entropy:.4f} bits"
        )

    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────
# 6. Comparison of two images
# ────────────────────────────────────────────────────────────────────────

def compare_microtexture_features(
    ycbcr_a: np.ndarray,
    ycbcr_b: np.ndarray,
    labels: tuple[str, str] = ("A", "B"),
) -> plt.Figure:
    """Side-by-side comparison of micro-texture features for two images.

    Top row: SRM residual energy bar chart.
    Bottom row: LBP histogram overlay for Cr and Cb.

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
    feats_a = extract_microtexture_features(ycbcr_a)
    feats_b = extract_microtexture_features(ycbcr_b)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Top-left: SRM MAR comparison ─────────────────────────────────
    srm_names = ["square3", "edge3", "square5"]
    mar_a = [feats_a[f"srm_{n}_mar_cr"] for n in srm_names]
    mar_b = [feats_b[f"srm_{n}_mar_cr"] for n in srm_names]
    x = np.arange(len(srm_names))
    w = 0.35
    axes[0, 0].bar(x - w / 2, mar_a, w, label=labels[0], color="steelblue")
    axes[0, 0].bar(x + w / 2, mar_b, w, label=labels[1], color="coral")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(srm_names)
    axes[0, 0].set_ylabel("MAR")
    axes[0, 0].set_title("SRM MAR (Cr)")
    axes[0, 0].legend()

    # ── Top-right: SRM Energy comparison ─────────────────────────────
    eng_a = [feats_a[f"srm_{n}_energy_cr"] for n in srm_names]
    eng_b = [feats_b[f"srm_{n}_energy_cr"] for n in srm_names]
    axes[0, 1].bar(x - w / 2, eng_a, w, label=labels[0], color="steelblue")
    axes[0, 1].bar(x + w / 2, eng_b, w, label=labels[1], color="coral")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(srm_names)
    axes[0, 1].set_ylabel("Energy")
    axes[0, 1].set_title("SRM Energy (Cr)")
    axes[0, 1].legend()

    # ── Bottom: LBP histogram overlays (Cr, Cb) ─────────────────────
    _Ya, Cr_a, Cb_a = _validate_ycrcb_array(ycbcr_a)
    _Yb, Cr_b, Cb_b = _validate_ycrcb_array(ycbcr_b)

    for col, (ch_a, ch_b, ch_label) in enumerate([
        (Cr_a, Cr_b, "Cr"),
        (Cb_a, Cb_b, "Cb"),
    ]):
        ax = axes[1, col]
        for ch, lbl, color in [(ch_a, labels[0], "steelblue"), (ch_b, labels[1], "coral")]:
            C_int = np.rint(ch).astype(np.int32)
            if int(C_int.max() - C_int.min()) == 0:
                prob = np.zeros(LBP_BINS, dtype=np.float64)
            else:
                codes = _lbp_codes_radius1_points8(C_int)
                mapped = _LBP_LUT_59[codes.ravel()]
                counts = np.bincount(mapped, minlength=LBP_BINS).astype(np.float64)
                total = counts.sum()
                prob = counts / total if total > 0 else counts
            ax.bar(
                np.arange(LBP_BINS) + (0.2 if color == "coral" else -0.2),
                prob, 0.4, label=lbl, color=color, alpha=0.7,
            )
        ax.set_xlabel("Bin (58 = non-uniform)")
        ax.set_ylabel("Probability")
        ax.set_title(f"LBP Histogram — {ch_label}")
        ax.legend()

    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────
# 7. Feature summary for notebook display
# ────────────────────────────────────────────────────────────────────────

def summarize_microtexture_features(
    ycbcr_npy: np.ndarray,
) -> dict[str, Any]:
    """Return a combined feature + audit summary, ready for notebook display.

    Calls both ``extract_microtexture_features`` and
    ``extract_microtexture_audit`` and merges results.

    Parameters
    ----------
    ycbcr_npy : np.ndarray
        ``(256, 256, 3)`` YCrCb array.

    Returns
    -------
    dict
        All 10 features plus audit diagnostics.
    """
    feats = extract_microtexture_features(ycbcr_npy)
    audit = extract_microtexture_audit(ycbcr_npy)
    # Audit already includes feats, but prefer the safety-net-cleaned feats
    audit.update(feats)
    return audit


# ────────────────────────────────────────────────────────────────────────
# 8. Sentinel case inspector
# ────────────────────────────────────────────────────────────────────────

def plot_sentinel_cases(
    ycbcr_npy: np.ndarray,
) -> plt.Figure:
    """Visualise which SRM kernels and LBP channels hit sentinel paths.

    Produces a 2-row figure:
    - Top: per-kernel energy with threshold line; bars colored red if
      sentinel was triggered.
    - Bottom: Cr and Cb ptp values with sentinel annotation.

    Parameters
    ----------
    ycbcr_npy : np.ndarray
        ``(256, 256, 3)`` YCrCb array.

    Returns
    -------
    matplotlib.figure.Figure
    """
    audit = extract_microtexture_audit(ycbcr_npy)
    _Y, Cr, Cb = _validate_ycrcb_array(ycbcr_npy)
    Cr_centred = Cr - 128.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Left: SRM energy per kernel vs threshold ─────────────────────
    kernel_names = [name for name, _ in _SRM_KERNELS]
    energies = []
    for name, kernel in _SRM_KERNELS:
        residual = _convolve_valid(Cr_centred, kernel)
        energies.append(float(np.mean(residual ** 2)))

    colors = ["crimson" if e < SRM_ENERGY_THRESHOLD else "steelblue" for e in energies]
    axes[0].bar(kernel_names, energies, color=colors)
    axes[0].axhline(SRM_ENERGY_THRESHOLD, color="red", linewidth=1, linestyle="--",
                     label=f"Threshold = {SRM_ENERGY_THRESHOLD:.0e}")
    axes[0].set_ylabel("Energy (mean R²)")
    axes[0].set_title("SRM Energy vs Sentinel Threshold")
    axes[0].set_yscale("log")
    axes[0].legend()

    # ── Right: LBP ptp sentinel status ───────────────────────────────
    cr_ptp = audit["cr_ptp"]
    cb_ptp = audit["cb_ptp"]
    bar_colors = [
        "crimson" if cr_ptp == 0 else "steelblue",
        "crimson" if cb_ptp == 0 else "steelblue",
    ]
    axes[1].bar(["Cr ptp", "Cb ptp"], [cr_ptp, cb_ptp], color=bar_colors)
    axes[1].set_ylabel("Peak-to-Peak (int32)")
    axes[1].set_title("LBP Sentinel Status (ptp == 0 → sentinel)")
    for i, (val, sent) in enumerate([(cr_ptp, audit["lbp_cr_sentinel_used"]),
                                      (cb_ptp, audit["lbp_cb_sentinel_used"])]):
        axes[1].text(i, val + 0.5, "SENTINEL" if sent else "OK",
                     ha="center", fontweight="bold",
                     color="crimson" if sent else "green")

    fig.tight_layout()
    return fig
