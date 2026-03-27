"""
src/visualization/color.py
================
Visualisation helpers for Group 2 color cross-correlation features.

All plotting uses **matplotlib only** (no seaborn).  Every function reuses
the same core math from ``src.feature_extraction.color`` to guarantee consistency
between numeric extraction and visual debugging.

Typical notebook usage::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path("..").resolve()))

    import numpy as np
    from src.visualization.color import (
        plot_chroma_magnitude,
        plot_circular_variance_map,
        plot_channel_scatter,
        plot_glcm_heatmap,
        plot_glcm_quantised_cr,
        compare_color_features,
        summarize_color_features,
    )

    arr = np.load("../data/processed/ADM/ai/0_adm_153.npy")
    plot_chroma_magnitude(arr)
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

from src.feature_extraction.color import (
    # Constants
    CHROMA_PERCENTILE,
    CROP_SIZE,
    EPS,
    FEATURE_KEYS,
    GLCM_DISTANCE,
    GLCM_LEVELS,
    MIN_SURVIVING_WINDOWS,
    MIN_VALID_PIXELS,
    MIN_WINDOW_CHROMA_FLOOR,
    NEAR_ZERO_CHROMA_CEILING,
    PIXEL_NOISE_FLOOR,
    WINDOW_SIZE,
    # Private helpers (reused to guarantee identical math)
    _compute_glcm_single,
    _extract_glcm_features,
    _extract_local_color_inconsistency,
    _extract_pearson_correlations,
    _extract_energy_ratio,
    _GLCM_OFFSETS,
    _glcm_metrics,
    _quantise_cr_for_glcm,
    _validate_ycrcb_array,
    # Public API
    extract_color_features,
)
from scipy.ndimage import uniform_filter

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# Internal helpers (vectorised LCI map)
# ────────────────────────────────────────────────────────────────────────

def _compute_lci_maps(
    Cr: np.ndarray,
    Cb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised computation of (var_circ_map, mbar_map) for visualisation.

    Mirrors the math in ``_extract_local_color_inconsistency`` but returns
    the full 2-D maps instead of the scalar mean.
    """
    Cr_c = Cr - 128.0
    Cb_c = Cb - 128.0
    M = np.sqrt(Cr_c ** 2 + Cb_c ** 2)

    half_w = WINDOW_SIZE // 2
    win_area = float(WINDOW_SIZE * WINDOW_SIZE)

    valid = (M > PIXEL_NOISE_FLOOR).astype(np.float64)
    M_safe = M + EPS
    U_pixel = (Cr_c / M_safe) * valid
    V_pixel = (Cb_c / M_safe) * valid

    sum_U = uniform_filter(U_pixel, size=WINDOW_SIZE, mode="constant") * win_area
    sum_V = uniform_filter(V_pixel, size=WINDOW_SIZE, mode="constant") * win_area
    n_valid = uniform_filter(valid, size=WINDOW_SIZE, mode="constant") * win_area
    mbar_full = uniform_filter(M, size=WINDOW_SIZE, mode="constant")

    sl = slice(half_w, -half_w)
    sum_U = sum_U[sl, sl]
    sum_V = sum_V[sl, sl]
    n_valid_map = n_valid[sl, sl]
    mbar_map = mbar_full[sl, sl]

    enough = n_valid_map >= MIN_VALID_PIXELS
    n_safe = np.where(enough, n_valid_map, 1.0)
    R_map = np.sqrt(sum_U ** 2 + sum_V ** 2) / n_safe
    var_circ_map = np.where(enough, 1.0 - R_map, 0.0)

    return var_circ_map, mbar_map


# ────────────────────────────────────────────────────────────────────────
# Public visualisation functions
# ────────────────────────────────────────────────────────────────────────

def plot_chroma_magnitude(
    ycbcr_npy: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title: str = "Chroma magnitude M",
) -> plt.Axes:
    """Display M = sqrt((Cr-128)² + (Cb-128)²) as a heatmap."""
    _, Cr, Cb = _validate_ycrcb_array(ycbcr_npy)
    M = np.sqrt((Cr - 128.0) ** 2 + (Cb - 128.0) ** 2)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(M, cmap="magma")
    ax.set_title(title)
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, label="magnitude")
    return ax


def plot_circular_variance_map(
    ycbcr_npy: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title: str = "Circular variance map (9×9 windows)",
) -> plt.Axes:
    """Heatmap of per-window circular variance, with dual-threshold mask."""
    _, Cr, Cb = _validate_ycrcb_array(ycbcr_npy)
    var_circ_map, mbar_map = _compute_lci_maps(Cr, Cb)

    threshold = max(
        float(np.percentile(mbar_map, CHROMA_PERCENTILE)),
        MIN_WINDOW_CHROMA_FLOOR,
    )
    masked = np.where(mbar_map >= threshold, var_circ_map, np.nan)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(masked, cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, label="1 − R (circular variance)")

    lci = _extract_local_color_inconsistency(Cr, Cb)
    ax.annotate(
        f"LCI = {lci:.4f}",
        xy=(0.02, 0.98), xycoords="axes fraction",
        ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )
    return ax


def plot_channel_scatter(
    ycbcr_npy: np.ndarray,
    *,
    n_sample: int = 5000,
    title: str = "Cross-channel scatter (sampled)",
) -> plt.Figure:
    """Scatter plots for the 3 channel pairs with Pearson annotations.

    Subsamples ``n_sample`` pixels for clarity.
    """
    Y, Cr, Cb = _validate_ycrcb_array(ycbcr_npy)
    p_y_cr, p_y_cb, p_cr_cb = _extract_pearson_correlations(Y, Cr, Cb)

    rng = np.random.default_rng(0)
    N = Y.size
    idx = rng.choice(N, size=min(n_sample, N), replace=False)
    y_s, cr_s, cb_s = Y.ravel()[idx], Cr.ravel()[idx], Cb.ravel()[idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [(y_s, cr_s, "Y", "Cr", p_y_cr),
             (y_s, cb_s, "Y", "Cb", p_y_cb),
             (cr_s, cb_s, "Cr", "Cb", p_cr_cb)]

    for ax, (a, b, na, nb, r) in zip(axes, pairs):
        ax.scatter(a, b, s=1, alpha=0.3, color="steelblue", rasterized=True)
        ax.set_xlabel(na)
        ax.set_ylabel(nb)
        ax.set_title(f"{na} vs {nb}")
        ax.annotate(
            f"r = {r:.4f}",
            xy=(0.95, 0.05), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_glcm_heatmap(
    ycbcr_npy: np.ndarray,
    *,
    title: str = "GLCM on Cr (4 angles, D=2)",
) -> plt.Figure:
    """Display the 4 GLCM matrices and their per-angle metrics."""
    _, Cr, _ = _validate_ycrcb_array(ycbcr_npy)
    q = _quantise_cr_for_glcm(Cr)

    angle_labels = ["0°", "45°", "90°", "135°"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for idx, ((dy, dx), lbl) in enumerate(zip(_GLCM_OFFSETS, angle_labels)):
        glcm = _compute_glcm_single(q, dy, dx)
        con, cor, eng, hom = _glcm_metrics(glcm)
        ax = axes[idx]
        im = ax.imshow(glcm, cmap="Blues", vmin=0)
        ax.set_title(f"{lbl}  (dy={dy}, dx={dx})")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.annotate(
            f"Con={con:.3f}\nCor={cor:.3f}\nEng={eng:.4f}\nHom={hom:.3f}",
            xy=(0.98, 0.02), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
        )

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_glcm_quantised_cr(
    ycbcr_npy: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title: str = "Quantised Cr (8 levels)",
) -> plt.Axes:
    """Display the quantised Cr channel used for GLCM computation."""
    _, Cr, _ = _validate_ycrcb_array(ycbcr_npy)
    q = _quantise_cr_for_glcm(Cr)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(q, cmap="coolwarm", vmin=0, vmax=GLCM_LEVELS - 1)
    ax.set_title(title)
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, label="quantisation level", ticks=range(GLCM_LEVELS))
    return ax


def compare_color_features(
    ycbcr_npy_a: np.ndarray,
    ycbcr_npy_b: np.ndarray,
    labels: tuple[str, str] = ("A", "B"),
) -> plt.Figure:
    """Side-by-side comparison of two images across three diagnostic views.

    Subplots:
        1. Chroma magnitude (A vs B)
        2. Circular variance map (A vs B)
        3. Quantised Cr (A vs B)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for row, (arr, lbl) in enumerate(
        [(ycbcr_npy_a, labels[0]), (ycbcr_npy_b, labels[1])]
    ):
        _, Cr, Cb = _validate_ycrcb_array(arr)

        # 1) Chroma magnitude
        M = np.sqrt((Cr - 128.0) ** 2 + (Cb - 128.0) ** 2)
        im0 = axes[row, 0].imshow(M, cmap="magma")
        axes[row, 0].set_title(f"{lbl}: Chroma magnitude")
        axes[row, 0].set_axis_off()
        plt.colorbar(im0, ax=axes[row, 0], shrink=0.8)

        # 2) Circular variance (vectorised)
        Cr_c, Cb_c = Cr - 128.0, Cb - 128.0
        vc, mb = _compute_lci_maps(Cr, Cb)

        thr = max(
            float(np.percentile(mb, CHROMA_PERCENTILE)),
            MIN_WINDOW_CHROMA_FLOOR,
        )
        masked = np.where(mb >= thr, vc, np.nan)
        im1 = axes[row, 1].imshow(masked, cmap="RdYlBu_r", vmin=0, vmax=1)
        axes[row, 1].set_title(f"{lbl}: Circular variance")
        axes[row, 1].set_axis_off()
        plt.colorbar(im1, ax=axes[row, 1], shrink=0.8)

        # 3) Quantised Cr
        q = _quantise_cr_for_glcm(Cr)
        im2 = axes[row, 2].imshow(q, cmap="coolwarm", vmin=0, vmax=GLCM_LEVELS - 1)
        axes[row, 2].set_title(f"{lbl}: Quantised Cr")
        axes[row, 2].set_axis_off()
        plt.colorbar(im2, ax=axes[row, 2], shrink=0.8)

    fig.tight_layout()
    return fig


def summarize_color_features(
    ycbcr_npy: np.ndarray,
) -> dict[str, object]:
    """Extract features and return an augmented summary dict.

    Includes all 9 features plus human-readable interpretation hints.
    """
    feats = extract_color_features(ycbcr_npy)
    Y, Cr, Cb = _validate_ycrcb_array(ycbcr_npy)

    er = feats["energy_ratio_chroma"]
    if er < 0.01:
        er_note = "very low chroma energy (near-monochrome)"
    elif er > 1.0:
        er_note = "chroma energy dominates luminance"
    else:
        er_note = "normal chroma-to-luminance ratio"

    lci = feats["local_color_inconsistency"]
    if lci < 0.01:
        lci_note = "very coherent chroma (or sentinel — check mask)"
    elif lci > 0.3:
        lci_note = "high local colour inconsistency"
    else:
        lci_note = "moderate local colour consistency"

    summary: dict[str, object] = {
        **feats,
        "energy_ratio_interpretation": er_note,
        "lci_interpretation": lci_note,
        "mean_magnitude": float(np.mean(
            np.sqrt((Cr - 128.0) ** 2 + (Cb - 128.0) ** 2)
        )),
    }
    return summary


# ────────────────────────────────────────────────────────────────────────
# Smoke test
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    rng = np.random.default_rng(42)
    dummy = rng.integers(
        0, 256, size=(CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8,
    )

    print("Smoke-test: generating plots for random noise input...")
    plot_chroma_magnitude(dummy)
    plot_circular_variance_map(dummy)
    plot_channel_scatter(dummy)
    plot_glcm_heatmap(dummy)
    plot_glcm_quantised_cr(dummy)

    summary = summarize_color_features(dummy)
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    plt.show()
    print("Done.")
