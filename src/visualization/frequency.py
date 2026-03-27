"""
src/visualization/frequency.py
====================
Visualisation helpers for Group 1 frequency-domain features.

All plotting uses **matplotlib only** (no seaborn).  Every function reuses
the same core math from ``src.feature_extraction.frequency`` to guarantee consistency
between numeric extraction and visual debugging.

Typical notebook usage::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path("..").resolve()))

    import numpy as np
    from src.visualization.frequency import (
        plot_y_channel,
        plot_log_power_spectrum,
        plot_ring_profile,
        plot_loglog_power_decay_fit,
        plot_dct_midband_distribution,
        compare_frequency_features,
        summarize_frequency_features,
    )

    arr = np.load("../data/processed/ADM/ai/0_adm_153.npy")
    plot_ring_profile(arr)
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dctn

from src.feature_extraction.frequency import (
    # Constants
    CROP_SIZE,
    DCT_BLOCK,
    DCT_ZZ_END,
    DCT_ZZ_START,
    EPS,
    FEATURE_KEYS,
    FRS_R_MAX,
    FRS_R_MIN,
    N_BLOCKS,
    NOISE_FLOOR,
    PS_R_MAX,
    PS_R_MIN,
    # Precomputed tables
    _R_INT_MAX,
    _ZZ_COLS,
    _ZZ_ROWS,
    # Private helpers (reused to guarantee identical math)
    _compute_power_spectrum,
    _get_dct_midband_squared,
    _ring_mean,
    _safe_skewness,
    _validate_ycrcb_array,
    # Public API
    extract_frequency_features,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────

def _get_Y_P_ring(
    ycbcr_npy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate, compute power spectrum and ring means in one pass."""
    Y = _validate_ycrcb_array(ycbcr_npy)
    P = _compute_power_spectrum(Y)
    ring = _ring_mean(P)
    return Y, P, ring


# ────────────────────────────────────────────────────────────────────────
# Public visualisation functions
# ────────────────────────────────────────────────────────────────────────

def plot_y_channel(
    ycbcr_npy: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title: str = "Y channel",
) -> plt.Axes:
    """Display the luminance (Y) channel as a grayscale image."""
    Y = _validate_ycrcb_array(ycbcr_npy)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(Y, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.set_axis_off()
    return ax


def plot_log_power_spectrum(
    ycbcr_npy: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title: str = "log10 power spectrum (fftshift)",
) -> plt.Axes:
    """Display log10(P + eps) as a 2-D heatmap."""
    Y = _validate_ycrcb_array(ycbcr_npy)
    P = _compute_power_spectrum(Y)
    log_P = np.log10(P + EPS)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(log_P, cmap="inferno", origin="upper")
    ax.set_title(title)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    plt.colorbar(im, ax=ax, label="log10(P)")
    return ax


def plot_ring_profile(
    ycbcr_npy: np.ndarray,
    *,
    r_min: int = 1,
    r_max: int = 128,
    show_frs_band: bool = True,
    show_ps_band: bool = True,
    ax: plt.Axes | None = None,
    title: str = "Azimuthal mean power E[r]",
) -> plt.Axes:
    """Plot the radial power profile with optional band highlights.

    Parameters
    ----------
    ycbcr_npy : np.ndarray
        (256, 256, 3) YCrCb array.
    r_min, r_max : int
        Radial range to display.
    show_frs_band : bool
        Shade the FRS mid-band [8, 32].
    show_ps_band : bool
        Shade the PS fit band [20, 64].
    """
    _, _, ring = _get_Y_P_ring(ycbcr_npy)
    r_end = min(r_max + 1, ring.size)
    r_vals = np.arange(r_min, r_end)
    E_vals = ring[r_min:r_end]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.semilogy(r_vals, E_vals + EPS, linewidth=1.0, color="black")

    if show_frs_band:
        ax.axvspan(
            FRS_R_MIN, FRS_R_MAX, alpha=0.15, color="blue",
            label=f"FRS band [{FRS_R_MIN}, {FRS_R_MAX}]",
        )
    if show_ps_band:
        ax.axvspan(
            PS_R_MIN, PS_R_MAX, alpha=0.15, color="red",
            label=f"PS fit band [{PS_R_MIN}, {PS_R_MAX}]",
        )

    ax.set_xlabel("Radius r (cycles)")
    ax.set_ylabel("E[r] (mean power)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_loglog_power_decay_fit(
    ycbcr_npy: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title: str = "1D power spectrum log-log fit",
) -> plt.Axes:
    """Log-log plot of c(r) with fitted line and annotated alpha / dev_var.

    Shows the full radial profile in gray for context and overlays the
    fit band [20, 64] with the OLS line.
    """
    _, _, ring = _get_Y_P_ring(ycbcr_npy)

    # Full profile for context (r=1..max)
    r_end_full = min(_R_INT_MAX + 1, ring.size)
    r_all = np.arange(1, r_end_full, dtype=np.float64)
    c_all = ring[1:r_end_full]

    # Fit band
    r_fit = np.arange(PS_R_MIN, PS_R_MAX + 1, dtype=np.float64)
    c_fit = ring[PS_R_MIN : PS_R_MAX + 1].astype(np.float64)
    log_r_fit = np.log(r_fit)
    log_c_fit = np.log(c_fit + NOISE_FLOOR)

    slope, intercept = np.polyfit(log_r_fit, log_c_fit, 1)
    alpha = -slope
    fitted_line = slope * log_r_fit + intercept
    dev_var = float(np.var(log_c_fit - fitted_line))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        np.log(r_all), np.log(c_all + NOISE_FLOOR),
        linewidth=0.8, color="gray", alpha=0.6, label="full profile",
    )
    ax.plot(
        log_r_fit, log_c_fit,
        linewidth=1.2, color="black",
        label=f"data r in [{PS_R_MIN}, {PS_R_MAX}]",
    )
    ax.plot(
        log_r_fit, fitted_line,
        linewidth=1.5, linestyle="--", color="red",
        label=f"fit: alpha={alpha:.3f}",
    )

    ax.set_xlabel("log r")
    ax.set_ylabel("log c(r)")
    ax.set_title(title)
    ax.annotate(
        f"alpha = {alpha:.4f}\ndev_var = {dev_var:.2e}",
        xy=(0.95, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_dct_midband_distribution(
    ycbcr_npy: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    n_bins: int = 80,
    title: str = "DCT mid-band A = coeff^2 distribution",
) -> plt.Axes:
    """Histogram of squared mid-band DCT coefficients with stat annotations.

    Uses the identical DCT computation from the core extraction module.
    """
    Y = _validate_ycrcb_array(ycbcr_npy)
    A = _get_dct_midband_squared(Y)

    dct_mean = float(np.mean(A))
    dct_var = float(np.var(A))
    dct_skew = _safe_skewness(A)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    # Clip for display (A can have a long tail)
    clip_hi = float(np.percentile(A, 99.5))
    ax.hist(
        A[A <= clip_hi], bins=n_bins, density=True,
        color="steelblue", edgecolor="white", linewidth=0.3,
    )
    ax.set_xlabel("A = coeff^2")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.annotate(
        f"mean = {dct_mean:.4e}\nvar  = {dct_var:.4e}\nskew = {dct_skew:.4f}",
        xy=(0.95, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )
    ax.grid(True, alpha=0.3)
    return ax


def compare_frequency_features(
    ycbcr_npy_a: np.ndarray,
    ycbcr_npy_b: np.ndarray,
    labels: tuple[str, str] = ("A", "B"),
) -> plt.Figure:
    """Side-by-side comparison of two images across three diagnostic views.

    Subplots:
        1. Ring profile E[r]  (overlay)
        2. Log-log power decay fit  (overlay)
        3. DCT mid-band histogram  (overlay)

    Useful for comparing a real image against a fake one.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#1f77b4", "#d62728"]

    for idx, (arr, lbl) in enumerate(
        [(ycbcr_npy_a, labels[0]), (ycbcr_npy_b, labels[1])]
    ):
        Y, P, ring = _get_Y_P_ring(arr)
        c = colors[idx]

        # ── 1) Ring profile ──────────────────────────────────────────
        r_end = min(129, ring.size)
        r_vals = np.arange(1, r_end)
        axes[0].semilogy(
            r_vals, ring[1:r_end] + EPS,
            linewidth=1.0, color=c, label=lbl,
        )

        # ── 2) Log-log fit ───────────────────────────────────────────
        r_fit = np.arange(PS_R_MIN, PS_R_MAX + 1, dtype=np.float64)
        c_fit = ring[PS_R_MIN : PS_R_MAX + 1].astype(np.float64)
        log_r = np.log(r_fit)
        log_c = np.log(c_fit + NOISE_FLOOR)
        slope, intercept = np.polyfit(log_r, log_c, 1)
        alpha = -slope
        axes[1].plot(
            log_r, log_c, linewidth=1.0, color=c,
            label=f"{lbl} (alpha={alpha:.3f})",
        )
        axes[1].plot(
            log_r, slope * log_r + intercept,
            linewidth=1.2, linestyle="--", color=c, alpha=0.6,
        )

        # ── 3) DCT histogram ─────────────────────────────────────────
        A = _get_dct_midband_squared(Y)
        clip_hi = float(np.percentile(A, 99))
        axes[2].hist(
            A[A <= clip_hi], bins=60, density=True,
            color=c, alpha=0.5, label=lbl, edgecolor="none",
        )

    # Decorate
    axes[0].axvspan(FRS_R_MIN, FRS_R_MAX, alpha=0.1, color="blue")
    axes[0].axvspan(PS_R_MIN, PS_R_MAX, alpha=0.1, color="red")
    axes[0].set_xlabel("r")
    axes[0].set_ylabel("E[r]")
    axes[0].set_title("Ring profile")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("log r")
    axes[1].set_ylabel("log c(r)")
    axes[1].set_title("Log-log power decay")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("A = coeff^2")
    axes[2].set_ylabel("Density")
    axes[2].set_title("DCT mid-band dist")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def summarize_frequency_features(
    ycbcr_npy: np.ndarray,
) -> dict[str, object]:
    """Extract features and return an augmented summary dict.

    Includes all 6 features plus human-readable interpretation hints.
    """
    feats = extract_frequency_features(ycbcr_npy)

    alpha = feats["ps_alpha"]
    if 2.0 <= alpha <= 3.5:
        alpha_note = "within natural-image range"
    elif alpha < 2.0:
        alpha_note = "unusually flat spectrum"
    else:
        alpha_note = "unusually steep spectrum"

    summary: dict[str, object] = {
        **feats,
        "alpha_interpretation": alpha_note,
        "n_dct_blocks": N_BLOCKS * N_BLOCKS,
        "n_midband_coeffs_per_block": DCT_ZZ_END - DCT_ZZ_START + 1,
    }
    return summary


# ────────────────────────────────────────────────────────────────────────
# Smoke test
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    # Ensure src is importable when running directly
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    rng = np.random.default_rng(42)
    dummy = rng.integers(
        0, 256, size=(CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8,
    )

    print("Smoke-test: generating plots for random noise input...")
    plot_y_channel(dummy)
    plot_log_power_spectrum(dummy)
    plot_ring_profile(dummy)
    plot_loglog_power_decay_fit(dummy)
    plot_dct_midband_distribution(dummy)

    summary = summarize_frequency_features(dummy)
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    plt.show()
    print("Done.")
