"""
src/feature_extraction/color.py
=====================
Group 2 — Color Cross-Correlation & Chroma Inconsistency feature extraction
(locked spec).

Extracts 9 features from the Cr / Cb / Y channels of preprocessed 256×256
YCrCb arrays produced by the hardened preprocessing pipeline (reflect padding,
misaligned center crop, deterministic JPEG bottleneck → .npy).

Features
--------
local_color_inconsistency : Weighted circular variance of chroma hue within
                            9×9 sliding windows, soft-masked by chroma
                            magnitude.  Detects AI colour bleeding.
pearson_y_cr              : Pearson correlation between Y and Cr channels.
pearson_y_cb              : Pearson correlation between Y and Cb channels.
pearson_cr_cb             : Pearson correlation between Cr and Cb channels.
energy_ratio_chroma       : (var(Cr) + var(Cb)) / (var(Y) + eps).
                            Relative chroma energy vs luminance.
glcm_contrast_cr          : GLCM Contrast on quantised Cr, averaged over
                            4 angles at distance 2.
glcm_correlation_cr       : GLCM Correlation (same setting).
glcm_energy_cr            : GLCM Energy (ASM) (same setting).
glcm_homogeneity_cr       : GLCM Homogeneity (IDM) (same setting).

NOTE on GLCM
-------------
The GLCM is computed manually (no skimage) for full transparency and
determinism.  Cr is quantised to 8 levels via centered quantisation
``clip(Cr + 16, 0, 255).astype(int32) // 32`` clamped to [0, 7].
The ``+16`` shift centres grey (128) in the middle of bin 4, preventing
micro-noise around the grey axis from straddling a bin boundary.

Dependencies: numpy, pandas, tqdm.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# Constants (locked — do not modify without spec update)
# ────────────────────────────────────────────────────────────────────────

CROP_SIZE: int = 256
"""Spatial size of preprocessed arrays (H × W)."""

WINDOW_SIZE: int = 9
"""Sliding window size for local colour inconsistency (stride 1)."""

PIXEL_NOISE_FLOOR: float = 1.5
"""Minimum chroma magnitude for a pixel to be treated as having signal."""

MIN_VALID_PIXELS: int = 2
"""Minimum valid pixels in a window for circular variance computation."""

CHROMA_PERCENTILE: int = 50
"""Percentile used for soft-mask threshold on per-window mean magnitude."""

MIN_WINDOW_CHROMA_FLOOR: float = 1.5
"""Absolute floor for per-window mean chroma — prevents threshold
collapsing to near-zero on monochrome / dark images."""

EPS: float = 1e-8
"""Global epsilon for numerical guards (denominators)."""

GLCM_LEVELS: int = 8
"""Number of quantisation levels for GLCM on Cr channel."""

GLCM_DISTANCE: int = 2
"""Pixel offset distance for GLCM co-occurrence."""

GLCM_SHIFT_OFFSET: int = 16
"""Half-bin width offset for centered quantisation (anti half-bin trap)."""

GLCM_BIN_WIDTH: int = 32
"""Bin width for GLCM quantisation (256 / 8 = 32)."""

MIN_SURVIVING_WINDOWS: int = 100
"""Minimum number of surviving windows for a meaningful LCI statistic.

When fewer than this many windows pass the dual-threshold mask, the
per-window circular-variance sample is too small for a stable mean.
Returns the 0.0 sentinel instead.  This prevents noisy estimates on
low-chroma / monochrome images from biasing tree-based models."""

NEAR_ZERO_CHROMA_CEILING: float = 0.5
"""If the global mean chroma magnitude is below this, skip LCI entirely.

Images with M globally near zero (dark/monochrome) produce unit-vectors
dominated by quantisation noise even after PIXEL_NOISE_FLOOR filtering.
Returns 0.0 sentinel."""

FEATURE_KEYS: tuple[str, ...] = (
    "local_color_inconsistency",
    "pearson_y_cr",
    "pearson_y_cb",
    "pearson_cr_cb",
    "energy_ratio_chroma",
    "glcm_contrast_cr",
    "glcm_correlation_cr",
    "glcm_energy_cr",
    "glcm_homogeneity_cr",
)
"""Locked output keys — order and naming must not change."""


# ────────────────────────────────────────────────────────────────────────
# Input validation
# ────────────────────────────────────────────────────────────────────────

def _validate_ycrcb_array(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate *arr* shape/dtype and return (Y, Cr, Cb) as float64.

    Parameters
    ----------
    arr : np.ndarray
        Expected shape ``(256, 256, 3)``, dtype uint8 / float32 / float64.
        Channel order: 0=Y, 1=Cr, 2=Cb (OpenCV ``COLOR_BGR2YCrCb``).

    Returns
    -------
    tuple of np.ndarray
        (Y, Cr, Cb) each as float64, shape ``(256, 256)``.

    Raises
    ------
    TypeError / ValueError
        On invalid input.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(arr).__name__}")
    if arr.ndim != 3 or arr.shape != (CROP_SIZE, CROP_SIZE, 3):
        raise ValueError(
            f"Expected shape ({CROP_SIZE}, {CROP_SIZE}, 3), got {arr.shape}"
        )
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"Expected numeric dtype, got {arr.dtype}")
    Y = arr[:, :, 0].astype(np.float64)
    Cr = arr[:, :, 1].astype(np.float64)
    Cb = arr[:, :, 2].astype(np.float64)
    return Y, Cr, Cb


# ────────────────────────────────────────────────────────────────────────
# Core math helpers (private)
# ────────────────────────────────────────────────────────────────────────

def _extract_local_color_inconsistency(
    Cr: np.ndarray,
    Cb: np.ndarray,
) -> float:
    """Weighted circular variance of chroma hue in 9×9 sliding windows.

    Fully vectorised implementation — no Python-level pixel loops.
    Uses ``scipy.ndimage.uniform_filter`` for all per-window aggregations
    (sum_U, sum_V, n_valid, mbar) in a single pass over pre-masked arrays.

    Steps:
      1. Centre chroma: Cr' = Cr - 128, Cb' = Cb - 128.
      2. Magnitude M = sqrt(Cr'² + Cb'²).
      3. Early exit: if global mean(M) < NEAR_ZERO_CHROMA_CEILING → 0.0.
      4. Build pixel-level valid mask (M > PIXEL_NOISE_FLOOR).
      5. Compute unit vectors U, V only where valid (zeroed elsewhere).
      6. uniform_filter over U, V, valid_mask → per-window sums.
      7. Circular variance: var_circ = 1 - R_window where N_valid >= 2.
      8. mbar = uniform_filter(M) (mean over all 81 pixels per spec).
      9. Dual-threshold soft-mask on mbar.
     10. Sentinel 0.0 if fewer than MIN_SURVIVING_WINDOWS pass.
    """
    Cr_c = Cr - 128.0
    Cb_c = Cb - 128.0
    M = np.sqrt(Cr_c ** 2 + Cb_c ** 2)

    # Early-exit: globally near-zero chroma → sentinel.
    # Prevents quantisation noise from masquerading as hue dispersion.
    if float(np.mean(M)) < NEAR_ZERO_CHROMA_CEILING:
        return 0.0

    half_w = WINDOW_SIZE // 2  # 4

    # ── Pixel-level valid mask and unit vectors ──────────────────────
    valid = (M > PIXEL_NOISE_FLOOR).astype(np.float64)  # 1.0 / 0.0
    M_safe = M + EPS  # avoid /0 on invalid pixels (zeroed by valid anyway)
    U_pixel = (Cr_c / M_safe) * valid  # unit Cr component (0 where invalid)
    V_pixel = (Cb_c / M_safe) * valid  # unit Cb component (0 where invalid)

    # ── Per-window aggregation via uniform_filter ────────────────────
    # uniform_filter computes the mean; multiply by window area to get sum.
    win_area = float(WINDOW_SIZE * WINDOW_SIZE)  # 81
    sum_U = uniform_filter(U_pixel, size=WINDOW_SIZE, mode="constant") * win_area
    sum_V = uniform_filter(V_pixel, size=WINDOW_SIZE, mode="constant") * win_area
    n_valid = uniform_filter(valid, size=WINDOW_SIZE, mode="constant") * win_area

    # mbar: mean of M over all 81 pixels (per spec §2.1 step 3).
    mbar_full = uniform_filter(M, size=WINDOW_SIZE, mode="constant")

    # ── Crop to valid window centres (skip half_w border) ───────────
    sl = slice(half_w, -half_w)
    sum_U = sum_U[sl, sl]
    sum_V = sum_V[sl, sl]
    n_valid_map = n_valid[sl, sl]
    mbar_map = mbar_full[sl, sl]

    # ── Fix IEEE 754 precision on integer count ──────────────────────
    # uniform_filter divides by 81 (3⁴, not a power of 2), so
    # mean × 81 can land at 1.9999…98 instead of 2.0.  Rounding to
    # the nearest integer before comparison restores determinism.
    n_valid_map = np.round(n_valid_map).astype(np.int32)

    # ── Circular variance per window ─────────────────────────────────
    # R = sqrt(sum_U² + sum_V²) / n_valid; var_circ = 1 - R.
    # Where n_valid < MIN_VALID_PIXELS → var_circ = 0.0 (spec guard).
    enough = n_valid_map >= MIN_VALID_PIXELS
    n_safe = np.where(enough, n_valid_map, 1).astype(np.float64)  # avoid /0
    R_map = np.sqrt(sum_U ** 2 + sum_V ** 2) / n_safe
    # Clip to [0, 1]: FP accumulation in sum_U/sum_V can push R
    # marginally above 1.0 when all valid pixels share identical hue,
    # yielding a mathematically impossible negative variance.
    var_circ_map = np.where(enough, np.clip(1.0 - R_map, 0.0, 1.0), 0.0)

    # ── Dual-threshold mask (spec §2.1 step 4) ──────────────────────
    threshold = max(
        float(np.percentile(mbar_map, CHROMA_PERCENTILE)),
        MIN_WINDOW_CHROMA_FLOOR,
    )
    # Joint mask: sufficient chroma AND enough valid pixels.
    # Without the n_valid guard, a window with 1 bright outlier among
    # 80 grey pixels would pass the mbar threshold yet carry a
    # meaningless var_circ = 0.0, biasing the mean downward.
    valid_mask = (mbar_map >= threshold) & (n_valid_map >= MIN_VALID_PIXELS)
    surviving = var_circ_map[valid_mask]

    if surviving.size < MIN_SURVIVING_WINDOWS:
        return 0.0
    return float(np.mean(surviving))


def _extract_pearson_correlations(
    Y: np.ndarray,
    Cr: np.ndarray,
    Cb: np.ndarray,
) -> tuple[float, float, float]:
    """Pearson correlation for (Y,Cr), (Y,Cb), (Cr,Cb).

    Returns 0.0 for any pair where either std < 1e-6.
    """
    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.ravel()
        b_flat = b.ravel()
        std_a = float(np.std(a_flat))
        std_b = float(np.std(b_flat))
        if std_a < 1e-6 or std_b < 1e-6:
            return 0.0
        cov = float(np.mean((a_flat - np.mean(a_flat)) * (b_flat - np.mean(b_flat))))
        return cov / (std_a * std_b + EPS)

    return _pearson(Y, Cr), _pearson(Y, Cb), _pearson(Cr, Cb)


def _extract_energy_ratio(
    Y: np.ndarray,
    Cr: np.ndarray,
    Cb: np.ndarray,
) -> float:
    """Chroma energy ratio: (var(Cr) + var(Cb)) / (var(Y) + eps).

    Returns 0.0 when var(Y) < 1e-6 (flat luminance).
    """
    var_y = float(np.var(Y))
    if var_y < 1e-6:
        return 0.0
    var_cr = float(np.var(Cr))
    var_cb = float(np.var(Cb))
    return (var_cr + var_cb) / (var_y + EPS)


def _quantise_cr_for_glcm(Cr: np.ndarray) -> np.ndarray:
    """Centered quantisation of Cr to [0, 7] integer levels.

    ``q = clip(clip(Cr + 16, 0, 255).astype(int32) // 32, 0, 7)``

    The +16 shift centres grey (128) in the middle of bin 4, preventing
    micro-noise from straddling a bin boundary.
    """
    shifted = np.clip(Cr + GLCM_SHIFT_OFFSET, 0, 255).astype(np.int32)
    q = np.clip(shifted // GLCM_BIN_WIDTH, 0, GLCM_LEVELS - 1)
    return q


def _compute_glcm_single(
    q: np.ndarray,
    dy: int,
    dx: int,
) -> np.ndarray:
    """Compute symmetric, normalised GLCM for one (dy, dx) offset.

    Uses ``abs(dy)``, ``abs(dx)`` for slicing so that negative offsets
    produce the *same* pixel-pair region as positive ones.  This ensures
    all 4 angles sample from the same (H-D) × (W-D) or (H-D) × W
    interior, giving equal co-occurrence counts across angles.

    Symmetry is enforced by accumulating both (p1→p2) and (p2→p1)
    via ``glcm + glcm.T``, then normalising to a probability matrix.

    Parameters
    ----------
    q : np.ndarray
        Quantised image, int32, values in [0, GLCM_LEVELS-1].
    dy, dx : int
        Row and column offset (distance vector).

    Returns
    -------
    np.ndarray
        Shape ``(GLCM_LEVELS, GLCM_LEVELS)``, float64, sums to 1.0.
    """
    H, W = q.shape
    ady, adx = abs(dy), abs(dx)

    # Reference pixel → offset pixel pairing.
    # For each (dy, dx), pair (row, col) with (row+dy, col+dx).
    r_ref = slice(0, H - ady) if dy >= 0 else slice(ady, H)
    c_ref = slice(0, W - adx) if dx >= 0 else slice(adx, W)
    r_off = slice(ady, H) if dy >= 0 else slice(0, H - ady)
    c_off = slice(adx, W) if dx >= 0 else slice(0, W - adx)

    p1 = q[r_ref, c_ref].ravel()
    p2 = q[r_off, c_off].ravel()

    # Build co-occurrence matrix via unbuffered indexing.
    glcm = np.zeros((GLCM_LEVELS, GLCM_LEVELS), dtype=np.float64)
    np.add.at(glcm, (p1, p2), 1)

    # Symmetric: P(i,j) = P(j,i)
    glcm = glcm + glcm.T

    # Normalise to probability matrix
    total = glcm.sum()
    if total > 0:
        glcm /= total
    return glcm


# Precomputed GLCM index grids (module-level singletons, computed once).
# Avoids re-creating meshgrid in every _glcm_metrics() call (4× per image).
_GLCM_II, _GLCM_JJ = np.meshgrid(
    np.arange(GLCM_LEVELS, dtype=np.float64),
    np.arange(GLCM_LEVELS, dtype=np.float64),
    indexing="ij",
)
_GLCM_DIFF_SQ = (_GLCM_II - _GLCM_JJ) ** 2
_GLCM_ABS_DIFF_P1 = 1.0 + np.abs(_GLCM_II - _GLCM_JJ)


def _glcm_metrics(glcm: np.ndarray) -> tuple[float, float, float, float]:
    """Compute Contrast, Correlation, Energy, Homogeneity from one GLCM.

    Uses module-level precomputed index grids to avoid per-call overhead.

    Parameters
    ----------
    glcm : np.ndarray
        Normalised ``(GLCM_LEVELS, GLCM_LEVELS)`` co-occurrence matrix.

    Returns
    -------
    contrast, correlation, energy, homogeneity : float
    """
    # Contrast = sum_ij (i-j)^2 * P(i,j)
    contrast = float(np.sum(_GLCM_DIFF_SQ * glcm))

    # Marginals
    mu_i = float(np.sum(_GLCM_II * glcm))
    mu_j = float(np.sum(_GLCM_JJ * glcm))
    sigma_i = np.sqrt(float(np.sum((_GLCM_II - mu_i) ** 2 * glcm)))
    sigma_j = np.sqrt(float(np.sum((_GLCM_JJ - mu_j) ** 2 * glcm)))

    # Correlation = sum_ij (i - mu_i)(j - mu_j) P(i,j) / (sigma_i * sigma_j)
    if sigma_i < EPS or sigma_j < EPS:
        correlation = 0.0
    else:
        correlation = float(
            np.sum((_GLCM_II - mu_i) * (_GLCM_JJ - mu_j) * glcm)
        ) / (sigma_i * sigma_j + EPS)

    # Energy (ASM) = sum_ij P(i,j)^2
    energy = float(np.sum(glcm ** 2))

    # Homogeneity (IDM) = sum_ij P(i,j) / (1 + |i-j|)
    homogeneity = float(np.sum(glcm / _GLCM_ABS_DIFF_P1))

    return contrast, correlation, energy, homogeneity


# Precompute GLCM offset vectors for 4 angles at distance D=2
# 0°: (0, D), 45°: (-D, D), 90°: (-D, 0), 135°: (-D, -D)
_GLCM_OFFSETS: list[tuple[int, int]] = [
    (0, GLCM_DISTANCE),               # 0°
    (-GLCM_DISTANCE, GLCM_DISTANCE),  # π/4
    (-GLCM_DISTANCE, 0),              # π/2
    (-GLCM_DISTANCE, -GLCM_DISTANCE), # 3π/4
]


def _extract_glcm_features(Cr: np.ndarray) -> tuple[float, float, float, float]:
    """GLCM features on the Cr channel, averaged over 4 angles.

    Returns
    -------
    contrast, correlation, energy, homogeneity : float
        Mean of 4-angle GLCM metrics.
    """
    q = _quantise_cr_for_glcm(Cr)

    contrast_sum = 0.0
    correlation_sum = 0.0
    energy_sum = 0.0
    homogeneity_sum = 0.0

    for dy, dx in _GLCM_OFFSETS:
        glcm = _compute_glcm_single(q, dy, dx)
        con, cor, eng, hom = _glcm_metrics(glcm)
        contrast_sum += con
        correlation_sum += cor
        energy_sum += eng
        homogeneity_sum += hom

    n = len(_GLCM_OFFSETS)
    return (
        contrast_sum / n,
        correlation_sum / n,
        energy_sum / n,
        homogeneity_sum / n,
    )


# ────────────────────────────────────────────────────────────────────────
# Audit helper
# ────────────────────────────────────────────────────────────────────────

def extract_color_audit(ycbcr_npy: np.ndarray) -> dict[str, object]:
    """Extract features plus intermediate diagnostics for auditing.

    Returns all 9 features plus additional fields useful during data
    exploration:

    - ``n_windows_total`` / ``n_windows_surviving``: How many 9×9
      windows existed / survived the dual-threshold mask.
    - ``chroma_threshold``: The effective chroma-magnitude threshold.
    - ``mean_magnitude``: Mean of M = sqrt(Cr'² + Cb'²).
    - ``var_Y``, ``var_Cr``, ``var_Cb``: Channel variances.
    """
    Y, Cr, Cb = _validate_ycrcb_array(ycbcr_npy)

    feats = _extract_all(Y, Cr, Cb)

    Cr_c = Cr - 128.0
    Cb_c = Cb - 128.0
    M = np.sqrt(Cr_c ** 2 + Cb_c ** 2)

    half_w = WINDOW_SIZE // 2
    n_rows = CROP_SIZE - 2 * half_w
    n_cols = CROP_SIZE - 2 * half_w
    n_total = n_rows * n_cols

    mean_mag = float(np.mean(M))

    # Mirror the early-exit in _extract_local_color_inconsistency:
    # if global chroma is near-zero, LCI returns sentinel 0.0 without
    # running window analysis — audit must reflect that truthfully.
    if mean_mag < NEAR_ZERO_CHROMA_CEILING:
        audit: dict[str, object] = {
            **feats,
            "n_windows_total": n_total,
            "n_windows_surviving": 0,
            "chroma_threshold": np.nan,
            "lci_early_exit": "near_zero_chroma",
            "mean_magnitude": mean_mag,
            "var_Y": float(np.var(Y)),
            "var_Cr": float(np.var(Cr)),
            "var_Cb": float(np.var(Cb)),
        }
        return audit

    # Recompute mbar_map and n_valid_map for audit stats (vectorised).
    # Must mirror the exact dual-condition mask used in
    # _extract_local_color_inconsistency to avoid misleading diagnostics.
    mbar_full = uniform_filter(M, size=WINDOW_SIZE, mode="constant")
    mbar_map = mbar_full[half_w:-half_w, half_w:-half_w]

    valid = (M > PIXEL_NOISE_FLOOR).astype(np.float64)
    n_valid_full = uniform_filter(valid, size=WINDOW_SIZE, mode="constant") * (WINDOW_SIZE * WINDOW_SIZE)
    n_valid_map = np.round(n_valid_full[half_w:-half_w, half_w:-half_w]).astype(np.int32)

    threshold = max(
        float(np.percentile(mbar_map, CHROMA_PERCENTILE)),
        MIN_WINDOW_CHROMA_FLOOR,
    )
    surviving_mask = (mbar_map >= threshold) & (n_valid_map >= MIN_VALID_PIXELS)
    n_surviving = int(np.count_nonzero(surviving_mask))

    audit = {
        **feats,
        "n_windows_total": n_total,
        "n_windows_surviving": n_surviving,
        "chroma_threshold": threshold,
        "lci_early_exit": None,
        "mean_magnitude": mean_mag,
        "var_Y": float(np.var(Y)),
        "var_Cr": float(np.var(Cr)),
        "var_Cb": float(np.var(Cb)),
    }
    return audit


# ────────────────────────────────────────────────────────────────────────
# Path metadata helper
# ────────────────────────────────────────────────────────────────────────

def _infer_label_generator_from_path(
    file_path: str | Path,
    root: Path,
) -> tuple[str, str]:
    """Best-effort extraction of (generator, label) from directory layout.

    Expects ``root / <generator> / <label> / file.npy``.
    Returns ``("", "")`` on failure — never raises.
    """
    try:
        rel = Path(file_path).relative_to(root)
        parts = rel.parts
        if len(parts) >= 3:
            return parts[0], parts[1]
    except (ValueError, IndexError):
        pass
    return "", ""


# ────────────────────────────────────────────────────────────────────────
# Internal aggregator
# ────────────────────────────────────────────────────────────────────────

def _extract_all(
    Y: np.ndarray,
    Cr: np.ndarray,
    Cb: np.ndarray,
) -> dict[str, float]:
    """Extract all 9 features from already-validated channels."""
    lci = _extract_local_color_inconsistency(Cr, Cb)
    p_y_cr, p_y_cb, p_cr_cb = _extract_pearson_correlations(Y, Cr, Cb)
    energy_ratio = _extract_energy_ratio(Y, Cr, Cb)
    g_con, g_cor, g_eng, g_hom = _extract_glcm_features(Cr)

    return {
        "local_color_inconsistency": lci,
        "pearson_y_cr": p_y_cr,
        "pearson_y_cb": p_y_cb,
        "pearson_cr_cb": p_cr_cb,
        "energy_ratio_chroma": energy_ratio,
        "glcm_contrast_cr": g_con,
        "glcm_correlation_cr": g_cor,
        "glcm_energy_cr": g_eng,
        "glcm_homogeneity_cr": g_hom,
    }


# ────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────

def extract_color_features(ycbcr_npy: np.ndarray) -> dict[str, float]:
    """Extract the 9 locked Group-2 color cross-correlation features.

    Parameters
    ----------
    ycbcr_npy : np.ndarray
        Shape ``(256, 256, 3)`` YCrCb array.
        Channel 0 = Y, 1 = Cr, 2 = Cb.
        Accepted dtypes: uint8, float32, float64.

    Returns
    -------
    dict[str, float]
        Exactly 9 keys as defined in ``FEATURE_KEYS``.
        Guaranteed finite (no NaN / Inf).
    """
    Y, Cr, Cb = _validate_ycrcb_array(ycbcr_npy)
    result = _extract_all(Y, Cr, Cb)

    # Safety net
    for key, val in result.items():
        if not np.isfinite(val):
            logger.error(
                "Non-finite value for '%s' = %s; clamped to 0.0", key, val,
            )
            result[key] = 0.0

    return result


def extract_color_features_from_file(
    path: str | Path,
) -> dict[str, float]:
    """Load a ``.npy`` file and extract color features.

    Convenience wrapper for single-file use in notebooks.
    """
    arr = np.load(Path(path), allow_pickle=False)
    return extract_color_features(arr)


def extract_color_batch(
    paths: Sequence[Path | str],
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Batch extraction with per-file error isolation.

    Parameters
    ----------
    paths : sequence of paths
        ``.npy`` files to process.
    show_progress : bool
        Show ``tqdm`` progress bar (default True).

    Returns
    -------
    pd.DataFrame
        Columns: ``file_path``, 9 feature columns, ``status``, ``error``.
        Failed files have ``status="error"`` and NaN features.
    """
    records: list[dict] = []
    iterator = tqdm(
        paths, desc="Extracting color features", disable=not show_progress,
    )
    for p in iterator:
        p_str = str(Path(p))
        rec: dict = {"file_path": p_str}
        try:
            arr = np.load(p_str, allow_pickle=False)
            feats = extract_color_features(arr)
            rec.update(feats)
            rec["status"] = "ok"
            rec["error"] = ""
        except Exception as exc:
            logger.warning("Feature extraction failed for %s: %s", p_str, exc)
            for k in FEATURE_KEYS:
                rec[k] = np.nan
            rec["status"] = "error"
            rec["error"] = str(exc)
        records.append(rec)

    columns = ["file_path"] + list(FEATURE_KEYS) + ["status", "error"]
    return pd.DataFrame(records, columns=columns)


def extract_color_dataset(
    input_root: Path | str,
    output_csv: Path | str | None = None,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Walk a processed directory tree, extract features, and optionally save.

    Expects layout ``input_root / <Generator> / <label> / *.npy``.

    Parameters
    ----------
    input_root : path
        Root of the preprocessed ``.npy`` tree (e.g. ``data/processed``).
    output_csv : path, optional
        If given, write the result DataFrame to CSV.
    show_progress : bool
        Show ``tqdm`` progress bar.

    Returns
    -------
    pd.DataFrame
        Columns: ``file_path``, ``generator``, ``label``,
        9 feature columns, ``status``, ``error``.
    """
    input_root = Path(input_root)
    npy_files = sorted(input_root.rglob("*.npy"))

    if not npy_files:
        logger.warning("No .npy files found under %s", input_root)
        return pd.DataFrame()

    logger.info("Found %d .npy files under %s", len(npy_files), input_root)

    df = extract_color_batch(npy_files, show_progress=show_progress)

    # Infer metadata from directory structure
    generators: list[str] = []
    labels: list[str] = []
    for p in npy_files:
        gen, lbl = _infer_label_generator_from_path(p, input_root)
        generators.append(gen)
        labels.append(lbl)
    df.insert(1, "generator", generators)
    df.insert(2, "label", labels)

    if output_csv is not None:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        logger.info("Saved feature CSV -> %s", out)

    return df


# ────────────────────────────────────────────────────────────────────────
# Minimal smoke test
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    # ── Test 1: Random noise (high chroma) ───────────────────────────
    rng = np.random.default_rng(42)
    dummy = rng.integers(0, 256, size=(CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    feats = extract_color_features(dummy)
    print("Test 1 — Random noise:")
    for k, v in feats.items():
        print(f"  {k}: {v:.8e}")
    assert all(np.isfinite(v) for v in feats.values()), "Non-finite!"

    # ── Test 2: Flat grey (all 128) → all sentinels ──────────────────
    flat = np.full((CROP_SIZE, CROP_SIZE, 3), 128, dtype=np.uint8)
    feats2 = extract_color_features(flat)
    print("\nTest 2 — Flat grey (128,128,128):")
    for k, v in feats2.items():
        print(f"  {k}: {v:.8e}")
    assert feats2["local_color_inconsistency"] == 0.0, "LCI should be sentinel"
    assert feats2["pearson_y_cr"] == 0.0
    assert feats2["energy_ratio_chroma"] == 0.0

    # ── Test 3: True grey (Y=128, Cr=Cb=128 → chroma=0) → sentinel ──
    grey = np.full((CROP_SIZE, CROP_SIZE, 3), 128, dtype=np.uint8)
    # Add tiny Y variation so Pearson isn't trivially tested twice
    grey[:, :, 0] = rng.integers(126, 131, size=(CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    feats3 = extract_color_features(grey)
    print("\nTest 3 — True grey chroma (Cr=Cb=128):")
    for k, v in feats3.items():
        print(f"  {k}: {v:.8e}")
    assert feats3["local_color_inconsistency"] == 0.0, "Zero chroma → LCI sentinel"

    # ── Test 4: Uniform saturated red → near-zero LCI ────────────────
    red = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    red[:, :, 0] = 80   # Y
    red[:, :, 1] = 240   # Cr (high)
    red[:, :, 2] = 90    # Cb (low)
    feats4 = extract_color_features(red)
    print("\nTest 4 — Uniform saturated red:")
    for k, v in feats4.items():
        print(f"  {k}: {v:.8e}")
    # Uniform hue → all unit vectors identical → R≈1 → var_circ≈0
    assert feats4["local_color_inconsistency"] < 1e-6, "Uniform hue → LCI ≈ 0"

    print("\nAll smoke tests passed.")
