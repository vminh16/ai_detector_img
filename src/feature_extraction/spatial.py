"""
src/feature_extraction/spatial.py
=======================
Group 4 — Normalized Spatial Stats feature extraction (locked spec).

Extracts 8 features from the Y / Cr / Cb channels of preprocessed 256×256
YCrCb arrays produced by the hardened preprocessing pipeline (reflect
padding, misaligned center crop, deterministic JPEG bottleneck → .npy).

Features
--------
**Spatial SNR Ratio (1 dim)** — relative edge-vs-flat residual imbalance.
Sobel 3×3 is used *only* as a router (localising edge / flat zones);
the ``SQUARE3x3`` high-pass kernel is used *only* as a measurer (residual
magnitude).  Log₁₀ ratio with noise-floor additive constant reduces
(but does not eliminate) dependence on global JPEG compression level.

spatial_snr_ratio : log₁₀((V_edge + c) / (V_flat + c))

**Cross Noise Ratio (1 dim)** — luma-vs-chroma residual magnitude ratio.
Compares mean absolute residual of Y and Cb channels using the same
high-pass kernel.
CAUTION: this feature has elevated risk of learning chroma subsampling
history (4:2:0) rather than a generator fingerprint.

cross_noise_ratio : Noise_Y / Noise_Cb

**Residual Distribution Stats (6 dims)** — skewness and excess kurtosis
of high-pass residuals on Y, Cr, Cb independently.
These are *empirical* statistics based on the working hypothesis that
optical noise residuals (shot noise, demosaicing, JPEG ringing) are
closer to Gaussian than structured latent-space decoding artifacts.
This is NOT a physical law; it is a testable conjecture for the
downstream classifier.

skew_noise_y, kurt_noise_y   : on Y residual
skew_noise_cr, kurt_noise_cr : on Cr residual
skew_noise_cb, kurt_noise_cb : on Cb residual

Dependencies: numpy, scipy (≥ 1.7), pandas, tqdm.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# Constants (locked — do not modify without spec update)
# ────────────────────────────────────────────────────────────────────────

CROP_SIZE: int = 256
"""Spatial size of preprocessed arrays (H × W)."""

MIN_GRAD_EDGE: float = 15.0
"""Physical minimum gradient magnitude for edge structure."""

MAX_GRAD_FLAT: float = 3.0
"""Physical maximum gradient magnitude for flat regions."""

NOISE_FLOOR: float = 1e-3
"""Laplace-smoothing constant for log₁₀ ratio to prevent divergence."""

SIGMA_GUARD: float = 1e-6
"""Minimum standard deviation for skew/kurtosis computation.
Below this the residual is considered dead signal → NaN."""

MIN_FLAT_PIXELS: int = 30
"""Minimum flat-zone pixels for meaningful skew/kurtosis estimation.
Below this threshold, moment estimates are statistically unreliable → NaN."""

FEATURE_KEYS: tuple[str, ...] = (
    "spatial_snr_ratio",
    "cross_noise_ratio",
    "skew_noise_y",
    "kurt_noise_y",
    "skew_noise_cr",
    "kurt_noise_cr",
    "skew_noise_cb",
    "kurt_noise_cb",
)
"""Locked output keys — order and naming must not change."""


# ────────────────────────────────────────────────────────────────────────
# Kernels (hardcoded, deterministic)
# ────────────────────────────────────────────────────────────────────────

_KERNEL_SQUARE3: np.ndarray = np.array(
    [[-1,  2, -1],
     [ 2, -4,  2],
     [-1,  2, -1]],
    dtype=np.float64,
)
"""3×3 multi-directional Laplacian (zero-sum, centrosymmetric).
Identical to the SQUARE3x3 kernel used in Group 3."""

_SOBEL_X: np.ndarray = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]],
    dtype=np.float64,
)
"""Sobel horizontal gradient kernel."""

_SOBEL_Y: np.ndarray = np.array(
    [[-1, -2, -1],
     [ 0,  0,  0],
     [ 1,  2,  1]],
    dtype=np.float64,
)
"""Sobel vertical gradient kernel."""


# ────────────────────────────────────────────────────────────────────────
# Input validation
# ────────────────────────────────────────────────────────────────────────

def _validate_ycrcb_array(
    arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate *arr* shape/dtype and return (Y, Cr, Cb) as float64.

    Parameters
    ----------
    arr : np.ndarray
        Expected shape ``(256, 256, 3)``.
        Channel order: 0=Y, 1=Cr, 2=Cb (OpenCV ``COLOR_BGR2YCrCb``).

    Returns
    -------
    tuple of np.ndarray
        (Y, Cr, Cb) each as float64, shape ``(256, 256)``.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(arr).__name__}")
    if arr.ndim != 3 or arr.shape != (CROP_SIZE, CROP_SIZE, 3):
        raise ValueError(
            f"Expected shape ({CROP_SIZE}, {CROP_SIZE}, 3), got {arr.shape}"
        )
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"Expected numeric dtype, got {arr.dtype}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "Array contains non-finite values (NaN or Inf). "
            "This indicates corrupted data from the preprocessing pipeline, "
            "not a valid physical state."
        )
    Y = arr[:, :, 0].astype(np.float64)
    Cr = arr[:, :, 1].astype(np.float64)
    Cb = arr[:, :, 2].astype(np.float64)
    return Y, Cr, Cb


# ────────────────────────────────────────────────────────────────────────
# Convolution helpers (private)
# ────────────────────────────────────────────────────────────────────────

def _convolve_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """True 2D convolution (NOT correlation) with mode='valid'.

    Uses ``scipy.signal.convolve2d`` which implements mathematical
    convolution (kernel is flipped).  Mode ``valid`` avoids boundary
    padding artifacts.

    Parameters
    ----------
    image : np.ndarray
        2D float64 array.
    kernel : np.ndarray
        2D float64 kernel (must be smaller than *image*).

    Returns
    -------
    np.ndarray
        Residual map, shape ``(H - kh + 1, W - kw + 1)``.
    """
    return convolve2d(image, kernel, mode="valid")


def _compute_residual_sq3(channel: np.ndarray) -> np.ndarray:
    """Compute high-pass residual using SQUARE3x3, mode='valid'.

    Parameters
    ----------
    channel : np.ndarray
        2D float64, shape ``(256, 256)``.

    Returns
    -------
    np.ndarray
        Residual, shape ``(254, 254)``.
    """
    return _convolve_valid(channel, _KERNEL_SQUARE3)


def _compute_gradient_valid(Y: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude via Sobel 3×3, mode='valid'.

    Parameters
    ----------
    Y : np.ndarray
        Luminance channel, float64, shape ``(256, 256)``.

    Returns
    -------
    np.ndarray
        Gradient magnitude ``sqrt(Gx² + Gy²)``, shape ``(254, 254)``.
    """
    Gx = _convolve_valid(Y, _SOBEL_X)
    Gy = _convolve_valid(Y, _SOBEL_Y)
    return np.sqrt(Gx ** 2 + Gy ** 2)


# ────────────────────────────────────────────────────────────────────────
# Mask computation (shared by SNR and skew/kurt)
# ────────────────────────────────────────────────────────────────────────

def _compute_flat_edge_masks(
    G: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute edge/flat boolean masks from gradient magnitude.

    Parameters
    ----------
    G : np.ndarray
        Gradient magnitude, shape ``(254, 254)``.

    Returns
    -------
    M_edge, M_flat : np.ndarray
        Boolean masks, same shape as *G*.
    T_edge, T_flat : float
        Thresholds actually used.
    """
    T_edge = max(float(np.percentile(G, 90)), MIN_GRAD_EDGE)
    T_flat = min(float(np.percentile(G, 30)), MAX_GRAD_FLAT)
    M_edge = G >= T_edge
    M_flat = G <= T_flat
    return M_edge, M_flat, T_edge, T_flat


# ────────────────────────────────────────────────────────────────────────
# Feature A: Spatial SNR Ratio
# ────────────────────────────────────────────────────────────────────────

def _extract_spatial_snr_ratio(
    R_Y: np.ndarray,
    M_edge: np.ndarray,
    M_flat: np.ndarray,
) -> float:
    """Compute spatial SNR ratio from precomputed residual and masks.

    Measuring: SQUARE3x3 residual magnitudes on edge vs flat zones.

    Returns ``np.nan`` if either mask is empty or V_flat < 1e-6
    (dead flat-zone signal — must surface as NaN for Dual-Imputation).

    Parameters
    ----------
    R_Y : np.ndarray
        SQUARE3x3 residual of Y, shape ``(254, 254)``.
    M_edge : np.ndarray
        Boolean mask for edge pixels.
    M_flat : np.ndarray
        Boolean mask for flat pixels.
    """
    if not np.any(M_edge):
        return np.nan
    if not np.any(M_flat):
        return np.nan

    abs_R_Y = np.abs(R_Y)
    V_edge = float(np.mean(abs_R_Y[M_edge]))
    V_flat = float(np.mean(abs_R_Y[M_flat]))

    # Dead flat-zone guard: signal too weak for meaningful ratio.
    if V_flat < 1e-6:
        return np.nan

    return float(np.log10((V_edge + NOISE_FLOOR) / (V_flat + NOISE_FLOOR)))


# ────────────────────────────────────────────────────────────────────────
# Feature B: Cross Noise Ratio
# ────────────────────────────────────────────────────────────────────────

def _extract_cross_noise_ratio(
    R_Y: np.ndarray,
    R_Cb: np.ndarray,
) -> float:
    """Compute cross-channel noise ratio from precomputed residuals.

    CAUTION: elevated risk of learning chroma 4:2:0 subsampling history
    rather than a genuine generator signature.  Not an absolute physical
    constant.

    Parameters
    ----------
    R_Y : np.ndarray
        SQUARE3x3 residual of Y, shape ``(254, 254)``.
    R_Cb : np.ndarray
        SQUARE3x3 residual of Cb, shape ``(254, 254)``.

    Returns ``np.nan`` if Noise_Cb < 1e-6.
    """
    Noise_Y = float(np.mean(np.abs(R_Y)))
    Noise_Cb = float(np.mean(np.abs(R_Cb)))

    if Noise_Cb < 1e-6:
        return np.nan

    # Laplace smoothing: caps max ratio at ~Noise_Y / NOISE_FLOOR,
    # preventing unbounded explosion on near-grayscale inputs.
    return (Noise_Y + NOISE_FLOOR) / (Noise_Cb + NOISE_FLOOR)


# ────────────────────────────────────────────────────────────────────────
# Feature C: Skewness / Kurtosis on residuals
# ────────────────────────────────────────────────────────────────────────

def _compute_skew_kurt_safe(
    R: np.ndarray,
) -> tuple[float, float]:
    """Compute empirical skewness and excess kurtosis of a residual array.

    Guards:
    - If sample count < ``MIN_FLAT_PIXELS``, estimates are statistically
      unreliable → ``(NaN, NaN)``.
    - If sigma < ``SIGMA_GUARD``, signal is dead → ``(NaN, NaN)``.

    Parameters
    ----------
    R : np.ndarray
        Residual values (any shape — typically 1-D after flat masking).

    Returns
    -------
    skew, kurt : float
        Excess kurtosis (kurt = m4/σ⁴ − 3).
    """
    if R.size < MIN_FLAT_PIXELS:
        return np.nan, np.nan

    mu = float(np.mean(R))
    diff = R - mu
    var = float(np.mean(diff ** 2))
    sigma = np.sqrt(var)

    if sigma < SIGMA_GUARD:
        return np.nan, np.nan

    m3 = float(np.mean(diff ** 3))
    m4 = float(np.mean(diff ** 4))

    skew = m3 / (sigma ** 3)
    kurt = m4 / (sigma ** 4) - 3.0

    return skew, kurt


def _extract_residual_shape_stats(
    R_Y: np.ndarray,
    R_Cr: np.ndarray,
    R_Cb: np.ndarray,
    M_flat: np.ndarray,
) -> dict[str, float]:
    """Compute skew/kurt on *flat-zone only* residuals of all three channels.

    The flat mask (derived from Y-channel gradient) is applied to all
    three channels.  This is physically correct: spatial structure is
    defined by luminance, and applying the same mask to chroma avoids
    edge residuals dominating the 4th-moment estimate.

    Parameters
    ----------
    R_Y, R_Cr, R_Cb : np.ndarray
        SQUARE3x3 residuals, each shape ``(254, 254)``.
    M_flat : np.ndarray
        Boolean mask for flat pixels, shape ``(254, 254)``.
    """
    result: dict[str, float] = {}
    for R, suffix in [(R_Y, "y"), (R_Cr, "cr"), (R_Cb, "cb")]:
        skew, kurt = _compute_skew_kurt_safe(R[M_flat])
        result[f"skew_noise_{suffix}"] = skew
        result[f"kurt_noise_{suffix}"] = kurt
    return result


# ────────────────────────────────────────────────────────────────────────
# Internal aggregator
# ────────────────────────────────────────────────────────────────────────

def _extract_all(
    Y: np.ndarray,
    Cr: np.ndarray,
    Cb: np.ndarray,
) -> dict[str, float]:
    """Extract all 8 features from validated channels.

    Computes each expensive convolution exactly once:
    - G (Sobel gradient on Y): 2 conv2d calls
    - R_Y, R_Cr, R_Cb (SQUARE3x3 residuals): 3 conv2d calls
    Total: 5 conv2d per image.

    The flat/edge masks are computed once from G and shared by
    both SNR-ratio and skew/kurtosis (flat-zone only).
    """
    # Precompute all convolutions once
    G = _compute_gradient_valid(Y)       # (254, 254)  — 2× conv2d
    R_Y = _compute_residual_sq3(Y)       # (254, 254)
    R_Cr = _compute_residual_sq3(Cr)     # (254, 254)
    R_Cb = _compute_residual_sq3(Cb)     # (254, 254)

    # Compute masks once — shared by SNR and skew/kurt
    M_edge, M_flat, _, _ = _compute_flat_edge_masks(G)

    feats: dict[str, float] = {}
    feats["spatial_snr_ratio"] = _extract_spatial_snr_ratio(R_Y, M_edge, M_flat)
    feats["cross_noise_ratio"] = _extract_cross_noise_ratio(R_Y, R_Cb)
    feats.update(_extract_residual_shape_stats(R_Y, R_Cr, R_Cb, M_flat))

    return feats


# ────────────────────────────────────────────────────────────────────────
# Audit helper
# ────────────────────────────────────────────────────────────────────────

def extract_spatial_audit(ycbcr_npy: np.ndarray) -> dict[str, object]:
    """Extract features plus intermediate diagnostics for auditing.

    Returns all 8 features plus:

    - ``edge_pixel_count`` : int — pixels in M_edge mask.
    - ``flat_pixel_count`` : int — pixels in M_flat mask.
    - ``T_edge`` : float — edge threshold used.
    - ``T_flat`` : float — flat threshold used.
    - ``spatial_snr_is_nan`` : bool
    - ``cross_noise_is_nan`` : bool
    - ``Noise_Y`` : float — mean |R_Y| (global).
    - ``Noise_Cb`` : float — mean |R_Cb| (global).
    - ``sigma_y``, ``sigma_cr``, ``sigma_cb`` : float — residual std.
    - ``skew_nan_count`` : int — how many channels hit sigma guard.
    - ``kurt_nan_count`` : int — same.
    - ``residual_shape`` : tuple — shape of valid convolution output.
    """
    Y, Cr, Cb = _validate_ycrcb_array(ycbcr_npy)

    # Precompute convolutions once (shared with _extract_all)
    G = _compute_gradient_valid(Y)
    R_Y = _compute_residual_sq3(Y)
    R_Cr = _compute_residual_sq3(Cr)
    R_Cb = _compute_residual_sq3(Cb)

    # Compute masks once
    M_edge, M_flat, T_edge, T_flat = _compute_flat_edge_masks(G)

    # Extract features using precomputed arrays and masks
    feats: dict[str, float] = {}
    feats["spatial_snr_ratio"] = _extract_spatial_snr_ratio(R_Y, M_edge, M_flat)
    feats["cross_noise_ratio"] = _extract_cross_noise_ratio(R_Y, R_Cb)
    feats.update(_extract_residual_shape_stats(R_Y, R_Cr, R_Cb, M_flat))

    edge_count = int(np.sum(M_edge))
    flat_count = int(np.sum(M_flat))

    # Cross noise audit
    noise_y_val = float(np.mean(np.abs(R_Y)))
    noise_cb_val = float(np.mean(np.abs(R_Cb)))

    # V_flat audit
    if np.any(M_flat):
        V_flat = float(np.mean(np.abs(R_Y)[M_flat]))
    else:
        V_flat = np.nan

    # Skew/kurt audit (flat-zone only, consistent with feature extraction)
    skew_nan = 0
    kurt_nan = 0
    sigmas: dict[str, float] = {}
    for R, suffix in [(R_Y, "y"), (R_Cr, "cr"), (R_Cb, "cb")]:
        R_flat = R[M_flat]
        if R_flat.size < MIN_FLAT_PIXELS:
            sigmas[f"sigma_{suffix}"] = np.nan
            skew_nan += 1
            kurt_nan += 1
            continue
        mu = float(np.mean(R_flat))
        var = float(np.mean((R_flat - mu) ** 2))
        sigma = np.sqrt(var)
        sigmas[f"sigma_{suffix}"] = sigma
        if sigma < SIGMA_GUARD:
            skew_nan += 1
            kurt_nan += 1

    audit: dict[str, object] = {
        **feats,
        "edge_pixel_count": edge_count,
        "flat_pixel_count": flat_count,
        "T_edge": T_edge,
        "T_flat": T_flat,
        "V_flat": V_flat,
        "spatial_snr_is_nan": np.isnan(feats["spatial_snr_ratio"]),
        "cross_noise_is_nan": np.isnan(feats["cross_noise_ratio"]),
        "Noise_Y": noise_y_val,
        "Noise_Cb": noise_cb_val,
        **sigmas,
        "skew_nan_count": skew_nan,
        "kurt_nan_count": kurt_nan,
        "residual_shape": R_Y.shape,
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
# Public API
# ────────────────────────────────────────────────────────────────────────

def extract_spatial_features(ycbcr_npy: np.ndarray) -> dict[str, float]:
    """Extract the 8 locked Group-4 normalized spatial stats features.

    Parameters
    ----------
    ycbcr_npy : np.ndarray
        Shape ``(256, 256, 3)`` YCrCb array.
        Channel 0 = Y, 1 = Cr, 2 = Cb.
        Accepted dtypes: uint8, float32, float64.

    Returns
    -------
    dict[str, float]
        Exactly 8 keys as defined in ``FEATURE_KEYS``.
        Values may include ``np.nan`` when physical guards trigger
        (empty masks, dead signal).  Downstream pipeline is responsible
        for imputation.
    """
    Y, Cr, Cb = _validate_ycrcb_array(ycbcr_npy)
    return _extract_all(Y, Cr, Cb)


def extract_spatial_features_from_file(
    path: str | Path,
) -> dict[str, float]:
    """Load a ``.npy`` file and extract spatial features.

    Convenience wrapper for single-file use in notebooks.
    """
    arr = np.load(Path(path), allow_pickle=False)
    return extract_spatial_features(arr)


def extract_spatial_batch(
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
        Columns: ``file_path``, 8 feature columns, ``status``, ``error``.
        Failed files have ``status="error"`` and NaN features.
    """
    records: list[dict] = []
    iterator = tqdm(
        paths,
        desc="Extracting spatial features",
        disable=not show_progress,
    )
    for p in iterator:
        p_str = str(Path(p))
        rec: dict = {"file_path": p_str}
        try:
            arr = np.load(p_str, allow_pickle=False)
            feats = extract_spatial_features(arr)
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


def extract_spatial_dataset(
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
        8 feature columns, ``status``, ``error``.
    """
    input_root = Path(input_root)
    npy_files = sorted(input_root.rglob("*.npy"))

    if not npy_files:
        logger.warning("No .npy files found under %s", input_root)
        return pd.DataFrame()

    logger.info("Found %d .npy files under %s", len(npy_files), input_root)

    df = extract_spatial_batch(npy_files, show_progress=show_progress)

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

    rng = np.random.default_rng(42)

    # ── Test 1: Random noise ───────────────────────────────────────────────
    # Random noise has near-zero flat pixels (gradient everywhere high),
    # so skew/kurt correctly return NaN.  SNR and cross_noise are finite.
    dummy = rng.integers(0, 256, size=(CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    feats = extract_spatial_features(dummy)
    print("Test 1 — Random noise:")
    for k, v in feats.items():
        print(f"  {k}: {v:.8e}" if np.isfinite(v) else f"  {k}: nan")
    assert np.isfinite(feats["spatial_snr_ratio"]), "SNR should be finite"
    assert np.isfinite(feats["cross_noise_ratio"]), "cross_noise should be finite"
    # skew/kurt are NaN because random noise has no flat zone
    for suffix in ("y", "cr", "cb"):
        assert np.isnan(feats[f"skew_noise_{suffix}"]), f"skew_{suffix} should be NaN (no flat zone)"
        assert np.isnan(feats[f"kurt_noise_{suffix}"]), f"kurt_{suffix} should be NaN (no flat zone)"
    print("  SNR + cross_noise finite; skew/kurt NaN (no flat zone) ✔")

    # ── Test 2: Flat grey → all NaN (sigma < guard, masks may be empty)
    flat = np.full((CROP_SIZE, CROP_SIZE, 3), 128, dtype=np.uint8)
    feats2 = extract_spatial_features(flat)
    print("\nTest 2 — Flat grey (128,128,128):")
    for k, v in feats2.items():
        print(f"  {k}: {v:.8e}" if np.isfinite(v) else f"  {k}: nan")
    # Flat → all residuals zero → sigma < guard → NaN for skew/kurt
    for suffix in ("y", "cr", "cb"):
        assert np.isnan(feats2[f"skew_noise_{suffix}"]), f"skew_{suffix} not NaN"
        assert np.isnan(feats2[f"kurt_noise_{suffix}"]), f"kurt_{suffix} not NaN"
    print("  All skew/kurt are NaN as expected.")

    # ── Test 3: Audit helper ─────────────────────────────────────────
    audit = extract_spatial_audit(dummy)
    print("\nTest 3 — Audit (random noise):")
    print(f"  edge_pixel_count:  {audit['edge_pixel_count']}")
    print(f"  flat_pixel_count:  {audit['flat_pixel_count']}")
    print(f"  T_edge:            {audit['T_edge']:.2f}")
    print(f"  T_flat:            {audit['T_flat']:.2f}")
    print(f"  spatial_snr_is_nan:{audit['spatial_snr_is_nan']}")
    print(f"  cross_noise_is_nan:{audit['cross_noise_is_nan']}")
    print(f"  residual_shape:    {audit['residual_shape']}")
    sig_y = audit['sigma_y']
    print(f"  sigma_y:           {sig_y:.4f}" if np.isfinite(sig_y) else "  sigma_y:           nan")
    print(f"  skew_nan_count:    {audit['skew_nan_count']}")
    assert audit["residual_shape"] == (254, 254), "Wrong residual shape!"

    # ── Test 4: Shape consistency check ──────────────────────────────
    Y = dummy[:, :, 0].astype(np.float64)
    G = _compute_gradient_valid(Y)
    R = _compute_residual_sq3(Y)
    assert G.shape == (254, 254), f"Gradient shape {G.shape}"
    assert R.shape == (254, 254), f"Residual shape {R.shape}"
    print("\nTest 4 — Shape consistency: G=(254,254), R=(254,254) ✓")
    # ── Test 5: Real image — skew/kurt should be finite (enough flat zone)
    import glob
    real_files = sorted(glob.glob("data/processed/ADM/ai/*.npy"))[:1]
    if real_files:
        real_arr = np.load(real_files[0], allow_pickle=False)
        feats5 = extract_spatial_features(real_arr)
        print("\nTest 5 — Real image (ADM/ai):")
        for k, v in feats5.items():
            print(f"  {k}: {v:.8e}" if np.isfinite(v) else f"  {k}: nan")
        finite_5 = sum(1 for v in feats5.values() if np.isfinite(v))
        print(f"  finite={finite_5}/8")
        assert finite_5 == 8, f"Expected 8 finite on real image, got {finite_5}"
    else:
        print("\nTest 5 — SKIPPED (no real data)")
    print("\n=== All smoke tests passed. ===")
