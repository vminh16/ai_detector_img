"""
src/feature_extraction/frequency.py
=========================
Group 1 — Frequency Domain feature extraction (locked spec).

Extracts 6 features from the Y channel of preprocessed 256x256 YCrCb arrays
produced by the hardened preprocessing pipeline (reflect padding, misaligned
center crop, deterministic JPEG bottleneck -> .npy).

Features
--------
frs_mid_variance      : Normalised variance (CV squared) of azimuthal mean
                        power, radii r in [8, 32].  Higher values indicate
                        spectral "spikiness" typical of upsampling artifacts.
ps_alpha              : Slope of 1D power-spectrum log-log fit, r in [20, 64].
                        Natural images ~ 2.0-3.5; AI images may deviate.
ps_deviation_variance : Variance of log-domain residuals from the fitted line.
dct_mid_mean          : Mean of squared mid-band DCT coefficients
                        (zigzag indices 10-40).
dct_mid_variance      : Variance of the same.
dct_mid_skewness      : Skewness of the same.

NOTE on DCT features
--------------------
The 8x8 DCT grid coincides with the JPEG bottleneck grid in the hardened
preprocessing pipeline.  These three statistics therefore conflate
quantisation residue with genuine generator artifacts.  They should **not**
be interpreted as a pure intrinsic AI fingerprint; consider dropping them
if Feature Importance is suspiciously high during model selection.

Dependencies: numpy, scipy (>=1.7), pandas, tqdm.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.fft import dctn
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# Constants (locked — do not modify without spec update)
# ────────────────────────────────────────────────────────────────────────

CROP_SIZE: int = 256
"""Spatial size of preprocessed arrays (H x W)."""

FRS_R_MIN: int = 8
FRS_R_MAX: int = 32
"""Radial band for Fourier Ring Statistic (inclusive)."""

PS_R_MIN: int = 20
PS_R_MAX: int = 64
"""Radial band for 1D power-spectrum log-log fit (inclusive)."""

DCT_BLOCK: int = 8
N_BLOCKS: int = CROP_SIZE // DCT_BLOCK  # 32
"""8x8 block size and number of blocks per dimension for DCT analysis."""

DCT_ZZ_START: int = 10
DCT_ZZ_END: int = 40
"""Zigzag index range for mid-band DCT coefficients (inclusive)."""

EPS: float = 1e-8
"""Global epsilon for numerical guards (denominators, log arguments)."""

NOISE_FLOOR: float = 1.0 / (CROP_SIZE * CROP_SIZE)
"""Parseval-normalised minimum meaningful ring-mean power (≈ 1.53e-5).

Equals σ²/N² for σ² = 1 LSB² quantisation noise on a 256×256 grid.
Ring-mean values below this fall below the JPEG bottleneck noise floor
(Q ∈ [90, 98]).  Used as:
- Additive log floor in power-spectrum decay fit (prevents log → −∞).
- Signal-sufficiency guard in FRS (returns sentinel when too weak).
"""

FEATURE_KEYS: tuple[str, ...] = (
    "frs_mid_variance",
    "dct_mid_mean",
    "dct_mid_variance",
    "dct_mid_skewness",
    "ps_alpha",
    "ps_deviation_variance",
)
"""Locked output keys — order and naming must not change."""


# ────────────────────────────────────────────────────────────────────────
# Precomputed lookup tables (module-level singletons, computed once)
# ────────────────────────────────────────────────────────────────────────

def _build_radius_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return (R_float, R_int) for the 256x256 centred frequency grid.

    Coordinates span [-128, 127] on both axes (integer meshgrid).
    R_int is computed via integer-domain squared radius followed by
    ``floor(sqrt(r²) + 0.5)`` to guarantee 100% cross-platform
    determinism.  Since ``sqrt(integer)`` is never half-integer,
    the 0.5-boundary ambiguity of ``np.round`` (banker's rounding)
    cannot arise.
    """
    coords = np.arange(CROP_SIZE) - (CROP_SIZE // 2)  # [-128 … 127]
    R_float = np.hypot(coords[:, None], coords[None, :])
    # Integer squared radius — exact arithmetic, no FP round-off.
    R_sq = (coords[:, None] ** 2 + coords[None, :] ** 2).astype(np.int64)
    R_int = np.floor(np.sqrt(R_sq.astype(np.float64)) + 0.5).astype(np.intp)
    return R_float, R_int


_RADIUS_FLOAT, _RADIUS_INT = _build_radius_grid()
_R_INT_MAX: int = int(_RADIUS_INT.max())  # ~181 for 256x256

# Precomputed 1-D views and ring pixel counts — avoids per-image
# ravel() allocation and bincount() recomputation in _ring_mean().
_RADIUS_INT_FLAT: np.ndarray = _RADIUS_INT.ravel()
_RING_COUNT: np.ndarray = np.bincount(
    _RADIUS_INT_FLAT, minlength=_R_INT_MAX + 1,
).astype(np.float64)
_RING_COUNT_SAFE: np.ndarray = np.where(_RING_COUNT > 0, _RING_COUNT, 1.0)
"""Per-ring pixel counts (float64); zeros replaced by 1.0 to
avoid division-by-zero in _ring_mean without branching."""


def _zigzag_indices_8x8() -> np.ndarray:
    """Return shape-(64, 2) array of (row, col) in JPEG zigzag scan order.

    Explicitly defined constant table — no external JPEG library dependency.
    Verified against ITU-T T.81 Annex A Figure A.6.
    """
    # fmt: off
    return np.array([
        [0, 0], [0, 1], [1, 0], [2, 0], [1, 1], [0, 2], [0, 3], [1, 2],
        [2, 1], [3, 0], [4, 0], [3, 1], [2, 2], [1, 3], [0, 4], [0, 5],
        [1, 4], [2, 3], [3, 2], [4, 1], [5, 0], [6, 0], [5, 1], [4, 2],
        [3, 3], [2, 4], [1, 5], [0, 6], [0, 7], [1, 6], [2, 5], [3, 4],
        [4, 3], [5, 2], [6, 1], [7, 0], [7, 1], [6, 2], [5, 3], [4, 4],
        [3, 5], [2, 6], [1, 7], [2, 7], [3, 6], [4, 5], [5, 4], [6, 3],
        [7, 2], [7, 3], [6, 4], [5, 5], [4, 6], [3, 7], [4, 7], [5, 6],
        [6, 5], [7, 4], [7, 5], [6, 6], [5, 7], [6, 7], [7, 6], [7, 7],
    ], dtype=np.intp)
    # fmt: on


_ZIGZAG_RC: np.ndarray = _zigzag_indices_8x8()
_ZZ_MID: np.ndarray = _ZIGZAG_RC[DCT_ZZ_START : DCT_ZZ_END + 1]  # (31, 2)
_ZZ_ROWS: np.ndarray = _ZZ_MID[:, 0]  # row indices for mid-band positions
_ZZ_COLS: np.ndarray = _ZZ_MID[:, 1]  # col indices for mid-band positions


# ────────────────────────────────────────────────────────────────────────
# Input validation
# ────────────────────────────────────────────────────────────────────────

def _validate_ycrcb_array(arr: np.ndarray) -> np.ndarray:
    """Validate *arr* shape/dtype and return the Y channel as float64.

    Parameters
    ----------
    arr : np.ndarray
        Expected shape ``(256, 256, 3)``, dtype uint8 / float32 / float64.
        Channel order: 0=Y, 1=Cr, 2=Cb (OpenCV ``COLOR_BGR2YCrCb``).

    Returns
    -------
    np.ndarray
        Copy of channel 0 (Y) cast to float64, shape ``(256, 256)``.

    Raises
    ------
    TypeError
        If *arr* is not an ndarray or has non-numeric dtype.
    ValueError
        If shape does not match ``(256, 256, 3)``.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(arr).__name__}")
    if arr.ndim != 3 or arr.shape != (CROP_SIZE, CROP_SIZE, 3):
        raise ValueError(
            f"Expected shape ({CROP_SIZE}, {CROP_SIZE}, 3), got {arr.shape}"
        )
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"Expected numeric dtype, got {arr.dtype}")
    # Return a *copy* cast to float64 — never mutate the caller's array.
    return arr[:, :, 0].astype(np.float64)


# ────────────────────────────────────────────────────────────────────────
# Core math helpers (private)
# ────────────────────────────────────────────────────────────────────────

def _compute_power_spectrum(Y: np.ndarray) -> np.ndarray:
    """Parseval-normalised 2-D power spectrum of the Y channel.

    ``F = fft2(Y) / N**2``  then  ``P = |fftshift(F)|**2``

    Parameters
    ----------
    Y : np.ndarray
        Float64 luminance channel, shape ``(256, 256)``.

    Returns
    -------
    np.ndarray
        Power spectrum, shape ``(256, 256)``, same dtype.
    """
    N = CROP_SIZE
    F = np.fft.fft2(Y) / (N * N)
    P: np.ndarray = np.abs(np.fft.fftshift(F)) ** 2
    return P


def _ring_mean(P: np.ndarray) -> np.ndarray:
    """Azimuthal mean power E[r] for each integer radius *r*.

    Uses module-level singletons ``_RADIUS_INT_FLAT`` and
    ``_RING_COUNT_SAFE`` so that per-image cost is a single
    ``bincount`` call with no temporary array allocation.

    Returns
    -------
    np.ndarray
        1-D array of length ``_R_INT_MAX + 1``.  Index *r* holds the mean
        power at radius *r*.  Radii with zero pixel count map to 0.0.
    """
    ring_sum = np.bincount(
        _RADIUS_INT_FLAT, weights=P.ravel(), minlength=_R_INT_MAX + 1,
    )
    return ring_sum / _RING_COUNT_SAFE


def _extract_frs(ring: np.ndarray) -> float:
    """FRS mid-band normalised variance (coefficient of variation squared).

    ``frs_mid_variance = Var(E[8..32]) / (Mean(E[8..32])**2 + eps)``

    Despite the legacy key name ``frs_mid_variance``, this is actually CV**2,
    which normalises for overall brightness so that the statistic measures
    spectral "spikiness" rather than raw energy level.

    Returns 0.0 (sentinel) when mean ring power falls below
    ``NOISE_FLOOR``, indicating insufficient spectral energy for a
    meaningful statistic (e.g. near-black / flat-background images).
    """
    E_mid = ring[FRS_R_MIN : FRS_R_MAX + 1]  # 25 values (r=8,9,...,32)
    mean_E = float(np.mean(E_mid))
    if mean_E < NOISE_FLOOR:
        # Signal below JPEG quantisation noise floor — CV² is meaningless.
        return 0.0
    var_E = float(np.var(E_mid))
    return var_E / (mean_E ** 2 + EPS)


def _extract_power_decay(ring: np.ndarray) -> tuple[float, float]:
    """1-D power-spectrum log-log linear fit on r in [20, 64].

    Model::

        log c(r) = -alpha * log(r) + b
        alpha = -slope

    Uses ``NOISE_FLOOR`` (JPEG quantisation noise level, ≈ 1.53e-5) as
    the additive log floor instead of a blind ``EPS``.  This prevents
    near-zero ring means from producing extreme negative log values that
    act as leverage points and distort the OLS fit.

    Returns ``(0.0, 0.0)`` sentinel when *all* ring means in the fit
    range fall below ``NOISE_FLOOR`` (image too flat for meaningful
    analysis).

    Returns
    -------
    alpha : float
        Power-law exponent (positive for natural-image-like decay).
    dev_var : float
        Variance of log-domain residuals vs. the fitted line.
    """
    r_range = np.arange(PS_R_MIN, PS_R_MAX + 1, dtype=np.float64)
    c_r = ring[PS_R_MIN : PS_R_MAX + 1].astype(np.float64)

    # Guard: if entire fit band is below noise floor, signal is
    # insufficient — return deterministic sentinel.
    if np.all(c_r < NOISE_FLOOR):
        return 0.0, 0.0

    log_r = np.log(r_range)
    # Additive floor = NOISE_FLOOR (physically: 1 LSB² Parseval-normalised
    # quantisation noise).  Clamps log to ≈ −11.1 instead of −18.4 (EPS),
    # eliminating leverage-point distortion on near-flat spectral bands.
    log_c = np.log(c_r + NOISE_FLOOR)

    # Ordinary least squares (degree-1 polynomial)
    slope, intercept = np.polyfit(log_r, log_c, 1)
    alpha = -slope

    fitted = slope * log_r + intercept
    residuals = log_c - fitted
    dev_var = float(np.var(residuals))

    return float(alpha), dev_var


def _safe_skewness(a: np.ndarray) -> float:
    """Population skewness with safe denominator guard.

    Returns 0.0 when sigma < EPS (near-constant data), avoiding NaN/Inf.
    """
    mean_a = np.mean(a)
    sigma = np.std(a)
    if sigma < EPS:
        return 0.0
    m3 = float(np.mean((a - mean_a) ** 3))
    return m3 / (sigma ** 3 + EPS)


def _get_dct_midband_squared(Y: np.ndarray) -> np.ndarray:
    """Return the pool of squared mid-band DCT coefficients.

    Shared by ``_extract_dct_stats`` and by visualisation helpers to
    guarantee identical computation.

    Parameters
    ----------
    Y : np.ndarray
        Float64 luminance channel, shape ``(256, 256)``.

    Returns
    -------
    np.ndarray
        1-D array of length ``N_BLOCKS**2 * (DCT_ZZ_END - DCT_ZZ_START + 1)``
        = 32*32*31 = 31 744.
    """
    blocks = Y.reshape(N_BLOCKS, DCT_BLOCK, N_BLOCKS, DCT_BLOCK)
    blocks = blocks.transpose(0, 2, 1, 3)  # (32, 32, 8, 8)
    dct_blocks = dctn(blocks, type=2, norm="ortho", axes=(-2, -1))
    mid_coeffs = dct_blocks[:, :, _ZZ_ROWS, _ZZ_COLS]  # (32, 32, 31)
    return (mid_coeffs ** 2).ravel()


def _extract_dct_stats(Y: np.ndarray) -> tuple[float, float, float]:
    """DCT mid-band statistics from all 8x8 blocks.

    For each of the 32x32 = 1024 blocks, compute ``scipy.fft.dctn``
    (type-II, orthonormal), extract zigzag indices 10..40 (31 coeffs),
    square them (A = coeff**2), then compute mean / variance / skewness
    over the full pool of 31 744 values.

    CAVEAT: The 8x8 DCT grid coincides with the JPEG bottleneck grid in the
    hardened preprocessing pipeline.  These statistics therefore conflate
    quantisation residue with genuine generator artifacts.  Do **not**
    interpret them as a pure intrinsic fingerprint.
    """
    A = _get_dct_midband_squared(Y)
    dct_mean = float(np.mean(A))
    dct_var = float(np.var(A))
    dct_skew = _safe_skewness(A)
    return dct_mean, dct_var, dct_skew


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

def extract_frequency_features(ycbcr_npy: np.ndarray) -> dict[str, float]:
    """Extract the 6 locked Group-1 frequency-domain features.

    Parameters
    ----------
    ycbcr_npy : np.ndarray
        Shape ``(256, 256, 3)`` YCrCb array (channel 0 = Y).
        Accepted dtypes: uint8, float32, float64.

    Returns
    -------
    dict[str, float]
        Exactly 6 keys as defined in ``FEATURE_KEYS``.
        Guaranteed finite (no NaN / Inf).
    """
    Y = _validate_ycrcb_array(ycbcr_npy)

    # — Fourier domain ─────────────────────────────────────────────────
    P = _compute_power_spectrum(Y)
    ring = _ring_mean(P)

    frs_mid_variance = _extract_frs(ring)
    ps_alpha, ps_deviation_variance = _extract_power_decay(ring)

    # — DCT domain ─────────────────────────────────────────────────────
    dct_mid_mean, dct_mid_variance, dct_mid_skewness = _extract_dct_stats(Y)

    result: dict[str, float] = {
        "frs_mid_variance": frs_mid_variance,
        "dct_mid_mean": dct_mid_mean,
        "dct_mid_variance": dct_mid_variance,
        "dct_mid_skewness": dct_mid_skewness,
        "ps_alpha": ps_alpha,
        "ps_deviation_variance": ps_deviation_variance,
    }

    # Safety net — should never trigger if numeric guards above are correct
    for key, val in result.items():
        if not np.isfinite(val):
            logger.error(
                "Non-finite value for '%s' = %s; clamped to 0.0", key, val,
            )
            result[key] = 0.0

    return result


def extract_frequency_features_from_file(
    path: str | Path,
) -> dict[str, float]:
    """Load a ``.npy`` file and extract frequency features.

    Convenience wrapper for single-file use in notebooks.
    """
    arr = np.load(Path(path), allow_pickle=False)
    return extract_frequency_features(arr)


def extract_frequency_batch(
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
        Columns: ``file_path``, 6 feature columns, ``status``, ``error``.
        Failed files have ``status="error"`` and NaN features.
    """
    records: list[dict] = []
    iterator = tqdm(
        paths, desc="Extracting freq features", disable=not show_progress,
    )
    for p in iterator:
        p_str = str(Path(p))
        rec: dict = {"file_path": p_str}
        try:
            arr = np.load(p_str, allow_pickle=False)
            feats = extract_frequency_features(arr)
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


def extract_frequency_dataset(
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
        6 feature columns, ``status``, ``error``.
    """
    input_root = Path(input_root)
    npy_files = sorted(input_root.rglob("*.npy"))

    if not npy_files:
        logger.warning("No .npy files found under %s", input_root)
        return pd.DataFrame()

    logger.info("Found %d .npy files under %s", len(npy_files), input_root)

    df = extract_frequency_batch(npy_files, show_progress=show_progress)

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
    dummy = rng.integers(0, 256, size=(CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    feats = extract_frequency_features(dummy)
    print("Smoke-test features (random noise input):")
    for k, v in feats.items():
        print(f"  {k}: {v:.8e}")
