"""
src/feature_extraction/microtexture.py
============================
Group 3 — Micro-Texture Residual & Chroma LBP feature extraction
(locked spec).

Extracts 10 features from the Cr / Cb channels of preprocessed 256×256
YCrCb arrays produced by the hardened preprocessing pipeline (reflect
padding, misaligned center crop, deterministic JPEG bottleneck → .npy).

Features
--------
**Advanced SRM on Cr (6 dims)** — zero-sum high-pass residual statistics
on the centred chroma channel.  These quantify micro-structural noise
left by generator upsampling layers, but should NOT be interpreted as
an intrinsic fingerprint immune to content; JPEG bottleneck residue and
natural texture edges also contribute.  ``valid`` convolution is used
to avoid boundary artifacts from padding.

srm_square3_mar_cr     : Mean Absolute Residual of 3×3 multi-directional
                         Laplacian filter on Cr'.
srm_square3_energy_cr  : Mean squared residual (L2 energy) of same.
srm_edge3_mar_cr       : MAR of 3×3 edge-concentrated Laplacian on Cr'.
srm_edge3_energy_cr    : Energy of same.
srm_square5_mar_cr     : MAR of 5×5 extended Laplacian on Cr'.
srm_square5_energy_cr  : Energy of same.

**Chroma LBP on Cr and Cb (4 dims)** — rotation-variant Local Binary
Patterns that preserve axis-aligned directional texture.  Generator
networks often leave axis-aligned checkerboard artifacts from 2D matrix
transposition in upsampling layers; rotation-variant LBP retains this
signal.  Risk: LBP at radius 1 may also respond to JPEG 8×8 block
boundaries (grid inversion), causing non-uniformity to *increase* on
over-smoothed AI images — a non-linear effect the downstream model
must learn.

lbp_nonuniform_ratio_cr : Fraction of Cr core pixels with non-uniform
                          LBP pattern (≥ 3 circular bit transitions).
lbp_entropy_cr          : Shannon entropy (log₂) of the 59-bin uniform
                          LBP histogram on Cr.
lbp_nonuniform_ratio_cb : Same ratio on the Cb channel.
lbp_entropy_cb          : Same entropy on the Cb channel.

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

SRM_ENERGY_THRESHOLD: float = 1e-12
"""Minimum residual energy to consider the SRM channel signal-bearing.
Below this, both MAR and Energy are set to 0.0 sentinel."""

LBP_RADIUS: int = 1
"""Radius of the LBP neighbourhood."""

LBP_POINTS: int = 8
"""Number of sampling points on the LBP circle."""

LBP_BINS: int = 59
"""Number of histogram bins after uniform mapping (58 uniform + 1 non-uniform)."""

EPS: float = 1e-12
"""Epsilon for numerical guards where strictly needed."""

FEATURE_KEYS: tuple[str, ...] = (
    "srm_square3_mar_cr",
    "srm_square3_energy_cr",
    "srm_edge3_mar_cr",
    "srm_edge3_energy_cr",
    "srm_square5_mar_cr",
    "srm_square5_energy_cr",
    "lbp_nonuniform_ratio_cr",
    "lbp_entropy_cr",
    "lbp_nonuniform_ratio_cb",
    "lbp_entropy_cb",
)
"""Locked output keys — order and naming must not change."""


# ────────────────────────────────────────────────────────────────────────
# SRM Kernels (hardcoded, zero-sum high-pass)
# ────────────────────────────────────────────────────────────────────────

_KERNEL_SQUARE3: np.ndarray = np.array(
    [[-1,  2, -1],
     [ 2, -4,  2],
     [-1,  2, -1]],
    dtype=np.float64,
)

_KERNEL_EDGE3: np.ndarray = np.array(
    [[ 0, -1,  0],
     [-1,  4, -1],
     [ 0, -1,  0]],
    dtype=np.float64,
)

_KERNEL_SQUARE5: np.ndarray = 0.25 * np.array(
    [[0,  0,  1,  0, 0],
     [0,  0, -2,  0, 0],
     [1, -2,  4, -2, 1],
     [0,  0, -2,  0, 0],
     [0,  0,  1,  0, 0]],
    dtype=np.float64,
)

# Compile kernel list for iteration: (name_prefix, kernel)
_SRM_KERNELS: list[tuple[str, np.ndarray]] = [
    ("square3", _KERNEL_SQUARE3),
    ("edge3",   _KERNEL_EDGE3),
    ("square5", _KERNEL_SQUARE5),
]


# ────────────────────────────────────────────────────────────────────────
# LBP Look-Up Table (59 bins: 58 uniform + 1 non-uniform)
# ────────────────────────────────────────────────────────────────────────

def _build_lbp_lut_59() -> np.ndarray:
    """Build deterministic LUT mapping 256 raw LBP codes → 59 bins.

    A pattern is *uniform* iff its number of circular 0↔1 transitions
    (treating the 8-bit code as a ring) is ≤ 2.  There are exactly
    58 such patterns for P=8.  All non-uniform patterns map to bin 58.

    Returns
    -------
    np.ndarray
        Shape ``(256,)``, dtype int32.  Values in ``[0, 58]``.
    """
    lut = np.empty(256, dtype=np.int32)
    uniform_idx = 0
    for code in range(256):
        # Count circular bit transitions
        bits = code
        transitions = 0
        for k in range(LBP_POINTS):
            b_curr = (bits >> k) & 1
            b_next = (bits >> ((k + 1) % LBP_POINTS)) & 1
            if b_curr != b_next:
                transitions += 1
        if transitions <= 2:
            lut[code] = uniform_idx
            uniform_idx += 1
        else:
            lut[code] = LBP_BINS - 1  # bin 58
    return lut


# Module-level singleton — computed once at import time.
_LBP_LUT_59: np.ndarray = _build_lbp_lut_59()


# ────────────────────────────────────────────────────────────────────────
# LBP neighbour offsets (locked convention)
# ────────────────────────────────────────────────────────────────────────

# 8 neighbours, clockwise starting from top-left around centre pixel.
# Bit k corresponds to offset _LBP_OFFSETS[k].
# bit_k = 1 if neighbour >= centre else 0
# code  = Σ bit_k << k  for k in 0..7
_LBP_OFFSETS: list[tuple[int, int]] = [
    (-1, -1),  # 0: top-left
    (-1,  0),  # 1: top
    (-1, +1),  # 2: top-right
    ( 0, +1),  # 3: right
    (+1, +1),  # 4: bottom-right
    (+1,  0),  # 5: bottom
    (+1, -1),  # 6: bottom-left
    ( 0, -1),  # 7: left
]


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
# SRM helpers (private)
# ────────────────────────────────────────────────────────────────────────

def _convolve_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """True 2D convolution (NOT correlation) with mode='valid'.

    Uses ``scipy.signal.convolve2d`` which implements mathematical
    convolution (kernel is flipped).  Mode ``valid`` ensures no
    boundary padding artifacts contaminate the residual.

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


def _compute_srm_pair(residual: np.ndarray) -> tuple[float, float]:
    """Compute (MAR, Energy) from a residual map with OOD sentinel guard.

    1. Energy = mean(R²).
    2. If Energy < SRM_ENERGY_THRESHOLD → (0.0, 0.0).
    3. Otherwise MAR = mean(|R|).

    Parameters
    ----------
    residual : np.ndarray
        2D residual array (output of valid convolution).

    Returns
    -------
    mar, energy : float
    """
    energy = float(np.mean(residual ** 2))
    if energy < SRM_ENERGY_THRESHOLD:
        return 0.0, 0.0
    mar = float(np.mean(np.abs(residual)))
    return mar, energy


def _extract_srm_features(Cr: np.ndarray) -> dict[str, float]:
    """Extract all 6 SRM features from the Cr channel.

    Steps
    -----
    1. Centre: Cr' = Cr − 128.
    2. Convolve Cr' with each of the 3 zero-sum kernels (mode='valid').
    3. Compute (MAR, Energy) pair per residual.

    Returns
    -------
    dict with 6 keys: srm_{name}_mar_cr, srm_{name}_energy_cr.
    """
    Cr_centred = Cr - 128.0
    result: dict[str, float] = {}
    for name, kernel in _SRM_KERNELS:
        residual = _convolve_valid(Cr_centred, kernel)
        mar, energy = _compute_srm_pair(residual)
        result[f"srm_{name}_mar_cr"] = mar
        result[f"srm_{name}_energy_cr"] = energy
    return result


# ────────────────────────────────────────────────────────────────────────
# LBP helpers (private)
# ────────────────────────────────────────────────────────────────────────

def _lbp_codes_radius1_points8(C_int: np.ndarray) -> np.ndarray:
    """Compute rotation-variant LBP codes on the core region.

    Vectorised via array slicing — no Python pixel loops.

    The 8 neighbours are sampled clockwise starting from top-left::

        bit 0: (-1,-1)  bit 1: (-1, 0)  bit 2: (-1,+1)
        bit 7: ( 0,-1)   [centre]       bit 3: ( 0,+1)
        bit 6: (+1,-1)  bit 5: (+1, 0)  bit 4: (+1,+1)

    Encoding: ``bit_k = 1 if neighbour >= centre else 0``
              ``code = Σ bit_k << k``

    Parameters
    ----------
    C_int : np.ndarray
        Integer-valued 2D array (int32), shape ``(256, 256)``.

    Returns
    -------
    np.ndarray
        LBP code map, shape ``(254, 254)``, dtype uint8.
        Values in ``[0, 255]``.
    """
    R = LBP_RADIUS  # 1
    # Core region: [R:-R, R:-R] → [1:255, 1:255] → 254×254
    centre = C_int[R:-R, R:-R]

    code = np.zeros_like(centre, dtype=np.uint16)
    for k, (dy, dx) in enumerate(_LBP_OFFSETS):
        # Neighbour slice aligned with centre
        row_start = R + dy
        row_end = row_start + centre.shape[0]
        col_start = R + dx
        col_end = col_start + centre.shape[1]
        neighbour = C_int[row_start:row_end, col_start:col_end]
        code |= ((neighbour >= centre).astype(np.uint16) << k)

    return code.astype(np.uint8)


def _lbp_histogram_stats(
    codes: np.ndarray,
) -> tuple[float, float]:
    """Compute non-uniformity ratio and entropy from LBP codes.

    Parameters
    ----------
    codes : np.ndarray
        Raw LBP codes (uint8), any 2D shape.

    Returns
    -------
    nonuniform_ratio, entropy : float
        Non-uniformity is ``p[58]`` (the non-uniform bin probability).
        Entropy is ``-Σ p_i log₂(p_i)`` for ``p_i > 0``.
    """
    mapped = _LBP_LUT_59[codes.ravel()]
    counts = np.bincount(mapped, minlength=LBP_BINS).astype(np.float64)

    total = counts.sum()
    if total == 0:
        return 0.0, 0.0

    prob = counts / total

    # Non-uniformity ratio = probability mass in the non-uniform bin (58)
    nonuniform_ratio = float(prob[LBP_BINS - 1])

    # Shannon entropy in bits (log₂), guarded against log2(0)
    mask = prob > 0
    entropy = float(-np.sum(prob[mask] * np.log2(prob[mask])))

    return nonuniform_ratio, entropy


def _extract_lbp_channel(C: np.ndarray) -> tuple[float, float]:
    """Extract LBP features for one chroma channel.

    Parameters
    ----------
    C : np.ndarray
        Single chroma channel, float64, shape ``(256, 256)``.

    Returns
    -------
    nonuniform_ratio, entropy : float

    Notes
    -----
    - Integer cast via ``rint`` + ``astype(int32)`` shields against
      IEEE 754 half-LSB noise on optically flat chroma regions.
    - Sentinel fast-path: if peak-to-peak == 0 (constant channel), both
      outputs are 0.0 — there is no texture to measure.
    """
    C_int = np.rint(C).astype(np.int32)

    # Sentinel: constant channel → no texture
    if int(C_int.max() - C_int.min()) == 0:
        return 0.0, 0.0

    codes = _lbp_codes_radius1_points8(C_int)
    return _lbp_histogram_stats(codes)


# ────────────────────────────────────────────────────────────────────────
# Internal aggregator
# ────────────────────────────────────────────────────────────────────────

def _extract_all(Cr: np.ndarray, Cb: np.ndarray) -> dict[str, float]:
    """Extract all 10 features from validated Cr and Cb channels."""
    feats = _extract_srm_features(Cr)

    nu_cr, ent_cr = _extract_lbp_channel(Cr)
    nu_cb, ent_cb = _extract_lbp_channel(Cb)
    feats["lbp_nonuniform_ratio_cr"] = nu_cr
    feats["lbp_entropy_cr"] = ent_cr
    feats["lbp_nonuniform_ratio_cb"] = nu_cb
    feats["lbp_entropy_cb"] = ent_cb

    return feats


# ────────────────────────────────────────────────────────────────────────
# Audit helper
# ────────────────────────────────────────────────────────────────────────

def extract_microtexture_audit(ycbcr_npy: np.ndarray) -> dict[str, object]:
    """Extract features plus intermediate diagnostics for auditing.

    Returns all 10 features plus additional fields:

    - ``srm_energy_zero_count``: How many of the 3 SRM kernels hit the
      energy sentinel (energy < threshold).
    - ``residual_shapes``: Dict mapping kernel name → residual shape tuple.
    - ``lbp_cr_sentinel_used``: Whether Cr LBP used the ptp==0 sentinel.
    - ``lbp_cb_sentinel_used``: Whether Cb LBP used the ptp==0 sentinel.
    - ``lbp_core_shape``: Shape of the LBP core region.
    - ``cr_ptp``, ``cb_ptp``: Peak-to-peak of integer-cast channels.
    """
    _Y, Cr, Cb = _validate_ycrcb_array(ycbcr_npy)

    feats = _extract_all(Cr, Cb)

    # SRM audit
    Cr_centred = Cr - 128.0
    energy_zero_count = 0
    residual_shapes: dict[str, tuple[int, int]] = {}
    for name, kernel in _SRM_KERNELS:
        residual = _convolve_valid(Cr_centred, kernel)
        residual_shapes[name] = residual.shape
        energy = float(np.mean(residual ** 2))
        if energy < SRM_ENERGY_THRESHOLD:
            energy_zero_count += 1

    # LBP audit
    Cr_int = np.rint(Cr).astype(np.int32)
    Cb_int = np.rint(Cb).astype(np.int32)
    cr_ptp = int(Cr_int.max() - Cr_int.min())
    cb_ptp = int(Cb_int.max() - Cb_int.min())
    core_shape = (CROP_SIZE - 2 * LBP_RADIUS, CROP_SIZE - 2 * LBP_RADIUS)

    audit: dict[str, object] = {
        **feats,
        "srm_energy_zero_count": energy_zero_count,
        "residual_shapes": residual_shapes,
        "lbp_cr_sentinel_used": cr_ptp == 0,
        "lbp_cb_sentinel_used": cb_ptp == 0,
        "lbp_core_shape": core_shape,
        "cr_ptp": cr_ptp,
        "cb_ptp": cb_ptp,
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

def extract_microtexture_features(ycbcr_npy: np.ndarray) -> dict[str, float]:
    """Extract the 10 locked Group-3 micro-texture features.

    Parameters
    ----------
    ycbcr_npy : np.ndarray
        Shape ``(256, 256, 3)`` YCrCb array.
        Channel 0 = Y, 1 = Cr, 2 = Cb.
        Accepted dtypes: uint8, float32, float64.

    Returns
    -------
    dict[str, float]
        Exactly 10 keys as defined in ``FEATURE_KEYS``.
        Guaranteed finite (no NaN / Inf).
    """
    _Y, Cr, Cb = _validate_ycrcb_array(ycbcr_npy)
    result = _extract_all(Cr, Cb)

    # Safety net
    for key, val in result.items():
        if not np.isfinite(val):
            logger.error(
                "Non-finite value for '%s' = %s; clamped to 0.0", key, val,
            )
            result[key] = 0.0

    return result


def extract_microtexture_features_from_file(
    path: str | Path,
) -> dict[str, float]:
    """Load a ``.npy`` file and extract microtexture features.

    Convenience wrapper for single-file use in notebooks.
    """
    arr = np.load(Path(path), allow_pickle=False)
    return extract_microtexture_features(arr)


def extract_microtexture_batch(
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
        Columns: ``file_path``, 10 feature columns, ``status``, ``error``.
        Failed files have ``status="error"`` and NaN features.
    """
    records: list[dict] = []
    iterator = tqdm(
        paths,
        desc="Extracting microtexture features",
        disable=not show_progress,
    )
    for p in iterator:
        p_str = str(Path(p))
        rec: dict = {"file_path": p_str}
        try:
            arr = np.load(p_str, allow_pickle=False)
            feats = extract_microtexture_features(arr)
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


def extract_microtexture_dataset(
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
        10 feature columns, ``status``, ``error``.
    """
    input_root = Path(input_root)
    npy_files = sorted(input_root.rglob("*.npy"))

    if not npy_files:
        logger.warning("No .npy files found under %s", input_root)
        return pd.DataFrame()

    logger.info("Found %d .npy files under %s", len(npy_files), input_root)

    df = extract_microtexture_batch(npy_files, show_progress=show_progress)

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

    # ── Test 1: Random noise (high texture) ──────────────────────────
    dummy = rng.integers(0, 256, size=(CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    feats = extract_microtexture_features(dummy)
    print("Test 1 — Random noise:")
    for k, v in feats.items():
        print(f"  {k}: {v:.8e}")
    assert all(np.isfinite(v) for v in feats.values()), "Non-finite!"

    # ── Test 2: Flat grey (128,128,128) → all sentinels ──────────────
    flat = np.full((CROP_SIZE, CROP_SIZE, 3), 128, dtype=np.uint8)
    feats2 = extract_microtexture_features(flat)
    print("\nTest 2 — Flat grey (128,128,128):")
    for k, v in feats2.items():
        print(f"  {k}: {v:.8e}")
    # Cr' = 0 everywhere → all SRM residuals exactly 0 → sentinel
    for name in ("square3", "edge3", "square5"):
        assert feats2[f"srm_{name}_mar_cr"] == 0.0
        assert feats2[f"srm_{name}_energy_cr"] == 0.0
    # Constant Cr/Cb → ptp==0 → LBP sentinel
    assert feats2["lbp_nonuniform_ratio_cr"] == 0.0
    assert feats2["lbp_entropy_cr"] == 0.0
    assert feats2["lbp_nonuniform_ratio_cb"] == 0.0
    assert feats2["lbp_entropy_cb"] == 0.0

    # ── Test 3: Audit helper ─────────────────────────────────────────
    audit = extract_microtexture_audit(flat)
    print("\nTest 3 — Audit on flat grey:")
    print(f"  srm_energy_zero_count: {audit['srm_energy_zero_count']}")
    print(f"  residual_shapes: {audit['residual_shapes']}")
    print(f"  lbp_cr_sentinel_used: {audit['lbp_cr_sentinel_used']}")
    print(f"  lbp_cb_sentinel_used: {audit['lbp_cb_sentinel_used']}")
    print(f"  lbp_core_shape: {audit['lbp_core_shape']}")
    assert audit["srm_energy_zero_count"] == 3
    assert audit["residual_shapes"]["square3"] == (254, 254)
    assert audit["residual_shapes"]["edge3"] == (254, 254)
    assert audit["residual_shapes"]["square5"] == (252, 252)
    assert audit["lbp_core_shape"] == (254, 254)

    # ── Test 4: Uniform saturated (non-zero Cr, const) → sentinel ────
    red = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    red[:, :, 0] = 80
    red[:, :, 1] = 240  # constant Cr
    red[:, :, 2] = 90   # constant Cb
    feats4 = extract_microtexture_features(red)
    print("\nTest 4 — Uniform saturated:")
    for k, v in feats4.items():
        print(f"  {k}: {v:.8e}")
    # Constant Cr → SRM residual = 0, LBP ptp = 0
    for name in ("square3", "edge3", "square5"):
        assert feats4[f"srm_{name}_mar_cr"] == 0.0
    assert feats4["lbp_nonuniform_ratio_cr"] == 0.0
    assert feats4["lbp_nonuniform_ratio_cb"] == 0.0

    print("\nAll smoke tests passed.")
