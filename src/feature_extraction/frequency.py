"""Always-on spectral features for v2."""

from __future__ import annotations

import numpy as np

from .constants import (
    CONTROL_FREQUENCY_KEYS,
    EPS,
    FFT_MIDBAND_KEYS,
    FFT_RADIAL_BINS,
    FFT_TOTAL_MAX,
    FFT_TOTAL_MIN,
    MID_BAND_MAX,
    MID_BAND_MIN,
    MID_BAND_SPLIT,
    PATCH_SIZE,
)
from .ops import tukey_window_2d
from .views import FeatureContext


FEATURE_KEYS = CONTROL_FREQUENCY_KEYS + FFT_MIDBAND_KEYS

_FREQ = np.fft.fftshift(np.fft.fftfreq(PATCH_SIZE, d=1.0)).astype(np.float32)
_FY, _FX = np.meshgrid(_FREQ, _FREQ, indexing="ij")
_RADIUS = np.sqrt(_FX ** 2 + _FY ** 2, dtype=np.float32)
_THETA = np.arctan2(_FY, _FX).astype(np.float32)
_MID_MASK = (_RADIUS >= MID_BAND_MIN) & (_RADIUS <= MID_BAND_MAX)
_INNER_MASK = (_RADIUS >= MID_BAND_MIN) & (_RADIUS < MID_BAND_SPLIT)
_OUTER_MASK = (_RADIUS >= MID_BAND_SPLIT) & (_RADIUS <= MID_BAND_MAX)
_TOTAL_MASK = (_RADIUS >= FFT_TOTAL_MIN) & (_RADIUS <= FFT_TOTAL_MAX)


def _angular_distance(theta: np.ndarray, center: float) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * (theta - center))), dtype=np.float32)


def _sector_mask(center: float, half_width_deg: float = 15.0) -> np.ndarray:
    half_width = np.deg2rad(half_width_deg)
    return _MID_MASK & (_angular_distance(_THETA, center) <= half_width)


_HORIZONTAL_MASK = _sector_mask(0.0) | _sector_mask(np.pi)
_VERTICAL_MASK = _sector_mask(np.pi / 2.0) | _sector_mask(-np.pi / 2.0)
_DIAG_POS_MASK = _sector_mask(np.pi / 4.0) | _sector_mask(-3.0 * np.pi / 4.0)
_DIAG_NEG_MASK = _sector_mask(3.0 * np.pi / 4.0) | _sector_mask(-np.pi / 4.0)
_RADIAL_EDGES = np.linspace(MID_BAND_MIN, MID_BAND_MAX, FFT_RADIAL_BINS + 1, dtype=np.float32)


def _power_spectrum(y: np.ndarray) -> np.ndarray:
    centered = (y - float(np.mean(y))).astype(np.float32)
    tapered = centered * tukey_window_2d()
    spectrum = np.fft.fft2(tapered) / float(PATCH_SIZE * PATCH_SIZE)
    return np.abs(np.fft.fftshift(spectrum)) ** 2


def _mean_or_zero(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values, dtype=np.float64))


def _radial_profile(power: np.ndarray) -> np.ndarray:
    means: list[float] = []
    for idx in range(FFT_RADIAL_BINS):
        left = _RADIAL_EDGES[idx]
        right = _RADIAL_EDGES[idx + 1]
        if idx == FFT_RADIAL_BINS - 1:
            mask = (_RADIUS >= left) & (_RADIUS <= right)
        else:
            mask = (_RADIUS >= left) & (_RADIUS < right)
        means.append(_mean_or_zero(power[mask]))
    return np.asarray(means, dtype=np.float64)


def extract_frequency_features(ctx: FeatureContext) -> dict[str, float]:
    power = _power_spectrum(ctx.y)

    radial = _radial_profile(power)
    radial_mean = float(np.mean(radial))
    frs_mid_variance = 0.0 if radial_mean < EPS else float(np.var(radial) / (radial_mean ** 2 + EPS))
    fft_mid_ring_var = float(np.var(np.log(radial + EPS)))

    total_energy = _mean_or_zero(power[_TOTAL_MASK])
    mid_energy = _mean_or_zero(power[_MID_MASK])
    inner_energy = _mean_or_zero(power[_INNER_MASK])
    outer_energy = _mean_or_zero(power[_OUTER_MASK])

    mid_values = power[_MID_MASK]
    flatness = 0.0
    if mid_values.size:
        flatness = float(np.exp(np.mean(np.log(mid_values + EPS))) / (np.mean(mid_values) + EPS))

    horizontal_energy = _mean_or_zero(power[_HORIZONTAL_MASK])
    vertical_energy = _mean_or_zero(power[_VERTICAL_MASK])
    diag_pos_energy = _mean_or_zero(power[_DIAG_POS_MASK])
    diag_neg_energy = _mean_or_zero(power[_DIAG_NEG_MASK])

    return {
        "frs_mid_variance": frs_mid_variance,
        "fft_mid_logenergy": float(np.log10((mid_energy + EPS) / (total_energy + EPS))),
        "fft_mid_flatness": flatness,
        "fft_mid_ring_var": fft_mid_ring_var,
        "fft_mid_inner_outer_ratio": float(np.log10((inner_energy + EPS) / (outer_energy + EPS))),
        "fft_mid_anisotropy_hv": float(abs(np.log((horizontal_energy + EPS) / (vertical_energy + EPS)))),
        "fft_mid_anisotropy_diag": float(abs(np.log((diag_pos_energy + EPS) / (diag_neg_energy + EPS)))),
    }
