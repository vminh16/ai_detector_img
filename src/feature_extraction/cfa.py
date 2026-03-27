"""Conditional RGB-domain CFA features."""

from __future__ import annotations

import numpy as np

from .constants import CONDITIONAL_CFA_KEYS, EDGE3_KERNEL, EPS, MID_BAND_MAX, MID_BAND_MIN, PATCH_SIZE, VALIDITY_HIGH_MAX, VALIDITY_HIGH_MIN, VALIDITY_KEYS
from .ops import convolve_valid
from .views import FeatureContext


FEATURE_KEYS = CONDITIONAL_CFA_KEYS + VALIDITY_KEYS

_CHECKERBOARD_246 = np.fromfunction(
    lambda y, x: ((x + y) % 2) * 2.0 - 1.0,
    (PATCH_SIZE - 2, PATCH_SIZE - 2),
    dtype=np.float32,
).astype(np.float32)
_FREQ = np.fft.fftshift(np.fft.fftfreq(PATCH_SIZE, d=1.0)).astype(np.float32)
_FY, _FX = np.meshgrid(_FREQ, _FREQ, indexing="ij")
_RADIUS = np.sqrt(_FX ** 2 + _FY ** 2, dtype=np.float32)
_MID_MASK = (_RADIUS >= MID_BAND_MIN) & (_RADIUS <= MID_BAND_MAX)
_HIGH_MASK = (_RADIUS >= VALIDITY_HIGH_MIN) & (_RADIUS <= VALIDITY_HIGH_MAX)


def _normalized_checkerboard_projection(diff_map: np.ndarray) -> float:
    highpass = convolve_valid(diff_map, EDGE3_KERNEL).astype(np.float32)
    numerator = float(abs(np.mean(highpass * _CHECKERBOARD_246, dtype=np.float64)))
    denominator = float(np.sqrt(np.mean(highpass ** 2, dtype=np.float64)) + EPS)
    return numerator / denominator


def _high_frequency_survival(diff_map: np.ndarray) -> float:
    centered = diff_map - float(np.mean(diff_map))
    fft = np.fft.fft2(centered.astype(np.float32)) / float(PATCH_SIZE * PATCH_SIZE)
    power = np.abs(np.fft.fftshift(fft)) ** 2
    high = float(np.mean(power[_HIGH_MASK], dtype=np.float64))
    mid = float(np.mean(power[_MID_MASK], dtype=np.float64))
    return float(np.log10((high + EPS) / (mid + EPS)))


def extract_conditional_cfa_features(ctx: FeatureContext) -> dict[str, float]:
    rg_score = _normalized_checkerboard_projection(ctx.rg_diff)
    bg_score = _normalized_checkerboard_projection(ctx.bg_diff)
    validity = 0.5 * (_high_frequency_survival(ctx.rg_diff) + _high_frequency_survival(ctx.bg_diff))
    return {
        "cfa_rg_pi_xy": rg_score,
        "cfa_bg_pi_xy": bg_score,
        "cfa_rgb_pi_xy_mean": 0.5 * (rg_score + bg_score),
        "cfa_rgb_pi_xy_gap": abs(rg_score - bg_score),
        "cfa_validity_score": validity,
    }
