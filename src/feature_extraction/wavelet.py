"""Research-only wavelet decay features."""

from __future__ import annotations

import numpy as np

from .constants import EPS, WAVELET_KEYS
from .views import FeatureContext


FEATURE_KEYS = WAVELET_KEYS


def _haar_level(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low_rows = 0.5 * (image[0::2, :] + image[1::2, :])
    high_rows = 0.5 * (image[0::2, :] - image[1::2, :])
    ll = 0.5 * (low_rows[:, 0::2] + low_rows[:, 1::2])
    lh = 0.5 * (low_rows[:, 0::2] - low_rows[:, 1::2])
    hl = 0.5 * (high_rows[:, 0::2] + high_rows[:, 1::2])
    hh = 0.5 * (high_rows[:, 0::2] - high_rows[:, 1::2])
    detail = np.stack([lh, hl, hh], axis=0)
    return ll.astype(np.float32), detail.astype(np.float32)


def extract_wavelet_features(ctx: FeatureContext) -> dict[str, float]:
    base_var = float(np.var(ctx.y, dtype=np.float64)) + EPS
    ll1, detail1 = _haar_level(ctx.y_centered.astype(np.float32))
    ll2, detail2 = _haar_level(ll1)
    ll3, detail3 = _haar_level(ll2)

    e1 = float(np.mean(detail1 ** 2, dtype=np.float64) / base_var)
    e2 = float(np.mean(detail2 ** 2, dtype=np.float64) / base_var)
    e3 = float(np.mean(detail3 ** 2, dtype=np.float64) / base_var)

    levels = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    energies = np.asarray([e1, e2, e3], dtype=np.float64)
    slope, _ = np.polyfit(levels, np.log(energies + EPS), 1)

    return {
        "wav_energy_l1": e1,
        "wav_energy_l2": e2,
        "wav_energy_l3": e3,
        "wav_decay_alpha": float(-slope),
        "wav_ratio_l1_l2": float(np.log10((e1 + EPS) / (e2 + EPS))),
        "wav_ratio_l2_l3": float(np.log10((e2 + EPS) / (e3 + EPS))),
    }
