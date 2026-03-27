"""Always-on Y-domain spatial control features."""

from __future__ import annotations

import numpy as np

from .constants import CONTROL_SPATIAL_KEYS, EPS, SQUARE3_KERNEL
from .ops import convolve_valid, safe_skew_kurt, sobel_magnitude_valid
from .views import FeatureContext


FEATURE_KEYS = CONTROL_SPATIAL_KEYS


def extract_spatial_features(ctx: FeatureContext) -> dict[str, float]:
    gradient = sobel_magnitude_valid(ctx.y)
    residual = convolve_valid(ctx.y, SQUARE3_KERNEL)

    edge_threshold = max(float(np.percentile(gradient, 90.0)), 15.0)
    flat_threshold = min(float(np.percentile(gradient, 30.0)), 3.0)
    edge_mask = gradient >= edge_threshold
    flat_mask = gradient <= flat_threshold

    abs_residual = np.abs(residual, dtype=np.float32)
    edge_mean = float(np.mean(abs_residual[edge_mask])) if np.any(edge_mask) else 0.0
    flat_values = residual[flat_mask] if np.any(flat_mask) else np.asarray([], dtype=np.float32)
    flat_mean = float(np.mean(np.abs(flat_values))) if flat_values.size else 0.0
    spatial_snr_ratio = float(np.log10((edge_mean + EPS) / (flat_mean + EPS)))
    skew_noise_y, kurt_noise_y = safe_skew_kurt(flat_values)

    return {
        "spatial_snr_ratio": spatial_snr_ratio,
        "skew_noise_y": skew_noise_y,
        "kurt_noise_y": kurt_noise_y,
    }
