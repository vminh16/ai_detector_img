"""Research-only dark textured heteroskedasticity features."""

from __future__ import annotations

import numpy as np

from .constants import EPS, HETERO_KEYS, HETERO_RESIDUAL_WINDOW, LOCAL_WINDOW, MASK_MIN_PIXELS, SQUARE3_KERNEL
from .ops import box_mean_valid, box_var_valid, center_crop_to_shape, convolve_valid, linear_fit_with_r2, monotone_violation_rate, sobel_magnitude_valid
from .views import FeatureContext


FEATURE_KEYS = HETERO_KEYS


def extract_dark_hetero_features(ctx: FeatureContext) -> dict[str, float]:
    mean_map = box_mean_valid(ctx.y, LOCAL_WINDOW).astype(np.float32)
    gradient = center_crop_to_shape(sobel_magnitude_valid(ctx.y), mean_map.shape)
    residual = convolve_valid(ctx.y, SQUARE3_KERNEL).astype(np.float32)
    var_map = box_var_valid(residual, HETERO_RESIDUAL_WINDOW).astype(np.float32)

    dark_threshold = float(np.percentile(mean_map, 35.0))
    grad_low = float(np.percentile(gradient, 30.0))
    grad_high = float(np.percentile(gradient, 70.0))
    var_floor = float(np.percentile(var_map, 50.0))

    base_mask = (
        (mean_map <= dark_threshold)
        & (gradient >= grad_low)
        & (gradient <= grad_high)
        & (var_map >= var_floor)
    )

    if int(np.count_nonzero(base_mask)) < MASK_MIN_PIXELS:
        return {key: 0.0 for key in FEATURE_KEYS}

    x = mean_map[base_mask]
    y = var_map[base_mask]
    slope, r2 = linear_fit_with_r2(x, y)
    cv = float(np.std(y, dtype=np.float64) / (np.mean(y, dtype=np.float64) + EPS))

    dark_only = mean_map <= dark_threshold
    dark_grad = gradient[dark_only]
    edge_cut = float(np.percentile(dark_grad, 75.0))
    flat_cut = float(np.percentile(dark_grad, 25.0))
    dark_edge_mask = dark_only & (gradient >= edge_cut)
    dark_flat_mask = dark_only & (gradient <= flat_cut)

    edge_var = float(np.mean(var_map[dark_edge_mask], dtype=np.float64)) if np.any(dark_edge_mask) else 0.0
    flat_var = float(np.mean(var_map[dark_flat_mask], dtype=np.float64)) if np.any(dark_flat_mask) else 0.0
    ratio = float(np.log10((edge_var + EPS) / (flat_var + EPS)))
    violation = monotone_violation_rate(x, y)

    return {
        "lochet_dark_flat_slope": slope,
        "lochet_dark_flat_r2": r2,
        "lochet_dark_flat_cv": cv,
        "lochet_dark_edge_flat_logratio": ratio,
        "lochet_dark_monotone_violation": violation,
    }
