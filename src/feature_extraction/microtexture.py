"""Research-only content-adaptive Y-SRM features."""

from __future__ import annotations

import numpy as np

from .constants import EDGE3_KERNEL, LOCAL_WINDOW, MASK_MIN_PIXELS, SQUARE3_KERNEL, SQUARE5_KERNEL, YSRM_KEYS
from .ops import box_var_valid, center_crop_to_shape, convolve_valid, sobel_magnitude_valid, trimmed_mean
from .views import FeatureContext


FEATURE_KEYS = YSRM_KEYS


def _medium_texture_mask(ctx: FeatureContext) -> np.ndarray:
    gradient = sobel_magnitude_valid(ctx.y)
    gradient240 = center_crop_to_shape(gradient, (240, 240))
    std240 = np.sqrt(box_var_valid(ctx.y, LOCAL_WINDOW), dtype=np.float32)

    grad_lo = float(np.percentile(gradient240, 35.0))
    grad_hi = float(np.percentile(gradient240, 80.0))
    std_lo = float(np.percentile(std240, 40.0))
    std_hi = float(np.percentile(std240, 85.0))

    return (
        (gradient240 >= grad_lo)
        & (gradient240 <= grad_hi)
        & (std240 >= std_lo)
        & (std240 <= std_hi)
    )


def _masked_energy_and_mar(residual: np.ndarray, mask240: np.ndarray) -> tuple[float, float]:
    cropped = center_crop_to_shape(residual, mask240.shape)
    values = cropped[mask240]
    if values.size < MASK_MIN_PIXELS:
        return 0.0, 0.0
    mar = trimmed_mean(np.abs(values))
    energy = trimmed_mean(values ** 2)
    return energy, mar


def extract_microtexture_features(ctx: FeatureContext) -> dict[str, float]:
    mask240 = _medium_texture_mask(ctx)

    edge3 = convolve_valid(ctx.y, EDGE3_KERNEL).astype(np.float32)
    square3 = convolve_valid(ctx.y, SQUARE3_KERNEL).astype(np.float32)
    square5 = convolve_valid(ctx.y, SQUARE5_KERNEL).astype(np.float32)

    edge3_energy, edge3_mar = _masked_energy_and_mar(edge3, mask240)
    square3_energy, square3_mar = _masked_energy_and_mar(square3, mask240)
    square5_energy, square5_mar = _masked_energy_and_mar(square5, mask240)

    return {
        "ysrm_midtex_edge3_energy": edge3_energy,
        "ysrm_midtex_edge3_mar": edge3_mar,
        "ysrm_midtex_square3_energy": square3_energy,
        "ysrm_midtex_square3_mar": square3_mar,
        "ysrm_midtex_square5_energy": square5_energy,
        "ysrm_midtex_square5_mar": square5_mar,
    }
