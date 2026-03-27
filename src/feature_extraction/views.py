"""Input validation and derived views for feature extraction v2."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .constants import PATCH_SIZE, RGB_CHANNELS


def _validate_patch(patch: np.ndarray) -> np.ndarray:
    if not isinstance(patch, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(patch).__name__}")
    if patch.shape != (PATCH_SIZE, PATCH_SIZE, RGB_CHANNELS):
        raise ValueError(f"Expected shape {(PATCH_SIZE, PATCH_SIZE, RGB_CHANNELS)}, got {patch.shape}")
    if patch.dtype != np.uint8:
        raise TypeError(f"Expected uint8 patch, got {patch.dtype}")
    if not np.isfinite(patch).all():
        raise ValueError("Patch contains non-finite values.")
    patch_u8 = np.asarray(patch, dtype=np.uint8)
    patch_u8.setflags(write=False)
    return patch_u8


def rgb_to_ycrcb_float32(rgb: np.ndarray) -> np.ndarray:
    rgb32 = np.asarray(rgb, dtype=np.float32)
    r = rgb32[:, :, 0]
    g = rgb32[:, :, 1]
    b = rgb32[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128.0
    cb = (b - y) * 0.564 + 128.0
    return np.stack([y, cr, cb], axis=-1, dtype=np.float32)


@dataclass
class FeatureContext:
    """Read-only canonical patch plus lazily derived views."""

    patch: np.ndarray

    def __post_init__(self) -> None:
        self.patch = _validate_patch(self.patch)

    @cached_property
    def rgb_f32(self) -> np.ndarray:
        return self.patch.astype(np.float32, copy=True)

    @cached_property
    def ycrcb_f32(self) -> np.ndarray:
        return rgb_to_ycrcb_float32(self.rgb_f32)

    @cached_property
    def y(self) -> np.ndarray:
        return self.ycrcb_f32[:, :, 0]

    @cached_property
    def cr(self) -> np.ndarray:
        return self.ycrcb_f32[:, :, 1]

    @cached_property
    def cb(self) -> np.ndarray:
        return self.ycrcb_f32[:, :, 2]

    @cached_property
    def y_centered(self) -> np.ndarray:
        return self.y - float(np.mean(self.y))

    @cached_property
    def rg_diff(self) -> np.ndarray:
        rgb = self.rgb_f32
        return rgb[:, :, 0] - rgb[:, :, 1]

    @cached_property
    def bg_diff(self) -> np.ndarray:
        rgb = self.rgb_f32
        return rgb[:, :, 2] - rgb[:, :, 1]
