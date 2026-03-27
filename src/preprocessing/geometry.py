"""Exact residue crop geometry for preprocessing v4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import LowSupportError


@dataclass(frozen=True)
class CropOrigin:
    x: int
    y: int


def min_length_for_exact_residue_crop(crop_size: int, residue: int) -> int:
    if crop_size <= 0:
        raise ValueError("crop_size must be positive.")
    if not (0 <= residue <= 7):
        raise ValueError("residue must be in [0, 7].")
    return crop_size + residue


def admissible_starts(length: int, crop_size: int, residue: int) -> tuple[int, ...]:
    if length < min_length_for_exact_residue_crop(crop_size, residue):
        return ()
    upper = length - crop_size
    k_max = (upper - residue) // 8
    return tuple(residue + 8 * k for k in range(k_max + 1))


def nearest_residue_start(length: int, crop_size: int, residue: int) -> int | None:
    starts = admissible_starts(length, crop_size, residue)
    if not starts:
        return None
    ideal = (length - crop_size) / 2.0
    return min(starts, key=lambda start: (abs(start - ideal), start))


def exact_crop_origin(
    height: int,
    width: int,
    crop_size: int,
    residue_x: int,
    residue_y: int,
) -> CropOrigin:
    x0 = nearest_residue_start(width, crop_size, residue_x)
    y0 = nearest_residue_start(height, crop_size, residue_y)
    if x0 is None or y0 is None:
        raise LowSupportError(
            f"LOW_SUPPORT: {(height, width)} cannot support crop={crop_size} residue=({residue_x},{residue_y})"
        )
    return CropOrigin(x=x0, y=y0)


def crop_exact_residue(
    rgb8: np.ndarray,
    crop_size: int,
    residue_x: int,
    residue_y: int,
) -> tuple[np.ndarray, CropOrigin]:
    if rgb8.ndim != 3 or rgb8.shape[2] != 3:
        raise ValueError(f"expected RGB array, got shape={rgb8.shape}")
    if rgb8.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB array, got dtype={rgb8.dtype}")
    height, width = rgb8.shape[:2]
    origin = exact_crop_origin(height, width, crop_size, residue_x, residue_y)
    patch = rgb8[origin.y : origin.y + crop_size, origin.x : origin.x + crop_size, :]
    expected_shape = (crop_size, crop_size, 3)
    if patch.shape != expected_shape:
        raise RuntimeError(f"unexpected patch shape {patch.shape}, expected {expected_shape}")
    return np.ascontiguousarray(patch), origin


def center_linf_distance(
    height: int,
    width: int,
    crop_size: int,
    residue_x: int,
    residue_y: int,
) -> float:
    """Return L_inf drift from the ideal geometric center."""

    origin = exact_crop_origin(height, width, crop_size, residue_x, residue_y)
    ideal_x = (width - crop_size) / 2.0
    ideal_y = (height - crop_size) / 2.0
    return float(max(abs(origin.x - ideal_x), abs(origin.y - ideal_y)))


def phase_distance(residue: int, block_size: int = 8) -> int:
    if not (0 <= residue < block_size):
        raise ValueError("residue must be in [0, block_size).")
    return min(residue, block_size - residue)
