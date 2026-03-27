"""Canonical decode and mode normalization for preprocessing v4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import DecodeImageError, UnsupportedInputError


@dataclass(frozen=True)
class DecodedImage:
    image: Image.Image
    format_name: str
    mode: str
    width: int
    height: int


@dataclass(frozen=True)
class NormalizedImage:
    rgb8: np.ndarray
    input_mode: str
    normalized_mode: str
    alpha_composited: bool


def decode_image(filepath: Path | str) -> DecodedImage:
    """Decode an image with the canonical Pillow-based decoder."""

    path = Path(filepath)
    try:
        with Image.open(path) as image:
            image.load()
            format_name = (image.format or "").upper()
            decoded = image.copy()
            mode = decoded.mode
            width, height = decoded.size
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise DecodeImageError(f"decode failed: {type(exc).__name__}: {exc}") from exc

    return DecodedImage(
        image=decoded,
        format_name=format_name,
        mode=mode,
        width=width,
        height=height,
    )


def composite_straight_alpha(
    rgba: np.ndarray,
    *,
    background_value: int = 128,
) -> np.ndarray:
    """Composite straight-alpha RGBA onto a neutral gray background."""

    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"expected RGBA array, got shape={rgba.shape}")
    rgba_u16 = rgba.astype(np.uint16, copy=False)
    rgb = rgba_u16[..., :3]
    alpha = rgba_u16[..., 3:4]
    background = np.uint16(background_value)
    out = (alpha * rgb + (255 - alpha) * background + 127) // 255
    return out.astype(np.uint8)


def normalize_mode_to_rgb(
    image: Image.Image,
    *,
    background_value: int = 128,
) -> NormalizedImage:
    """Normalize champion-supported modes to RGB uint8."""

    if image.mode == "RGB":
        rgb8 = np.array(image, dtype=np.uint8, copy=True)
        return NormalizedImage(
            rgb8=rgb8,
            input_mode="RGB",
            normalized_mode="RGB",
            alpha_composited=False,
        )

    if image.mode == "RGBA":
        rgba = np.array(image, dtype=np.uint8, copy=True)
        rgb8 = composite_straight_alpha(rgba, background_value=background_value)
        return NormalizedImage(
            rgb8=rgb8,
            input_mode="RGBA",
            normalized_mode="RGB",
            alpha_composited=True,
        )

    raise UnsupportedInputError(
        f"image mode {image.mode or 'UNKNOWN'} is unsupported for champion v4"
    )
