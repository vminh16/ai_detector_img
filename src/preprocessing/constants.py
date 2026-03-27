"""Constants and configuration for preprocessing v4."""

from __future__ import annotations

from dataclasses import dataclass

DISCOVERY_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
)
COUNT_EXTENSIONS: frozenset[str] = DISCOVERY_EXTENSIONS | frozenset({".npy"})
SUPPORTED_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG"})
SUPPORTED_CHAMPION_MODES: frozenset[str] = frozenset({"RGB", "RGBA"})

CROP_SIZE: int = 248
RESIDUE_X: int = 4
RESIDUE_Y: int = 4
SUPPORT_THRESHOLD: int = 252
ALPHA_BACKGROUND_VALUE: int = 128
OUTPUT_EXT: str = ".npy"
PREPROCESS_VERSION: str = "v4_rgb248_r4_exact"


@dataclass(frozen=True)
class PreprocessConfig:
    """Immutable preprocessing contract for champion v4."""

    crop_size: int = CROP_SIZE
    residue_x: int = RESIDUE_X
    residue_y: int = RESIDUE_Y
    support_threshold: int = SUPPORT_THRESHOLD
    alpha_background_value: int = ALPHA_BACKGROUND_VALUE
    output_ext: str = OUTPUT_EXT
    preprocess_version: str = PREPROCESS_VERSION

    def __post_init__(self) -> None:
        if self.crop_size <= 0:
            raise ValueError("crop_size must be positive.")
        if not (0 <= self.residue_x <= 7 and 0 <= self.residue_y <= 7):
            raise ValueError("residue must be in [0, 7].")
        min_support = self.crop_size + max(self.residue_x, self.residue_y)
        if self.support_threshold < min_support:
            raise ValueError(
                "support_threshold must be >= crop_size + max(residue_x, residue_y)."
            )


DEFAULT_CONFIG = PreprocessConfig()
