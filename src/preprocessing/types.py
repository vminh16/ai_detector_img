"""Typed results and statuses for preprocessing v4."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class PreprocessStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    LOW_SUPPORT = "LOW_SUPPORT"
    UNSUPPORTED_INPUT = "UNSUPPORTED_INPUT"
    DECODE_ERROR = "DECODE_ERROR"


@dataclass
class PreprocessResult:
    """Full result for one source file."""

    file_path: str
    output_path: str
    generator: str
    label: str
    status: PreprocessStatus
    preprocess_version: str
    error: str | None = None
    input_format: str | None = None
    input_mode: str | None = None
    normalized_mode: str | None = None
    width: int | None = None
    height: int | None = None
    support: int | None = None
    support_threshold: int | None = None
    orientation: int | None = None
    orientation_applied: bool = False
    alpha_composited: bool = False
    crop_size: int | None = None
    residue_x: int | None = None
    residue_y: int | None = None
    crop_origin_x: int | None = None
    crop_origin_y: int | None = None
    patch_shape: str | None = None
    patch_dtype: str | None = None
    saved_patch: bool = False
    stale_output_removed: bool = False
    patch: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def accepted(self) -> bool:
        return self.status == PreprocessStatus.ACCEPTED

    def manifest_row(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "output_path": self.output_path,
            "generator": self.generator,
            "label": self.label,
            "status": self.status.value,
            "error": self.error,
            "input_format": self.input_format,
            "input_mode": self.input_mode,
            "normalized_mode": self.normalized_mode,
            "width": self.width,
            "height": self.height,
            "support": self.support,
            "support_threshold": self.support_threshold,
            "orientation": self.orientation,
            "orientation_applied": self.orientation_applied,
            "alpha_composited": self.alpha_composited,
            "crop_size": self.crop_size,
            "residue_x": self.residue_x,
            "residue_y": self.residue_y,
            "crop_origin_x": self.crop_origin_x,
            "crop_origin_y": self.crop_origin_y,
            "patch_shape": self.patch_shape,
            "patch_dtype": self.patch_dtype,
            "saved_patch": self.saved_patch,
            "stale_output_removed": self.stale_output_removed,
            "preprocess_version": self.preprocess_version,
        }


MANIFEST_COLUMNS: list[str] = list(
    PreprocessResult(
        file_path="",
        output_path="",
        generator="",
        label="",
        status=PreprocessStatus.DECODE_ERROR,
        preprocess_version="",
    ).manifest_row().keys()
)
