"""Typed records for feature extraction v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class FeatureExtractionStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


@dataclass
class FeatureExtractionResult:
    row_id: int
    source_file_path: str
    patch_path: str
    generator: str
    label: str
    split_role: str
    dataset_name: str
    preprocess_version: str
    feature_version: str
    status: FeatureExtractionStatus
    error: str | None = None
    features: dict[str, float] = field(default_factory=dict)

    def manifest_row(self, feature_names: tuple[str, ...]) -> dict[str, Any]:
        row: dict[str, Any] = {
            "source_file_path": self.source_file_path,
            "patch_path": self.patch_path,
            "generator": self.generator,
            "label": self.label,
            "split_role": self.split_role,
            "dataset_name": self.dataset_name,
            "preprocess_version": self.preprocess_version,
            "feature_version": self.feature_version,
        }
        for name in feature_names:
            row[name] = self.features.get(name, np.nan)
        row["status"] = self.status.value
        row["error"] = self.error or ""
        return row


BASE_COLUMNS = [
    "source_file_path",
    "patch_path",
    "generator",
    "label",
    "split_role",
    "dataset_name",
    "preprocess_version",
    "feature_version",
]
