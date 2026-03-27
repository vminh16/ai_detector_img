"""
inference/config.py
===================
Centralised configuration for the inference pipeline.

All tunable parameters are loaded from artifact files and environment
variables — nothing is hardcoded in business logic.  Paths default to
the project's ``models/param/`` directory.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """Locate the project root (parent of the ``inference/`` package)."""
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class InferenceConfig:
    """Immutable snapshot of all runtime configuration."""

    # ── Paths ────────────────────────────────────────────────────
    artifacts_dir: Path = field(default_factory=lambda: _project_root() / "models" / "param")

    # ── File-size limit (bytes) ──────────────────────────────────
    max_upload_bytes: int = 20 * 1024 * 1024  # 20 MB

    # ── Preprocessing ────────────────────────────────────────────
    crop_size: int = 256
    pad_min_size: int = 259
    grid_misalign_offset: int = 3
    jpeg_q_min: int = 90
    jpeg_q_max: int = 98

    # ── Accepted MIME / magic signatures ─────────────────────────
    # (header bytes, format label)
    accepted_signatures: tuple[tuple[bytes, str], ...] = (
        (b"\xff\xd8\xff", "jpeg"),
        (b"\x89PNG\r\n\x1a\n", "png"),
        (b"RIFF", "webp"),   # first 4 bytes; bytes 8-12 = "WEBP"
    )

    # ── Versioning ───────────────────────────────────────────────
    preprocess_version: str = "1.0.0"
    feature_version: str = "1.0.0"

    # ── Loaded lazily from JSON artifacts at startup ─────────────
    _threshold_info: dict = field(default_factory=dict, repr=False)
    _winsor_caps: dict = field(default_factory=dict, repr=False)
    _feature_schema: dict = field(default_factory=dict, repr=False)

    # ── Derived properties ──────────────────────────────────────
    @property
    def champion_model_path(self) -> Path:
        pipeline = self._threshold_info.get("champion_pipeline", "gbdt")
        name_map = {"gbdt": "champion_lgbm.joblib", "rf": "champion_rf.joblib", "svm": "champion_svm.joblib"}
        return self.artifacts_dir / name_map.get(pipeline, "champion_lgbm.joblib")

    @property
    def calibrator_path(self) -> Path:
        return self.artifacts_dir / "platt_calibrator.joblib"

    @property
    def tau_op(self) -> float:
        return float(self._threshold_info.get("tau_op", 0.5))

    @property
    def low_threshold(self) -> float:
        return 0.3

    @property
    def champion_pipeline(self) -> str:
        return self._threshold_info.get("champion_pipeline", "gbdt")

    @property
    def model_version(self) -> str:
        return self._threshold_info.get("champion_model", "unknown")

    @property
    def feature_order(self) -> list[str]:
        return self._feature_schema.get("active_features", [])

    @property
    def winsor_caps(self) -> dict[str, list[float]]:
        return self._winsor_caps


def load_config(artifacts_dir: Path | str | None = None) -> InferenceConfig:
    """Build an ``InferenceConfig`` from on-disk artifacts.

    Parameters
    ----------
    artifacts_dir : optional override for the default artifacts path.
        Can also be set via ``AI_DETECTOR_ARTIFACTS_DIR`` env var.
    """
    if artifacts_dir is not None:
        art = Path(artifacts_dir)
    elif os.environ.get("AI_DETECTOR_ARTIFACTS_DIR"):
        art = Path(os.environ["AI_DETECTOR_ARTIFACTS_DIR"])
    else:
        art = _project_root() / "models" / "param"

    threshold_path = art / "threshold_lock.json"
    winsor_path = art / "winsor_caps.json"
    schema_path = art / "feature_schema.json"

    for p in (threshold_path, winsor_path, schema_path):
        if not p.exists():
            raise FileNotFoundError(f"Required artifact missing: {p}")

    with open(threshold_path, "r") as f:
        threshold_info = json.load(f)
    with open(winsor_path, "r") as f:
        winsor_caps = json.load(f)
    with open(schema_path, "r") as f:
        feature_schema = json.load(f)

    return InferenceConfig(
        artifacts_dir=art,
        _threshold_info=threshold_info,
        _winsor_caps=winsor_caps,
        _feature_schema=feature_schema,
    )
