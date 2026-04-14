"""Active runtime inference pipeline for the web app and API.

This module loads the selected benchmark artifacts produced by
``training_v2_phase_closure_20260403`` and serves synchronous prediction
from raw image bytes or files.

Runtime contract
----------------
- preprocessing: ``v4_rgb248_r4_exact``
- feature extraction: ``v2_rgb248_exact_multibranch``
- model artifact: selected candidate from ``models/param/<run>/``
- accepted upload formats: JPEG, PNG
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError

from inference.errors import (
    FeatureExtractionError,
    ImageTooLargeError,
    InvalidImageError,
    ModelLoadError,
    PreprocessingError,
    UnsupportedFormatError,
)
from src.feature_extraction import DEFAULT_CONFIG as DEFAULT_FEATURE_CONFIG
from src.feature_extraction import extract_feature_vector
from src.feature_extraction.constants import FEATURE_VERSION as ACTIVE_FEATURE_VERSION
from src.preprocessing.constants import (
    ALPHA_BACKGROUND_VALUE,
    DEFAULT_CONFIG as DEFAULT_PREPROCESS_CONFIG,
    PREPROCESS_VERSION as ACTIVE_PREPROCESS_VERSION,
    SUPPORTED_FORMATS,
)
from src.preprocessing.decode import normalize_mode_to_rgb
from src.preprocessing.errors import LowSupportError, UnsupportedInputError
from src.preprocessing.geometry import crop_exact_residue
from src.preprocessing.orientation import ORIENTATION_TAG, apply_orientation_pil
from src.training.metrics import safe_logit

logger = logging.getLogger("deploy.pipeline")

REQUIRED_ARTIFACT_FILES = (
    "model_manifest.json",
    "selected_feature_schema.json",
    "selected_model.joblib",
    "selected_platt_calibrator.joblib",
    "selected_threshold.json",
)
DEFAULT_ARTIFACTS_ROOT = Path("models") / "param"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
LOW_THRESHOLD = 0.30
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class RuntimeArtifacts:
    artifact_dir: Path
    candidate_name: str
    feature_set: str
    model_name: str
    feature_columns: tuple[str, ...]
    threshold: float
    target_fpr: float
    cfa_threshold: float
    preprocess_version: str
    feature_version: str
    model: Any
    platt: Any


@dataclass(frozen=True)
class PreprocessMetadata:
    format_detected: str
    input_mode: str
    normalized_mode: str
    alpha_composited: bool
    orientation: int | None
    crop_origin_x: int
    crop_origin_y: int


@dataclass
class PredictionResult:
    calibrated_score: float
    raw_score: float
    zone: str
    decision: str
    top_contributors: list[dict[str, float | str]]
    image_hash: str
    format_detected: str
    preprocess_version: str
    feature_version: str
    model_version: str
    candidate_name: str
    artifact_version: str
    low_threshold: float
    operating_threshold: float
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibrated_score": round(self.calibrated_score, 6),
            "raw_score": round(self.raw_score, 6),
            "zone": self.zone,
            "decision": self.decision,
            "top_contributors": self.top_contributors,
            "image_hash": self.image_hash,
            "format_detected": self.format_detected,
            "preprocess_version": self.preprocess_version,
            "feature_version": self.feature_version,
            "model_version": self.model_version,
            "candidate_name": self.candidate_name,
            "artifact_version": self.artifact_version,
            "low_threshold": round(self.low_threshold, 6),
            "operating_threshold": round(self.operating_threshold, 6),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _compute_hash(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _setup_logging(log_level: int) -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _required_files_present(path: Path) -> bool:
    return all((path / name).is_file() for name in REQUIRED_ARTIFACT_FILES)


def _resolve_artifact_dir(artifacts_dir: Path | str | None) -> Path:
    root = Path(artifacts_dir) if artifacts_dir is not None else DEFAULT_ARTIFACTS_ROOT
    root = root.resolve()
    if _required_files_present(root):
        return root
    if not root.is_dir():
        raise ModelLoadError(f"Artifact directory not found: {root}")

    candidates: list[tuple[datetime, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or not _required_files_present(child):
            continue
        manifest_path = child / "model_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            saved_at = datetime.fromisoformat(str(manifest["saved_at_utc"]))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ModelLoadError(f"Invalid model manifest at {manifest_path}: {exc}") from exc
        candidates.append((saved_at, child))

    if not candidates:
        raise ModelLoadError(
            f"No active artifact directory found under {root}. Expected one of: {REQUIRED_ARTIFACT_FILES}"
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _load_runtime_artifacts(artifacts_dir: Path | str | None) -> RuntimeArtifacts:
    artifact_dir = _resolve_artifact_dir(artifacts_dir)
    try:
        manifest = json.loads((artifact_dir / "model_manifest.json").read_text(encoding="utf-8"))
        schema = json.loads((artifact_dir / "selected_feature_schema.json").read_text(encoding="utf-8"))
        threshold_payload = json.loads((artifact_dir / "selected_threshold.json").read_text(encoding="utf-8"))
        model = joblib.load(artifact_dir / "selected_model.joblib")
        platt = joblib.load(artifact_dir / "selected_platt_calibrator.joblib")
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
        raise ModelLoadError(f"Failed to load runtime artifacts from {artifact_dir}: {exc}") from exc

    feature_columns = tuple(str(name) for name in schema.get("feature_columns", []))
    if not feature_columns:
        raise ModelLoadError(f"selected_feature_schema.json under {artifact_dir} has no feature_columns")

    preprocess_version = str(manifest.get("preprocess_version", ""))
    feature_version = str(manifest.get("feature_version", ""))
    manifest_threshold = float(manifest["threshold"])
    threshold_threshold = float(threshold_payload["threshold"])
    if not np.isclose(manifest_threshold, threshold_threshold, atol=1e-12):
        raise ModelLoadError(
            f"Threshold mismatch between model_manifest.json ({manifest_threshold}) "
            f"and selected_threshold.json ({threshold_threshold}) under {artifact_dir}"
        )
    if preprocess_version != ACTIVE_PREPROCESS_VERSION:
        raise ModelLoadError(
            f"Artifact preprocess_version={preprocess_version!r} does not match active {ACTIVE_PREPROCESS_VERSION!r}"
        )
    if feature_version != ACTIVE_FEATURE_VERSION:
        raise ModelLoadError(
            f"Artifact feature_version={feature_version!r} does not match active {ACTIVE_FEATURE_VERSION!r}"
        )

    return RuntimeArtifacts(
        artifact_dir=artifact_dir,
        candidate_name=str(manifest["candidate_name"]),
        feature_set=str(manifest["feature_set"]),
        model_name=str(manifest["model_name"]),
        feature_columns=feature_columns,
        threshold=manifest_threshold,
        target_fpr=float(manifest["target_fpr"]),
        cfa_threshold=float(manifest.get("cfa_threshold", float("nan"))),
        preprocess_version=preprocess_version,
        feature_version=feature_version,
        model=model,
        platt=platt,
    )


def _detect_format(payload: bytes) -> str:
    if payload.startswith(JPEG_MAGIC):
        return "JPEG"
    if payload.startswith(PNG_MAGIC):
        return "PNG"
    raise UnsupportedFormatError("Unsupported image format. Accepted: JPEG, PNG.")


def _read_orientation(image: Image.Image) -> int | None:
    try:
        exif = image.getexif()
    except (AttributeError, OSError, ValueError):
        return None
    if exif is None:
        return None
    value = exif.get(ORIENTATION_TAG)
    if isinstance(value, int) and 1 <= value <= 8:
        return value
    return None


def _decode_canonical_patch(payload: bytes) -> tuple[np.ndarray, PreprocessMetadata]:
    format_detected = _detect_format(payload)
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            orientation = _read_orientation(image)
            decoded = image.copy()
            decoded_format = (image.format or format_detected).upper()
            input_mode = decoded.mode
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"decode failed: {type(exc).__name__}: {exc}") from exc

    if decoded_format not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Decoded format {decoded_format or 'UNKNOWN'} is unsupported. Accepted: JPEG, PNG."
        )

    try:
        oriented = apply_orientation_pil(decoded, orientation)
        normalized = normalize_mode_to_rgb(
            oriented,
            background_value=ALPHA_BACKGROUND_VALUE,
        )
        support = min(normalized.rgb8.shape[:2])
        if support < DEFAULT_PREPROCESS_CONFIG.support_threshold:
            raise LowSupportError(
                f"LOW_SUPPORT: support={support} is below threshold={DEFAULT_PREPROCESS_CONFIG.support_threshold}"
            )
        patch, origin = crop_exact_residue(
            normalized.rgb8,
            crop_size=DEFAULT_PREPROCESS_CONFIG.crop_size,
            residue_x=DEFAULT_PREPROCESS_CONFIG.residue_x,
            residue_y=DEFAULT_PREPROCESS_CONFIG.residue_y,
        )
    except UnsupportedInputError as exc:
        raise PreprocessingError(str(exc)) from exc
    except LowSupportError as exc:
        raise PreprocessingError(str(exc)) from exc

    return patch, PreprocessMetadata(
        format_detected=decoded_format,
        input_mode=input_mode,
        normalized_mode=normalized.normalized_mode,
        alpha_composited=normalized.alpha_composited,
        orientation=orientation,
        crop_origin_x=origin.x,
        crop_origin_y=origin.y,
    )


def _feature_vector(
    patch: np.ndarray,
    *,
    feature_columns: Sequence[str],
) -> tuple[dict[str, float], np.ndarray]:
    try:
        features = extract_feature_vector(patch, config=DEFAULT_FEATURE_CONFIG)
    except Exception as exc:
        raise FeatureExtractionError(f"feature extraction failed: {type(exc).__name__}: {exc}") from exc

    missing = [name for name in feature_columns if name not in features]
    if missing:
        raise FeatureExtractionError(f"Missing required features for runtime schema: {missing}")

    vector = np.asarray([float(features[name]) for name in feature_columns], dtype=np.float64).reshape(1, -1)
    if not np.isfinite(vector).all():
        bad = [feature_columns[idx] for idx in np.where(~np.isfinite(vector[0]))[0].tolist()]
        raise FeatureExtractionError(f"Non-finite values detected in runtime feature vector: {bad}")
    return features, vector


def _raw_margin(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=np.float64)
    if hasattr(model, "predict_proba"):
        prob = np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
        return safe_logit(prob)
    raise FeatureExtractionError(f"Model {type(model).__name__} does not expose decision_function or predict_proba")


def _calibrated_probability(platt: Any, raw_margin: np.ndarray) -> np.ndarray:
    return np.asarray(platt.predict_proba(raw_margin.reshape(-1, 1))[:, 1], dtype=np.float64)


def _top_contributors(
    model: Any,
    x: np.ndarray,
    feature_columns: Sequence[str],
    *,
    top_n: int = 3,
) -> list[dict[str, float | str]]:
    try:
        if hasattr(model, "booster_"):
            contrib = np.asarray(model.booster_.predict(x, pred_contrib=True), dtype=np.float64)
            values = contrib[0, :-1]
        elif hasattr(model, "named_steps") and "clf" in model.named_steps:
            transformed = model[:-1].transform(x) if len(model.steps) > 1 else x
            coef = np.asarray(model.named_steps["clf"].coef_[0], dtype=np.float64)
            values = np.asarray(transformed[0], dtype=np.float64) * coef
        elif hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=np.float64)
        else:
            return []
    except Exception as exc:
        logger.warning("top contributor computation failed: %s", exc)
        return []

    magnitude = np.abs(values)
    if magnitude.size == 0:
        return []
    total = float(np.sum(magnitude))
    if total <= 0.0:
        shares = np.zeros_like(magnitude, dtype=np.float64)
    else:
        shares = magnitude / total
    order = np.argsort(magnitude)[::-1][:top_n]
    contributors: list[dict[str, float | str]] = []
    for idx in order:
        contributors.append(
            {
                "feature": str(feature_columns[idx]),
                "importance": float(shares[idx]),
            }
        )
    return contributors


def _route(score: float, *, low_threshold: float, operating_threshold: float) -> tuple[str, str]:
    if score < low_threshold:
        return "LOW", "pass"
    if score >= operating_threshold:
        return "HIGH", "flag"
    return "MEDIUM", "review"


class InferencePipeline:
    """Load-once active inference runtime."""

    def __init__(
        self,
        artifacts_dir: Path | str | None = None,
        *,
        max_upload_bytes: int = MAX_UPLOAD_BYTES,
        low_threshold: float = LOW_THRESHOLD,
        log_level: int = logging.INFO,
    ) -> None:
        _setup_logging(log_level)
        self.max_upload_bytes = int(max_upload_bytes)
        self.low_threshold = float(low_threshold)
        t0 = time.perf_counter()
        self.artifacts = _load_runtime_artifacts(artifacts_dir)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Active runtime ready artifact=%s candidate=%s features=%d load=%.0fms",
            self.artifacts.artifact_dir.name,
            self.artifacts.candidate_name,
            len(self.artifacts.feature_columns),
            elapsed,
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "artifact_dir": str(self.artifacts.artifact_dir),
            "artifact_version": self.artifacts.artifact_dir.name,
            "candidate_name": self.artifacts.candidate_name,
            "feature_set": self.artifacts.feature_set,
            "model_name": self.artifacts.model_name,
            "model_version": self.artifacts.candidate_name,
            "preprocess_version": self.artifacts.preprocess_version,
            "feature_version": self.artifacts.feature_version,
            "operating_threshold": self.artifacts.threshold,
            "low_threshold": self.low_threshold,
            "target_fpr": self.artifacts.target_fpr,
            "n_features": len(self.artifacts.feature_columns),
        }

    def predict_from_bytes(self, payload: bytes) -> PredictionResult:
        started = time.perf_counter()
        if not payload:
            raise InvalidImageError("Empty upload payload.")
        if len(payload) > self.max_upload_bytes:
            limit_mb = self.max_upload_bytes / (1024 * 1024)
            actual_mb = len(payload) / (1024 * 1024)
            raise ImageTooLargeError(f"File size {actual_mb:.1f} MB exceeds limit of {limit_mb:.0f} MB.")

        image_hash = _compute_hash(payload)
        patch, meta = _decode_canonical_patch(payload)
        _, vector = _feature_vector(patch, feature_columns=self.artifacts.feature_columns)
        raw_margin = _raw_margin(self.artifacts.model, vector)
        calibrated = _calibrated_probability(self.artifacts.platt, raw_margin)
        score = float(calibrated[0])
        raw_score = float(raw_margin[0])
        zone, decision = _route(
            score,
            low_threshold=self.low_threshold,
            operating_threshold=self.artifacts.threshold,
        )
        contributors = _top_contributors(
            self.artifacts.model,
            vector,
            self.artifacts.feature_columns,
            top_n=3,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return PredictionResult(
            calibrated_score=score,
            raw_score=raw_score,
            zone=zone,
            decision=decision,
            top_contributors=contributors,
            image_hash=image_hash,
            format_detected=meta.format_detected,
            preprocess_version=self.artifacts.preprocess_version,
            feature_version=self.artifacts.feature_version,
            model_version=self.artifacts.candidate_name,
            candidate_name=self.artifacts.candidate_name,
            artifact_version=self.artifacts.artifact_dir.name,
            low_threshold=self.low_threshold,
            operating_threshold=self.artifacts.threshold,
            elapsed_ms=elapsed_ms,
        )

    def predict_from_file(self, path: Path | str) -> PredictionResult:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Image not found: {file_path}")
        return self.predict_from_bytes(file_path.read_bytes())

    def predict_batch(self, paths: Sequence[Path | str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for path in paths:
            current = Path(path)
            try:
                result = self.predict_from_file(current)
                row = result.to_dict()
                row["file"] = str(current)
                row["status"] = "ok"
                results.append(row)
            except Exception as exc:
                results.append(
                    {
                        "file": str(current),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return results


def _collect_images(target: Path, glob_pattern: str = "*") -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(
            path
            for path in target.glob(glob_pattern)
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()
        )
    return []


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="AI Image Detector active runtime")
    parser.add_argument("target", nargs="?", default=None, help="Image file or directory to score.")
    parser.add_argument("--glob", default="*", help="Glob pattern when target is a directory.")
    parser.add_argument("--artifacts-dir", default=None, help="Optional override for models/param/<run>.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    pipeline = InferencePipeline(artifacts_dir=args.artifacts_dir)

    if args.target is None:
        print(json.dumps(pipeline.health(), indent=2, ensure_ascii=False))
        return

    target = Path(args.target)
    images = _collect_images(target, args.glob)
    if not images:
        print(f"No JPEG/PNG images found at: {target}")
        sys.exit(1)

    results = pipeline.predict_batch(images)
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    for row in results:
        if row["status"] == "ok":
            print(
                f"{Path(row['file']).name:40s} "
                f"score={row['calibrated_score']:.4f} "
                f"zone={row['zone']:6s} "
                f"decision={row['decision']:6s} "
                f"{row['elapsed_ms']:.0f}ms"
            )
        else:
            print(f"{Path(row['file']).name:40s} ERROR: {row['error']}")


if __name__ == "__main__":
    main()
