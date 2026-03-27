"""
deploy/pipeline.py
==================
Self-contained, standalone inference pipeline for the AI Image Detector.

This module is **completely independent** from the ``inference/`` FastAPI
service.  It imports only the logic modules from ``inference.*`` and
orchestrates them in a single synchronous ``predict_from_file()`` /
``predict_from_bytes()`` call — no server, no REST, no async.

Use cases
---------
- CLI batch scoring (e.g. loop over a directory of images)
- Notebook integration (import and call directly)
- CI smoke tests (verify the pipeline loads and scores)
- Any context where spinning up a server is unnecessary

Usage
-----
::

    # From project root
    python -m deploy.pipeline                              # demo with self-test
    python -m deploy.pipeline  path/to/image.jpg           # score one file
    python -m deploy.pipeline  path/to/dir --glob "*.png"  # batch scoring

API
---
::

    from deploy.pipeline import InferencePipeline

    pipe = InferencePipeline()                  # loads config + artifacts once
    result = pipe.predict_from_file("photo.jpg")
    print(result)
    # PredictionResult(calibrated_score=0.123, zone='LOW', decision='pass', ...)
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

# ── Inference module imports ─────────────────────────────────────────
from inference.config import InferenceConfig, load_config
from inference.artifact_loader import Artifacts, load_artifacts
from inference.validation import validate_payload
from inference.preprocessing import preprocess_image
from inference.feature_vectorizer import featurise
from inference.calibration import predict as model_predict
from inference.routing import route
from inference.explainer import top_contributors
from inference.telemetry import compute_hash, setup_logging, timer
from inference.schemas import TriageZone, Decision, FeatureContributor
from inference.errors import InferenceBaseError

logger = logging.getLogger("deploy.pipeline")


# =====================================================================
#  Result container
# =====================================================================

@dataclass
class PredictionResult:
    """Plain-dataclass result — no Pydantic dependency for CLI use."""

    calibrated_score: float
    raw_score: float
    zone: str
    decision: str
    top_contributors: list[dict]
    image_hash: str
    format_detected: str
    preprocess_version: str
    model_version: str
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "calibrated_score": round(self.calibrated_score, 6),
            "raw_score": round(self.raw_score, 6),
            "zone": self.zone,
            "decision": self.decision,
            "top_contributors": self.top_contributors,
            "image_hash": self.image_hash,
            "format_detected": self.format_detected,
            "preprocess_version": self.preprocess_version,
            "model_version": self.model_version,
            "elapsed_ms": self.elapsed_ms,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# =====================================================================
#  Pipeline class
# =====================================================================

class InferencePipeline:
    """End-to-end inference pipeline — load once, predict many.

    Parameters
    ----------
    artifacts_dir : optional override for ``models/param``.
    log_level : logging verbosity (default ``INFO``).
    """

    def __init__(
        self,
        artifacts_dir: Path | str | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        setup_logging(log_level)
        logger.info("Initialising InferencePipeline …")

        t0 = time.perf_counter()
        self.config: InferenceConfig = load_config(artifacts_dir)
        self.artifacts: Artifacts = load_artifacts(self.config)
        elapsed = (time.perf_counter() - t0) * 1000

        logger.info(
            "Pipeline ready  model=%s  tau=%.6f  features=%d  load=%.0fms",
            self.config.model_version,
            self.config.tau_op,
            len(self.config.feature_order),
            elapsed,
        )

    # ── Public API ───────────────────────────────────────────────

    def predict_from_bytes(self, payload: bytes) -> PredictionResult:
        """Score raw image bytes through the full pipeline.

        Stages
        ------
        1. Validate  (magic bytes + size check)
        2. Hash      (SHA-256 for deterministic Q + audit trail)
        3. Preprocess (EXIF → pad → crop → JPEG bottleneck → YCrCb)
        4. Featurise  (33-dim vector, winsorised)
        5. Predict    (LightGBM raw → Platt calibration)
        6. Route      (LOW / MEDIUM / HIGH)
        7. Explain    (top-3 gain contributors)
        """
        t_start = time.perf_counter()

        # 1. Validate
        fmt = validate_payload(payload, self.config)

        # 2. Hash
        image_hash = compute_hash(payload)

        # 3. Preprocess
        ycrcb = preprocess_image(payload, image_hash, self.config)

        # 4. Featurise
        features_dict, vector = featurise(ycrcb, self.config)

        # 5. Predict
        raw_score, calibrated = model_predict(vector, self.artifacts)

        # 6. Route
        zone, decision = route(calibrated, self.config)

        # 7. Explain
        contribs = top_contributors(self.artifacts, n=3)

        elapsed = (time.perf_counter() - t_start) * 1000

        return PredictionResult(
            calibrated_score=calibrated,
            raw_score=raw_score,
            zone=zone.value,
            decision=decision.value,
            top_contributors=[
                {"feature": c.feature, "importance": c.importance}
                for c in contribs
            ],
            image_hash=image_hash,
            format_detected=fmt,
            preprocess_version=self.config.preprocess_version,
            model_version=self.config.model_version,
            elapsed_ms=round(elapsed, 2),
        )

    def predict_from_file(self, path: Path | str) -> PredictionResult:
        """Read an image file from disk and score it."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        payload = path.read_bytes()
        return self.predict_from_bytes(payload)

    def predict_batch(
        self,
        paths: Sequence[Path | str],
    ) -> list[dict]:
        """Score a batch of image files.  Returns list of result dicts.

        Errors for individual files are caught and recorded in the
        ``"error"`` key rather than aborting the entire batch.
        """
        results: list[dict] = []
        for p in paths:
            p = Path(p)
            try:
                res = self.predict_from_file(p)
                d = res.to_dict()
                d["file"] = str(p)
                d["status"] = "ok"
                results.append(d)
            except Exception as exc:
                results.append({
                    "file": str(p),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
        return results

    # ── Convenience ──────────────────────────────────────────────

    def health(self) -> dict:
        """Quick health check (no image needed)."""
        return {
            "status": "ok",
            "model_version": self.config.model_version,
            "champion_pipeline": self.config.champion_pipeline,
            "tau_op": self.config.tau_op,
            "n_features": len(self.config.feature_order),
        }


# =====================================================================
#  CLI entry point
# =====================================================================

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def _collect_images(target: Path, glob_pattern: str = "*") -> list[Path]:
    """Collect image files from a path (file or directory)."""
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(
            p for p in target.glob(glob_pattern)
            if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
        )
    return []


def main() -> None:
    """CLI entry point for deploy/pipeline.py."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Image Detector — standalone inference pipeline"
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Image file or directory to score. Omit for self-test.",
    )
    parser.add_argument(
        "--glob",
        default="*",
        help="Glob pattern when target is a directory (default: '*').",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Override path to models/param/.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )
    args = parser.parse_args()

    # ── Initialise pipeline ──────────────────────────────────────
    pipe = InferencePipeline(artifacts_dir=args.artifacts_dir)

    # ── Self-test mode ───────────────────────────────────────────
    if args.target is None:
        print("\n=== Health Check ===")
        h = pipe.health()
        for k, v in h.items():
            print(f"  {k}: {v}")
        print("\nNo image target provided.  Pass a file or directory to score.")
        return

    # ── Score images ─────────────────────────────────────────────
    target = Path(args.target)
    images = _collect_images(target, args.glob)

    if not images:
        print(f"No images found at: {target}")
        sys.exit(1)

    print(f"\nScoring {len(images)} image(s) …\n")

    results = pipe.predict_batch(images)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for r in results:
            status = r.get("status", "?")
            if status == "ok":
                print(
                    f"  {Path(r['file']).name:40s}  "
                    f"score={r['calibrated_score']:.4f}  "
                    f"zone={r['zone']:6s}  "
                    f"decision={r['decision']:6s}  "
                    f"{r['elapsed_ms']:.0f}ms"
                )
            else:
                print(f"  {Path(r['file']).name:40s}  ERROR: {r.get('error', '?')}")

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] != "ok"]
    print(f"\n  Total: {len(results)}  OK: {len(ok)}  Errors: {len(err)}")
    if ok:
        scores = [r["calibrated_score"] for r in ok]
        zones = [r["zone"] for r in ok]
        print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}]")
        for z in ["LOW", "MEDIUM", "HIGH"]:
            cnt = zones.count(z)
            if cnt:
                print(f"    {z}: {cnt}")


if __name__ == "__main__":
    main()
