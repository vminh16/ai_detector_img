"""
inference/api.py
================
FastAPI application — single ``POST /predict`` endpoint + health check.

Lifecycle:
    1. On startup, ``load_artifacts`` loads the champion model and
       calibrator into a module-level ``Artifacts`` singleton.
    2. Each request flows through:
       validation → preprocessing → featurisation → inference →
       calibration → triage routing → response assembly.
    3. Errors are caught at the appropriate layer and returned as
       structured JSON (never a raw stack trace).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from inference.artifact_loader import Artifacts, load_artifacts
from inference.calibration import predict as model_predict
from inference.config import InferenceConfig, load_config
from inference.errors import (
    ImageTooLargeError,
    InferenceBaseError,
    InvalidImageError,
    UnsupportedFormatError,
)
from inference.explainer import top_contributors
from inference.feature_vectorizer import featurise
from inference.preprocessing import preprocess_image
from inference.routing import route
from inference.schemas import (
    ErrorResponse,
    HealthResponse,
    InferenceResult,
)
from inference.telemetry import compute_hash, logger, setup_logging, timer
from inference.validation import validate_payload

# ── Module-level singleton (populated at startup) ────────────────
_artifacts: Artifacts | None = None
_config: InferenceConfig | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: load config + artefacts.  Shutdown: no-op."""
    global _artifacts, _config
    setup_logging(logging.INFO)
    logger.info("Loading configuration and artifacts …")
    _config = load_config()
    _artifacts = load_artifacts(_config)
    logger.info(
        "Ready — model=%s  tau_op=%.6f  features=%d",
        _config.model_version,
        _config.tau_op,
        len(_config.feature_order),
    )
    yield
    logger.info("Shutting down inference service.")


app = FastAPI(
    title="AI Image Detector — Inference API",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Error handlers ───────────────────────────────────────────────

@app.exception_handler(InferenceBaseError)
async def inference_error_handler(_request, exc: InferenceBaseError):
    status_map = {
        UnsupportedFormatError: 415,
        ImageTooLargeError: 413,
        InvalidImageError: 422,
    }
    status = status_map.get(type(exc), 500)
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(error=type(exc).__name__, detail=str(exc)).model_dump(),
    )


# ── Health check ─────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=_artifacts is not None,
        model_version=_config.model_version if _config else None,
    )


# ── Main prediction endpoint ────────────────────────────────────

@app.post("/predict", response_model=InferenceResult)
async def predict(file: UploadFile = File(...)):
    """Score a single image for AI-generated probability.

    Accepts JPEG, PNG, or WebP (max 20 MB).
    """
    if _artifacts is None or _config is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # ---- 1. Read payload ----
    with timer("read_upload") as t_read:
        payload: bytes = await file.read()
    logger.info(
        "Received %d bytes (%s)  read=%.1fms",
        len(payload), file.filename or "unknown", t_read["elapsed_ms"],
    )

    # ---- 2. Validate ----
    with timer("validation"):
        fmt = validate_payload(payload, _config)

    # ---- 3. Hash ----
    image_hash = compute_hash(payload)

    # ---- 4. Preprocess ----
    with timer("preprocess") as t_pre:
        ycrcb = preprocess_image(payload, image_hash, _config)
    logger.debug("preprocess %.1fms", t_pre["elapsed_ms"])

    # ---- 5. Featurise ----
    with timer("featurise") as t_feat:
        features_dict, vector = featurise(ycrcb, _config)
    logger.debug("featurise %.1fms  features=%d", t_feat["elapsed_ms"], len(features_dict))

    # ---- 6. Predict ----
    with timer("inference") as t_inf:
        raw_score, calibrated = model_predict(vector, _artifacts)
    logger.debug("inference %.1fms  raw=%.4f  cal=%.4f", t_inf["elapsed_ms"], raw_score, calibrated)

    # ---- 7. Route ----
    zone, decision = route(calibrated, _config)

    # ---- 8. Explain ----
    contributors = top_contributors(_artifacts, n=3)

    # ---- 9. Assemble response ----
    result = InferenceResult(
        calibrated_score=round(calibrated, 6),
        raw_score=round(raw_score, 6),
        zone=zone,
        decision=decision,
        top_contributors=contributors,
        image_hash=image_hash,
        format_detected=fmt,
        preprocess_version=_config.preprocess_version,
        model_version=_config.model_version,
    )

    logger.info(
        "hash=%s  score=%.4f  zone=%s  decision=%s",
        image_hash[:12], calibrated, zone.value, decision.value,
    )

    return result
