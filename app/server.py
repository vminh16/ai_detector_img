"""
app/server.py
=============
FastAPI backend for the AI Image Detector Web UI.

Loads the InferencePipeline once at startup and exposes:
  POST /api/predict   — upload an image, get scored
  GET  /api/health    — pipeline health check

Run::

    cd <project_root>
    uvicorn app.server:app --host 127.0.0.1 --port 8000 --reload
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from deploy.pipeline import InferencePipeline
from inference.errors import (
    FeatureExtractionError,
    ImageTooLargeError,
    InvalidImageError,
    ModelLoadError,
    PreprocessingError,
    UnsupportedFormatError,
)

logger = logging.getLogger("app.server")

# ── Constants ────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
STATIC_DIR = Path(__file__).resolve().parent / "static"

# ── Pipeline singleton ───────────────────────────────────────────────
_pipeline: InferencePipeline | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the inference pipeline once on startup."""
    global _pipeline
    logger.info("Loading InferencePipeline …")
    t0 = time.perf_counter()
    _pipeline = InferencePipeline()
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("Pipeline ready in %.0f ms", elapsed)
    yield
    _pipeline = None
    logger.info("Pipeline released.")


app = FastAPI(
    title="AI Image Detector",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow local frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API routes ───────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Quick health-check."""
    if _pipeline is None:
        raise HTTPException(503, detail="Pipeline not loaded")
    return JSONResponse(content=_pipeline.health())


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Score a single uploaded image.

    Accepts JPEG or PNG.  Returns calibrated score, zone,
    decision, top contributors, and timing info.
    """
    if _pipeline is None:
        raise HTTPException(503, detail="Pipeline not loaded")

    # ── Content-type guard (advisory — real validation by pipeline) ───
    ct = (file.content_type or "").lower()
    if ct and ct not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            400,
            detail=f"Unsupported content type: {ct}. "
                   f"Accepted: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    # ── Read payload ─────────────────────────────────────────────────
    payload = await file.read()
    if len(payload) == 0:
        raise HTTPException(400, detail="Empty file")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            detail=f"File too large ({len(payload) / 1024 / 1024:.1f} MB). "
                   f"Max: {MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB",
        )

    # ── Run pipeline ─────────────────────────────────────────────────
    try:
        result = _pipeline.predict_from_bytes(payload)
    except UnsupportedFormatError as exc:
        raise HTTPException(415, detail=str(exc)) from exc
    except ImageTooLargeError as exc:
        raise HTTPException(413, detail=str(exc)) from exc
    except (InvalidImageError, PreprocessingError) as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    except (FeatureExtractionError, ModelLoadError) as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(500, detail=f"Prediction error: {exc}") from exc
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(500, detail="Prediction error: unexpected internal failure.") from exc

    resp = result.to_dict()
    resp["server_hash"] = result.image_hash
    resp["filename"] = file.filename or "unknown"
    return JSONResponse(content=resp)


# ── Serve static frontend ───────────────────────────────────────────
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
