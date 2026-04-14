"""FastAPI inference API bound to the active runtime stack."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from deploy.pipeline import InferencePipeline
from inference.errors import (
    FeatureExtractionError,
    ImageTooLargeError,
    InvalidImageError,
    ModelLoadError,
    PreprocessingError,
    UnsupportedFormatError,
)
from inference.schemas import ErrorResponse, HealthResponse, InferenceResult

logger = logging.getLogger("inference.api")

_pipeline: InferencePipeline | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    global _pipeline
    logging.basicConfig(level=logging.INFO)
    logger.info("Loading active inference pipeline …")
    _pipeline = InferencePipeline()
    yield
    logger.info("Shutting down inference service.")
    _pipeline = None


app = FastAPI(
    title="AI Image Detector — Active Inference API",
    version="2.0.0",
    lifespan=lifespan,
)


@app.exception_handler(UnsupportedFormatError)
async def unsupported_format_handler(_request, exc: UnsupportedFormatError):
    return JSONResponse(
        status_code=415,
        content=ErrorResponse(error=type(exc).__name__, detail=str(exc)).model_dump(),
    )


@app.exception_handler(ImageTooLargeError)
async def image_too_large_handler(_request, exc: ImageTooLargeError):
    return JSONResponse(
        status_code=413,
        content=ErrorResponse(error=type(exc).__name__, detail=str(exc)).model_dump(),
    )


@app.exception_handler(InvalidImageError)
async def invalid_image_handler(_request, exc: InvalidImageError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error=type(exc).__name__, detail=str(exc)).model_dump(),
    )


@app.exception_handler(PreprocessingError)
async def preprocessing_failure_handler(_request, exc: PreprocessingError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error=type(exc).__name__, detail=str(exc)).model_dump(),
    )


@app.exception_handler(FeatureExtractionError)
async def feature_failure_handler(_request, exc: FeatureExtractionError):
    logger.exception("Inference runtime failure: %s", exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error=type(exc).__name__, detail=str(exc)).model_dump(),
    )


@app.exception_handler(ModelLoadError)
async def runtime_failure_handler(_request, exc: ModelLoadError):
    logger.exception("Inference runtime failure: %s", exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error=type(exc).__name__, detail=str(exc)).model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return HealthResponse(model_loaded=True, **_pipeline.health())


@app.post("/predict", response_model=InferenceResult)
async def predict(file: UploadFile = File(...)):
    """Score a single JPEG/PNG upload with the active runtime."""

    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    payload = await file.read()
    result = _pipeline.predict_from_bytes(payload)
    return InferenceResult(**result.to_dict())
