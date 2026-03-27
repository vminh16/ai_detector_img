"""AI Image Detector — Production Inference Service.

Modules
-------
config              – Centralised settings and artifact-path resolution
errors              – Custom exception hierarchy
schemas             – Pydantic request / response models
validation          – Magic-byte detection and size checks
preprocessing       – EXIF → pad → crop → JPEG bottleneck → YCrCb
feature_vectorizer  – 33-dim feature extraction + winsorisation
artifact_loader     – Load champion model, calibrator, and metadata
calibration         – Platt-scaling raw score → P(AI)
routing             – LOW / MEDIUM / HIGH triage zones
explainer           – Top-N feature contributors
telemetry           – Structured logging and latency tracking
api                 – FastAPI application (POST /predict, GET /health)
main                – Uvicorn launcher
"""
