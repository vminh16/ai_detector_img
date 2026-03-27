"""
inference/schemas.py
====================
Pydantic request / response models for the REST API.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────

class TriageZone(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Decision(str, Enum):
    PASS = "pass"
    REVIEW = "review"
    FLAG = "flag"


# ── Response sub-models ───────────────────────────────────────────

class FeatureContributor(BaseModel):
    """Top-N feature contributor to the prediction."""
    feature: str = Field(..., description="Canonical feature name")
    importance: float = Field(..., description="Gain-based importance share (0-1)")


class InferenceResult(BaseModel):
    """Full response payload for a single prediction."""
    calibrated_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Platt-calibrated probability of AI-generated",
    )
    raw_score: float = Field(
        ..., description="Un-calibrated LightGBM raw margin (logit)",
    )
    zone: TriageZone = Field(
        ..., description="Triage routing zone",
    )
    decision: Decision = Field(
        ..., description="Recommended action",
    )
    top_contributors: list[FeatureContributor] = Field(
        default_factory=list,
        description="Top-3 features driving the prediction",
    )
    image_hash: str = Field(
        ..., description="SHA-256 hex digest of the uploaded bytes",
    )
    format_detected: str = Field(
        ..., description="Image format detected from magic bytes",
    )
    preprocess_version: str = Field(
        ..., description="Preprocessing pipeline version",
    )
    model_version: str = Field(
        ..., description="Champion model identifier",
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False
    model_version: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
