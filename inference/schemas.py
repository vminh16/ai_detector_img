"""
inference/schemas.py
====================
Pydantic request / response models for the REST API.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


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

    model_config = ConfigDict(protected_namespaces=())
    feature: str = Field(..., description="Canonical feature name")
    importance: float = Field(..., description="Gain-based importance share (0-1)")


class InferenceResult(BaseModel):
    """Full response payload for a single prediction."""

    model_config = ConfigDict(protected_namespaces=())
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
    feature_version: str = Field(
        ..., description="Feature extraction pipeline version",
    )
    model_version: str = Field(
        ..., description="Selected model identifier",
    )
    candidate_name: str = Field(
        ..., description="Candidate name saved by the training phase closure run",
    )
    artifact_version: str = Field(
        ..., description="Artifact directory version used by the runtime",
    )
    low_threshold: float = Field(
        ..., ge=0.0, le=1.0,
        description="Lower routing threshold for auto-pass",
    )
    operating_threshold: float = Field(
        ..., ge=0.0, le=1.0,
        description="Operating threshold used to flag likely AI images",
    )
    elapsed_ms: float = Field(
        ..., ge=0.0,
        description="End-to-end inference latency in milliseconds",
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str = "ok"
    model_loaded: bool = False
    model_version: Optional[str] = None
    candidate_name: Optional[str] = None
    feature_version: Optional[str] = None
    preprocess_version: Optional[str] = None
    artifact_version: Optional[str] = None
    feature_set: Optional[str] = None
    model_name: Optional[str] = None
    operating_threshold: Optional[float] = None
    low_threshold: Optional[float] = None
    target_fpr: Optional[float] = None
    n_features: Optional[int] = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    error: str
    detail: Optional[str] = None
