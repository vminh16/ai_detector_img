"""
inference/errors.py
===================
Custom exception hierarchy for the inference pipeline.
"""
from __future__ import annotations


class InferenceBaseError(Exception):
    """Base class for all inference pipeline errors."""


class InvalidImageError(InferenceBaseError):
    """Raised when the uploaded payload cannot be decoded as a valid image."""


class UnsupportedFormatError(InferenceBaseError):
    """Raised when the file's magic bytes do not match any accepted format."""


class ImageTooLargeError(InferenceBaseError):
    """Raised when the uploaded file exceeds the size limit."""


class PreprocessingError(InferenceBaseError):
    """Raised when any preprocessing step fails unexpectedly."""


class ModelLoadError(InferenceBaseError):
    """Raised when model or artifact files cannot be loaded."""


class FeatureExtractionError(InferenceBaseError):
    """Raised when feature extraction fails for a given image."""
