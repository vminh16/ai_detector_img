"""
inference/validation.py
=======================
Input validation: magic-byte format detection and size checks.
"""
from __future__ import annotations

from inference.config import InferenceConfig
from inference.errors import ImageTooLargeError, UnsupportedFormatError


def detect_format(payload: bytes, config: InferenceConfig) -> str:
    """Return format label (``"jpeg"``, ``"png"``, ``"webp"``) from magic bytes.

    Raises ``UnsupportedFormatError`` when no accepted signature matches.
    """
    for header, label in config.accepted_signatures:
        if payload[:len(header)] == header:
            # WebP: bytes 8-12 must also equal b"WEBP"
            if label == "webp" and payload[8:12] != b"WEBP":
                continue
            return label
    raise UnsupportedFormatError(
        "Unsupported image format. Accepted: JPEG, PNG, WebP."
    )


def validate_size(payload: bytes, config: InferenceConfig) -> None:
    """Raise ``ImageTooLargeError`` if *payload* exceeds the limit."""
    if len(payload) > config.max_upload_bytes:
        limit_mb = config.max_upload_bytes / (1024 * 1024)
        actual_mb = len(payload) / (1024 * 1024)
        raise ImageTooLargeError(
            f"File size {actual_mb:.1f} MB exceeds limit of {limit_mb:.0f} MB."
        )


def validate_payload(payload: bytes, config: InferenceConfig) -> str:
    """Run all input-level checks.  Returns the detected format label."""
    validate_size(payload, config)
    return detect_format(payload, config)
