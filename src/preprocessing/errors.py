"""Custom exceptions for preprocessing v4."""

from __future__ import annotations


class PreprocessingError(RuntimeError):
    """Base class for preprocessing failures."""


class DecodeImageError(PreprocessingError):
    """Raised when the source bytes cannot be decoded as an image."""


class UnsupportedInputError(PreprocessingError):
    """Raised when the decoded image violates the v4 input contract."""


class LowSupportError(PreprocessingError):
    """Raised when the image is too small for the exact residue crop."""
