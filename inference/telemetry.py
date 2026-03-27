"""
inference/telemetry.py
======================
Structured JSON logging, latency tracking, and hash computation.
"""
from __future__ import annotations

import hashlib
import logging
import time
from contextlib import contextmanager
from typing import Generator

# ── Logger setup ─────────────────────────────────────────────────

logger = logging.getLogger("inference")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured JSON-style logging for the inference service."""
    handler = logging.StreamHandler()
    fmt = logging.Formatter(
        '{"ts":"%(asctime)s","level":"%(levelname)s",'
        '"logger":"%(name)s","msg":"%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(fmt)

    root = logging.getLogger("inference")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


# ── Image hash ───────────────────────────────────────────────────

def compute_hash(payload: bytes) -> str:
    """Return the SHA-256 hex digest of *payload*."""
    return hashlib.sha256(payload).hexdigest()


# ── Latency timer ────────────────────────────────────────────────

@contextmanager
def timer(label: str) -> Generator[dict, None, None]:
    """Context manager that measures wall-clock time in milliseconds.

    Usage::

        with timer("preprocess") as t:
            run_preprocess()
        print(t["elapsed_ms"])
    """
    record: dict = {"label": label, "elapsed_ms": 0.0}
    start = time.perf_counter()
    try:
        yield record
    finally:
        record["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
        logger.debug("%s completed in %.2f ms", label, record["elapsed_ms"])
