"""
inference/main.py
=================
Uvicorn entry point for the inference service.

Usage::

    python -m inference.main
    # or
    uvicorn inference.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run(
        "inference.api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
