"""AI Image Detector — active FastAPI inference surface.

Active entrypoints
------------------
- ``inference.api``  : FastAPI app bound to the active runtime stack
- ``inference.main`` : Uvicorn launcher for ``inference.api``

Notes
-----
The lower-level legacy modules under ``inference/`` are kept in the repo for
historical reference. The active web/API runtime is now implemented in
``deploy.pipeline`` and used by both ``app.server`` and ``inference.api``.
"""
