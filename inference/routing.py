"""
inference/routing.py
====================
Triage routing — maps a calibrated probability to a zone + decision.

Zone definitions (from threshold_lock.json):
  LOW    :  score < 0.3           → auto-pass  (likely natural)
  MEDIUM :  0.3 ≤ score < tau_op  → manual review
  HIGH   :  score ≥ tau_op        → flag as AI-generated
"""
from __future__ import annotations

from inference.config import InferenceConfig
from inference.schemas import Decision, TriageZone


def route(score: float, config: InferenceConfig) -> tuple[TriageZone, Decision]:
    """Assign a triage zone and recommended action.

    Parameters
    ----------
    score : calibrated probability of AI-generated.
    config : provides ``low_threshold`` and ``tau_op``.

    Returns
    -------
    (zone, decision)
    """
    if score < config.low_threshold:
        return TriageZone.LOW, Decision.PASS
    if score >= config.tau_op:
        return TriageZone.HIGH, Decision.FLAG
    return TriageZone.MEDIUM, Decision.REVIEW
