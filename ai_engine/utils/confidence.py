"""
Enterprise confidence calibration for Risk360 AI.

Combines:
- ML confidence
- Rule engine confidence
- Data completeness
"""

from __future__ import annotations

from typing import Dict

from ai_engine.schemas.request_schema import AIRequest


class ConfidenceEngine:
    """
    Calculates a calibrated confidence score for AI agents.
    """

    @staticmethod
    def calculate(
        request: AIRequest,
        rule_score: int,
    ) -> float:

        # ML confidence (0-1)
        ml = request.model_confidence

        # Rule confidence
        if rule_score >= 80:
            rule = 0.95
        elif rule_score >= 60:
            rule = 0.85
        elif rule_score >= 40:
            rule = 0.75
        elif rule_score >= 20:
            rule = 0.65
        else:
            rule = 0.55

        # Data completeness
        fields = [
            request.borrower_profile,
            request.shap_values,
            request.similar_cases,
        ]

        available = sum(bool(f) for f in fields)
        completeness = available / len(fields)

        confidence = (
            0.60 * ml +
            0.25 * rule +
            0.15 * completeness
        )

        return round(min(confidence, 1.0), 2)