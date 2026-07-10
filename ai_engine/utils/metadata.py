"""
Utility for building standardized metadata returned by AI agents.
"""

from __future__ import annotations

from typing import Any, Dict

from ai_engine.schemas.request_schema import AIRequest


class MetadataBuilder:
    """
    Builds standardized metadata for AI agent responses.
    """

    @staticmethod
    def build(
        request: AIRequest,
        rule_score: int,
        rule_risk: str,
    ) -> Dict[str, Any]:

        return {
            "rule_engine": {
                "score": rule_score,
                "risk_level": rule_risk,
            },
            "ml_model": {
                "default_probability": request.default_probability,
                "confidence": request.model_confidence,
            },
            "explainability": {
                "top_features": request.shap_values,
            },
            "history": {
                "similar_cases": len(request.similar_cases),
            },
            "loan": {
                "loan_id": request.loan_id,
                "loan_type": request.loan_type,
            },
        }