"""
Enterprise Risk Fusion Engine

Combines specialist agent assessments using
loan-type specific weighting.
"""

from __future__ import annotations

from typing import Dict, List

from ai_engine.schemas.agent_schema import AgentOutput


class RiskFusion:

    RISK_MAP = {
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
    }

    ####################################################################
    # Loan-specific weights
    ####################################################################

    LOAN_WEIGHTS = {

        "home loan": {
            "Behavior Agent": 0.30,
            "Income Agent": 0.40,
            "Transaction Agent": 0.10,
            "Document Agent": 0.20,
        },

        "personal loan": {
            "Behavior Agent": 0.40,
            "Income Agent": 0.30,
            "Transaction Agent": 0.20,
            "Document Agent": 0.10,
        },

        "credit card": {
            "Behavior Agent": 0.30,
            "Income Agent": 0.15,
            "Transaction Agent": 0.45,
            "Document Agent": 0.10,
        },

        "auto loan": {
            "Behavior Agent": 0.30,
            "Income Agent": 0.30,
            "Transaction Agent": 0.20,
            "Document Agent": 0.20,
        },
    }

    DEFAULT_WEIGHTS = {
        "Behavior Agent": 0.25,
        "Income Agent": 0.25,
        "Transaction Agent": 0.25,
        "Document Agent": 0.25,
    }

    ####################################################################

    @classmethod
    def get_weights(
        cls,
        loan_type: str,
    ) -> Dict[str, float]:

        return cls.LOAN_WEIGHTS.get(
            loan_type.lower(),
            cls.DEFAULT_WEIGHTS,
        )

    ####################################################################

    @classmethod
    def overall_risk(
        cls,
        loan_type: str,
        agents: List[AgentOutput],
    ) -> str:

        weights = cls.get_weights(
            loan_type,
        )

        weighted_score = 0

        total_weight = 0

        for agent in agents:

            confidence = max(
                agent.confidence,
                0.1,
            )

            weight = weights.get(
                agent.agent_name,
                0.25,
            )

            weighted_score += (
                cls.RISK_MAP[agent.risk_level]
                * confidence
                * weight
            )

            total_weight += (
                confidence
                * weight
            )

        average = weighted_score / total_weight

        if average >= 2.5:
            return "HIGH"

        if average >= 1.5:
            return "MEDIUM"

        return "LOW"