"""
Enterprise Bank Approval Policy.
"""

from __future__ import annotations

from typing import List

from ai_engine.schemas.agent_schema import AgentOutput


class ApprovalPolicy:

    @staticmethod
    def recommend(
        overall_risk: str,
        default_probability: float,
        agents: List[AgentOutput],
    ) -> str:

        # Rule 1: Very high ML probability
        if default_probability >= 0.90:
            return "REJECT"

        # Rule 2: Count HIGH-risk agents
        high_agents = sum(
            1 for agent in agents
            if agent.risk_level == "HIGH"
        )

        if high_agents >= 3:
            return "REJECT"

        # Rule 3: Fraud/document override
        for agent in agents:

            if agent.agent_name == "Document Agent":

                reasons = (
                    agent.metadata.get("risk_reasons", [])
                )

                fraud_keywords = [
                    "fraud",
                    "fake",
                    "forged",
                ]

                if any(
                    keyword in reason.lower()
                    for keyword in fraud_keywords
                    for reason in reasons
                ):
                    return "REJECT"

        # Rule 4
        if overall_risk == "HIGH":
            return "REJECT"

        # Rule 5
        if overall_risk == "MEDIUM":
            return "MANUAL_REVIEW"

        return "APPROVE"