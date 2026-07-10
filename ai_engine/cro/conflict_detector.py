"""
Detects disagreements between specialist agents.
"""

from __future__ import annotations

from typing import List

from ai_engine.schemas.agent_schema import AgentOutput


class ConflictDetector:

    @staticmethod
    def detect(
        agents: List[AgentOutput],
    ) -> List[str]:

        conflicts = []

        risks = {
            agent.agent_name: agent.risk_level
            for agent in agents
        }

        values = set(risks.values())

        if len(values) <= 1:
            return conflicts

        high_agents = [
            name
            for name, risk in risks.items()
            if risk == "HIGH"
        ]

        low_agents = [
            name
            for name, risk in risks.items()
            if risk == "LOW"
        ]

        if high_agents and low_agents:

            conflicts.append(
                "Significant disagreement exists between specialist agents."
            )

        elif "HIGH" in values and "MEDIUM" in values:

            conflicts.append(
                "Some agents indicate elevated risk while others indicate moderate risk."
            )

        return conflicts