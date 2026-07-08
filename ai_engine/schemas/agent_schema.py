from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


class AgentOutput(BaseModel):
    """
    Standard response returned by every AI agent.
    """

    agent_name: str = Field(
        ...,
        description="Name of the agent"
    )

    risk_level: RiskLevel = Field(
        ...,
        description="Risk level assessed by the agent"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )

    key_findings: List[str] = Field(
        default_factory=list,
        description="Most important observations"
    )

    reasoning: str = Field(
        ...,
        description="Human-readable explanation"
    )

    evidence: List[str] = Field(
        default_factory=list,
        description="Supporting evidence extracted from the input"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific structured information"
    )