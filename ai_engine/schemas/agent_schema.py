"""


Defines the standard output returned by every AI Risk Agent.
All specialized agents (Behavior, Income, Transaction, Document)
must return this schema.
"""

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


class AgentOutput(BaseModel):
    """
    Standard response returned by every AI Risk Agent.
    """

    agent_name: str = Field(
        ...,
        description="Name of the AI agent"
    )

    risk_level: RiskLevel = Field(
        ...,
        description="Risk level assigned by the agent"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )

    summary: str = Field(
        ...,
        description="Executive summary of the assessment"
    )

    key_findings: List[str] = Field(
        default_factory=list,
        description="Most important findings"
    )

    evidence: List[str] = Field(
        default_factory=list,
        description="Supporting evidence used by the agent"
    )

    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommended actions suggested by the agent"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific structured information"
    )

    agent_version: str = Field(
        default="1.0",
        description="Version of the AI agent"
    )

    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Time taken by the agent in milliseconds"
    )