"""
Chief Risk Officer output schema.
"""

from typing import List, Literal

from pydantic import BaseModel, Field

RiskLevel = Literal[
    "LOW",
    "MEDIUM",
    "HIGH",
]


class CROOutput(BaseModel):
    """
    Final executive decision produced
    by the Chief Risk Officer.
    """

    overall_risk: RiskLevel = Field(
        ...,
        description="Overall borrower risk."
    )

    confidence: float = Field(
        ...,
        ge=0,
        le=1,
    )

    executive_summary: str = Field(
        ...,
    )

    top_risk_drivers: List[str] = Field(
        default_factory=list,
    )

    conflicting_assessments: List[str] = Field(
        default_factory=list,
    )

    recommended_actions: List[str] = Field(
        default_factory=list,
    )

    approval_recommendation: Literal[
        "APPROVE",
        "MANUAL_REVIEW",
        "REJECT",
    ] = Field(
        ...,
    )

    processing_time_ms: float = Field(
        default=0,
    )

    metadata: dict = Field(
        default_factory=dict,
    )