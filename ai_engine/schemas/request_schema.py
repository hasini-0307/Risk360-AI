"""


Defines the standardized input received by the AI Engine
from the backend.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AIRequest(BaseModel):
    """
    Standard request passed into the AI Engine.
    """

    borrower_id: str = Field(
        ...,
        description="Unique borrower identifier"
    )

    loan_id: str = Field(
        ...,
        description="Unique loan identifier"
    )

    loan_type: str = Field(
        ...,
        description="Type of loan"
    )

    default_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of default predicted by ML model"
    )

    model_confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence score of the prediction model"
    )

    borrower_profile: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured borrower information"
    )

    shap_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Feature importance values from SHAP"
    )

    branch_notes: Optional[str] = Field(
        default=None,
        description="Unstructured branch officer remarks"
    )

    document_text: Optional[str] = Field(
        default=None,
        description="OCR text, customer documents or verification notes"
    )

    similar_cases: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Most similar historical borrower cases"
    )

    additional_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata supplied by backend"
    )