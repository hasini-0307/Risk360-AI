"""


Defines the request and response objects exchanged
between the AI Engine and the language model.
"""

from typing import Dict, Optional

from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    """
    Standard request sent to the language model.
    """

    system_prompt: str = Field(
        ...,
        description="Instructions defining the AI assistant's behavior."
    )

    user_prompt: str = Field(
        ...,
        description="The actual task or input provided to the model."
    )

    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature."
    )

    max_tokens: int = Field(
        default=800,
        gt=0,
        description="Maximum number of tokens to generate."
    )


class LLMResponse(BaseModel):
    """
    Standard response returned by the language model.
    """

    text: str = Field(
        ...,
        description="Generated response."
    )

    model: str = Field(
        ...,
        description="Model used to generate the response."
    )

    latency_ms: float = Field(
        ...,
        description="Inference latency in milliseconds."
    )

    prompt_tokens: Optional[int] = Field(
        default=None,
        description="Prompt token count."
    )

    completion_tokens: Optional[int] = Field(
        default=None,
        description="Completion token count."
    )

    total_tokens: Optional[int] = Field(
        default=None,
        description="Total token usage."
    )

    metadata: Dict = Field(
        default_factory=dict,
        description="Additional provider-specific metadata."
    )