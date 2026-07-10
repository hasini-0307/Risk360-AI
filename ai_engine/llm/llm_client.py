"""
Module: llm_client.py

Enterprise-grade LLM Gateway for the AI Engine.

Responsibilities:
- Connect to the configured LLM provider
- Handle authentication
- Measure latency
- Return structured responses
- Handle provider errors
"""

from __future__ import annotations

import logging
import time

from groq import Groq

from .config import LLMConfig
from .exceptions import (
    APIConnectionError,
    AuthenticationError,
    ConfigurationError,
    InvalidResponseError,
)
from .types import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Central gateway for all LLM interactions.
    """

    def __init__(self) -> None:

        if not LLMConfig.API_KEY:
            raise ConfigurationError(
                "GROQ_API_KEY not found in environment variables."
            )

        self.client = Groq(api_key=LLMConfig.API_KEY)

        logger.info(
            "Initialized LLM Client with model: %s",
            LLMConfig.MODEL_NAME,
        )

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Generate a response from the configured LLM.
        """

        start = time.perf_counter()

        try:

            response = self.client.chat.completions.create(
                model=LLMConfig.MODEL_NAME,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": request.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": request.user_prompt,
                    },
                ],
            )

        except Exception as exc:

            error = str(exc).lower()

            if "authentication" in error:
                raise AuthenticationError(str(exc))

            if "connection" in error:
                raise APIConnectionError(str(exc))

            raise APIConnectionError(str(exc))

        latency = (time.perf_counter() - start) * 1000

        if not response.choices:
            raise InvalidResponseError(
                "LLM returned no choices."
            )

        message = response.choices[0].message.content

        usage = response.usage

        logger.info(
            "LLM response generated in %.2f ms",
            latency,
        )

        return LLMResponse(
            text=message,
            model=response.model,
            latency_ms=latency,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            metadata={
                "provider": LLMConfig.PROVIDER,
            },
        )