"""


Centralized configuration for all LLM providers.
Reads configuration from environment variables.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class LLMConfig:
    """
    Configuration settings for the AI Model Gateway.
    """

    # Provider
    PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")

    # Model
    MODEL_NAME: str = os.getenv(
        "LLM_MODEL",
        "llama-3.3-70b-versatile"
    )

    # Authentication
    API_KEY: str | None = os.getenv("GROQ_API_KEY")

    # Generation Parameters
    TEMPERATURE: float = float(
        os.getenv("LLM_TEMPERATURE", "0.0")
    )

    MAX_TOKENS: int = int(
        os.getenv("LLM_MAX_TOKENS", "400")
    )

    TOP_P: float = float(
        os.getenv("LLM_TOP_P", "0.95")
    )

    # Timeout
    TIMEOUT: int = int(
        os.getenv("LLM_TIMEOUT", "30")
    )