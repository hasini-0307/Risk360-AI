"""
Module: exceptions.py

Custom exceptions used throughout the AI Engine.
"""


class LLMError(Exception):
    """Base exception for all LLM-related errors."""


class ConfigurationError(LLMError):
    """Raised when the LLM configuration is invalid."""


class APIConnectionError(LLMError):
    """Raised when the AI provider cannot be reached."""


class InvalidResponseError(LLMError):
    """Raised when the AI provider returns an invalid response."""


class RateLimitError(LLMError):
    """Raised when the API rate limit is exceeded."""


class AuthenticationError(LLMError):
    """Raised when authentication with the provider fails."""