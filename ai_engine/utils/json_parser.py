"""
Utility functions for safely parsing LLM JSON responses.
"""

from __future__ import annotations

import json
import re
from typing import Any


def clean_json_response(response: str) -> str:
    """
    Removes markdown code fences and surrounding whitespace.
    """

    response = response.strip()

    # Remove opening code fence
    response = re.sub(
        r"^```(?:json)?\s*",
        "",
        response,
        flags=re.IGNORECASE,
    )

    # Remove closing code fence
    response = re.sub(
        r"\s*```$",
        "",
        response,
    )

    return response.strip()


def parse_llm_json(response: str) -> dict[str, Any]:
    """
    Safely parse JSON returned by an LLM.
    """

    cleaned = clean_json_response(response)

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON returned by the LLM:\n\n{cleaned}"
        ) from exc