"""
Validates and normalizes LLM responses.
"""

from __future__ import annotations

from typing import Any, Dict


class ResponseValidator:

    REQUIRED_FIELDS = (
        "risk_level",
        "summary",
        "key_findings",
        "evidence",
        "recommendations",
    )

    @staticmethod
    def validate(response: Dict[str, Any]) -> Dict[str, Any]:

        validated = response.copy()

        for field in ResponseValidator.REQUIRED_FIELDS:

            if field not in validated:

                if field in (
                    "key_findings",
                    "evidence",
                    "recommendations",
                ):
                    validated[field] = []
                else:
                    validated[field] = ""

        if validated["risk_level"] not in (
            "LOW",
            "MEDIUM",
            "HIGH",
        ):
            validated["risk_level"] = "MEDIUM"

        return validated