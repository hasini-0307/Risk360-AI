"""
Reusable prompt builder for AI agents.
"""

from __future__ import annotations

from ai_engine.schemas.request_schema import AIRequest


class PromptBuilder:

    @staticmethod
    def build(
        title: str,
        request: AIRequest,
        rule_score: int,
    ) -> str:

        return f"""
{title}

================================

Probability of Default:
{request.default_probability:.2f}

Model Confidence:
{request.model_confidence:.2f}

Rule Score:
{rule_score}/100

Borrower Profile:

{request.borrower_profile}

Top SHAP Features:

{request.shap_values}

Similar Historical Cases:

{request.similar_cases}

Return ONLY JSON.

Expected format:

{{
    "risk_level":"",
    "summary":"",
    "key_findings":[],
    "evidence":[],
    "recommendations":[]
}}
"""