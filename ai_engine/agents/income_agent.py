"""
Income Risk Agent

Evaluates borrower income stability and repayment capacity using:
- Rule-based scoring
- ML prediction
- SHAP explanations
- Similar historical borrowers
- LLM reasoning
"""

from __future__ import annotations

from ai_engine.agents.base_agent import BaseAgent
from ai_engine.llm.prompts import INCOME_SYSTEM_PROMPT
from ai_engine.schemas.agent_schema import AgentOutput
from ai_engine.schemas.request_schema import AIRequest

from ai_engine.utils.confidence import ConfidenceEngine
from ai_engine.utils.metadata import MetadataBuilder
from ai_engine.utils.prompt_builder import PromptBuilder
from ai_engine.utils.response_validator import ResponseValidator
from ai_engine.utils.risk_reason_builder import RiskReasonBuilder


class IncomeAgent(BaseAgent):
    """
    AI agent responsible for analysing
    borrower income stability and repayment capacity.
    """

    def __init__(self):

        super().__init__(INCOME_SYSTEM_PROMPT)

    ####################################################################
    # Rule Engine
    ####################################################################

    def calculate_income_score(
        self,
        request: AIRequest,
    ) -> int:

        profile = request.borrower_profile

        score = 0

        employment = profile.get(
            "employment_status",
            "",
        ).lower()

        if employment == "unemployed":
            score += 40

        employment_type = profile.get(
            "employment_type",
            "",
        ).lower()

        if employment_type == "temporary":
            score += 20

        dti = profile.get(
            "debt_to_income_ratio",
            0,
        )

        if dti >= 0.60:
            score += 30
        elif dti >= 0.50:
            score += 20
        elif dti >= 0.40:
            score += 10

        salary_trend = profile.get(
            "salary_trend",
            "",
        ).lower()

        if salary_trend == "declining":
            score += 15

        years = profile.get(
            "employment_years",
            0,
        )

        if years < 1:
            score += 10

        return min(score, 100)

    ####################################################################

    def rule_risk_level(
        self,
        score: int,
    ) -> str:

        if score >= 70:
            return "HIGH"

        if score >= 35:
            return "MEDIUM"

        return "LOW"

    ####################################################################

    def build_prompt(
        self,
        request: AIRequest,
        score: int,
    ) -> str:

        return PromptBuilder.build(
            title="""
You are a Senior Banking Income Risk Analyst.

Analyze ONLY:

- Income stability
- Employment stability
- Debt-to-income ratio
- Salary trend
- Repayment capacity

Do NOT analyze:

- Transactions
- Behaviour
- Documents

Return ONLY JSON.
""",
            request=request,
            rule_score=score,
        )

    ####################################################################

    def build_fallback_output(
        self,
        request: AIRequest,
        score: int,
        latency: float = 0.0,
    ) -> AgentOutput:

        risk = self.rule_risk_level(score)

        return AgentOutput(

            agent_name="Income Agent",

            risk_level=risk,

            confidence=ConfidenceEngine.calculate(
                request,
                score,
            ),

            summary="Rule-based income assessment generated because AI analysis was unavailable.",

            key_findings=RiskReasonBuilder.income(
                request,
            ),

            evidence=[],

            recommendations=[
                "Verify employment records.",
                "Review recent salary statements.",
                "Assess debt-to-income ratio before approval.",
            ],

            processing_time_ms=latency,

            metadata=MetadataBuilder.build(
                request,
                score,
                risk,
            ),
        )

    ####################################################################

    def analyze(
        self,
        request: AIRequest,
    ) -> AgentOutput:

        score = self.calculate_income_score(
            request,
        )

        risk = self.rule_risk_level(score)

        prompt = self.build_prompt(
            request,
            score,
        )

        try:

            response, latency = self.call_llm(
                prompt,
            )

            response = ResponseValidator.validate(
                response,
            )

        except Exception:

            return self.build_fallback_output(
                request=request,
                score=score,
            )

        confidence = ConfidenceEngine.calculate(
            request,
            score,
        )

        metadata = MetadataBuilder.build(
            request,
            score,
            risk,
        )

        metadata["risk_reasons"] = (
            RiskReasonBuilder.income(
                request,
            )
        )

        metadata["agent"] = {
            "name": "Income Agent",
            "version": "1.0",
        }

        metadata["rule_engine"]["type"] = (
            "Income Rules"
        )

        return AgentOutput(

            agent_name="Income Agent",

            risk_level=self.resolve_final_risk(
                risk,
                response["risk_level"],
            ),

            confidence=confidence,

            summary=response["summary"],

            key_findings=response["key_findings"],

            evidence=response["evidence"],

            recommendations=response["recommendations"],

            processing_time_ms=latency,

            metadata=metadata,
        )