"""
Transaction Risk Agent

Evaluates borrower transaction behaviour using:
- Rule-based scoring
- ML prediction
- SHAP explanations
- Similar historical borrowers
- LLM reasoning
"""

from __future__ import annotations

from ai_engine.agents.base_agent import BaseAgent
from ai_engine.llm.prompts import TRANSACTION_SYSTEM_PROMPT
from ai_engine.schemas.agent_schema import AgentOutput
from ai_engine.schemas.request_schema import AIRequest

from ai_engine.utils.confidence import ConfidenceEngine
from ai_engine.utils.metadata import MetadataBuilder
from ai_engine.utils.prompt_builder import PromptBuilder
from ai_engine.utils.response_validator import ResponseValidator
from ai_engine.utils.risk_reason_builder import RiskReasonBuilder


class TransactionAgent(BaseAgent):
    """
    AI agent responsible for analysing
    borrower transaction behaviour and financial discipline.
    """

    def __init__(self):

        super().__init__(TRANSACTION_SYSTEM_PROMPT)

    ####################################################################
    # Rule Engine
    ####################################################################

    def calculate_transaction_score(
        self,
        request: AIRequest,
    ) -> int:

        profile = request.borrower_profile

        score = 0

        utilization = profile.get(
            "credit_utilization",
            0,
        )

        if utilization >= 0.90:
            score += 35
        elif utilization >= 0.80:
            score += 25
        elif utilization >= 0.60:
            score += 15

        average_balance = profile.get(
            "average_balance",
            999999,
        )

        if average_balance < 5000:
            score += 20

        cash_withdrawals = profile.get(
            "cash_withdrawals",
            0,
        )

        if cash_withdrawals >= 10:
            score += 15

        large_transactions = profile.get(
            "large_transactions",
            0,
        )

        if large_transactions >= 3:
            score += 15

        emi_bounces = profile.get(
            "emi_bounces",
            0,
        )

        if emi_bounces > 0:
            score += 25

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
You are a Senior Banking Transaction Risk Analyst.

Analyze ONLY:

- Credit utilization
- Spending behaviour
- Cash withdrawals
- Average account balance
- Financial discipline
- Transaction anomalies
- EMI bounce patterns

Do NOT analyze:

- Income
- Repayment behaviour
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

            agent_name="Transaction Agent",

            risk_level=risk,

            confidence=ConfidenceEngine.calculate(
                request,
                score,
            ),

            summary="Rule-based transaction assessment generated because AI analysis was unavailable.",

            key_findings=RiskReasonBuilder.transaction(
                request,
            ),

            evidence=[],

            recommendations=[
                "Review recent account transactions.",
                "Monitor credit utilization closely.",
                "Verify unusually large transactions.",
                "Investigate repeated EMI payment failures.",
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

        score = self.calculate_transaction_score(
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
            RiskReasonBuilder.transaction(
                request,
            )
        )

        metadata["agent"] = {
            "name": "Transaction Agent",
            "version": "1.0",
        }

        metadata["rule_engine"]["type"] = (
            "Transaction Rules"
        )

        return AgentOutput(

            agent_name="Transaction Agent",

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