"""
Behavior Risk Agent

Analyzes borrower repayment behaviour using:
- Rule-based scoring
- ML prediction
- SHAP explanations
- Similar historical borrowers
- LLM reasoning
"""

from __future__ import annotations

from ai_engine.agents.base_agent import BaseAgent
from ai_engine.llm.prompts import BEHAVIOR_SYSTEM_PROMPT
from ai_engine.schemas.agent_schema import AgentOutput
from ai_engine.schemas.request_schema import AIRequest

from ai_engine.utils.confidence import ConfidenceEngine
from ai_engine.utils.metadata import MetadataBuilder
from ai_engine.utils.prompt_builder import PromptBuilder
from ai_engine.utils.response_validator import ResponseValidator
from ai_engine.utils.risk_reason_builder import RiskReasonBuilder

class BehaviorAgent(BaseAgent):
    """
    Specialized AI agent responsible for analysing
    borrower repayment behaviour.
    """

    def __init__(self):

        super().__init__(BEHAVIOR_SYSTEM_PROMPT)

    ####################################################################
    # Rule Engine
    ####################################################################


    
    
    def build_fallback_output(
    self,
    request: AIRequest,
    score: int,
    latency: float = 0.0,
) -> AgentOutput:

        risk = self.rule_risk_level(score)

        confidence = ConfidenceEngine.calculate(
            request,
            score,
        )

        return AgentOutput(

            agent_name="Behavior Agent",

            risk_level=risk,

            confidence=confidence,

            summary="Rule-based behaviour assessment used because AI analysis was unavailable.",

            key_findings=RiskReasonBuilder.behavior(request),

            evidence=[],

            recommendations=[
                "Review repayment history manually."
            ],

            processing_time_ms=latency,

            metadata=MetadataBuilder.build(
                request,
                score,
                risk,
            )
        )
    
    
    def calculate_behavior_score(
        self,
        request: AIRequest,
    ) -> int:

        profile = request.borrower_profile

        score = 0

        missed = profile.get("missed_payments", 0)

        if missed >= 6:
            score += 40
        elif missed >= 3:
            score += 25
        elif missed >= 1:
            score += 10

        dpd = profile.get("days_past_due", 0)

        if dpd >= 90:
            score += 30
        elif dpd >= 30:
            score += 20
        elif dpd >= 1:
            score += 10

        defaults = profile.get("previous_defaults", 0)
        score += defaults * 15

        restructures = profile.get("loan_restructures", 0)
        score += restructures * 10

        return min(score, 100)

    ####################################################################
    # Rule Risk Mapping
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
    # Prompt Builder
    ####################################################################

    def build_prompt(
        self,
        request: AIRequest,
        score: int,
    ) -> str:

        return PromptBuilder.build(
            title="""
    You are evaluating ONLY borrower repayment behaviour.

    Focus only on:
    - missed payments
    - repayment consistency
    - delinquency
    - restructuring history
    - previous defaults

    Do NOT evaluate income, transactions or documents.
    """,
            request=request,
            rule_score=score,
        )


    ####################################################################
    # Main Analysis
    ####################################################################

    def analyze(self, request: AIRequest) -> AgentOutput:

        score = self.calculate_behavior_score(request)

        risk = self.rule_risk_level(score)

        prompt = self.build_prompt(
            request,
            score,
        )

        try:

            response, latency = self.call_llm(prompt)
            response = ResponseValidator.validate(response)

        except Exception:

            return self.build_fallback_output(
                request = request,
                score = score,
                latency=0.0,
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

        metadata["risk_reasons"] = RiskReasonBuilder.behavior(request)

        metadata["agent"] = {
            "name": "Behavior Agent",
            "version": "1.0",
        }

        metadata["rule_engine"]["type"] = "Behavior Rules"

        

        metadata["risk_reasons"] = RiskReasonBuilder.behavior(
        request
    )

        return AgentOutput(

            agent_name="Behavior Agent",

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