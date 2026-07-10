"""
Document Risk Agent

Evaluates borrower documents and verification notes using:
- Rule-based scoring
- Branch officer remarks
- OCR document text
- LLM reasoning
"""

from __future__ import annotations

from ai_engine.agents.base_agent import BaseAgent
from ai_engine.llm.prompts import DOCUMENT_SYSTEM_PROMPT
from ai_engine.schemas.agent_schema import AgentOutput
from ai_engine.schemas.request_schema import AIRequest

from ai_engine.utils.confidence import ConfidenceEngine
from ai_engine.utils.metadata import MetadataBuilder
from ai_engine.utils.prompt_builder import PromptBuilder
from ai_engine.utils.response_validator import ResponseValidator
from ai_engine.utils.risk_reason_builder import RiskReasonBuilder


class DocumentAgent(BaseAgent):
    """
    AI agent responsible for analysing
    borrower documents and verification notes.
    """

    def __init__(self):

        super().__init__(DOCUMENT_SYSTEM_PROMPT)

    ####################################################################
    # Rule Engine
    ####################################################################

    def calculate_document_score(
        self,
        request: AIRequest,
    ) -> int:

        score = 0

        text = (
            (request.branch_notes or "")
            + " "
            + (request.document_text or "")
        ).lower()

        if "fraud" in text:
            score += 40

        if "fake" in text:
            score += 40

        if "forged" in text:
            score += 40

        if "mismatch" in text:
            score += 25

        if "suspicious" in text:
            score += 25

        if "unverifiable" in text:
            score += 30

        if "missing" in text:
            score += 20

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
You are a Senior Banking Document Verification Specialist.

Analyze ONLY:

- Branch officer remarks
- OCR extracted document text
- Verification notes
- Fraud indicators
- Missing or inconsistent information
- Document authenticity concerns

Do NOT analyze:

- Income
- Transaction behaviour
- Repayment behaviour

Return ONLY JSON.

Required JSON format:

{
    "risk_level":"",
    "summary":"",
    "key_findings":[],
    "evidence":[],
    "recommendations":[]
}
""",
            request=request,
            rule_score=score,
        )

    ####################################################################
    # Fallback
    ####################################################################

    def build_fallback_output(
        self,
        request: AIRequest,
        score: int,
        latency: float = 0.0,
    ) -> AgentOutput:

        risk = self.rule_risk_level(score)

        return AgentOutput(

            agent_name="Document Agent",

            risk_level=risk,

            confidence=ConfidenceEngine.calculate(
                request,
                score,
            ),

            summary="Rule-based document assessment generated because AI analysis was unavailable.",

            key_findings=RiskReasonBuilder.document(
                request,
            ),

            evidence=[],

            recommendations=[
                "Perform manual document verification.",
                "Request additional supporting documents.",
                "Validate employment and address records.",
                "Escalate suspicious cases for fraud review.",
            ],

            processing_time_ms=latency,

            metadata=MetadataBuilder.build(
                request,
                score,
                risk,
            ),
        )

    ####################################################################
    # Main Analysis
    ####################################################################

    def analyze(
        self,
        request: AIRequest,
    ) -> AgentOutput:

        score = self.calculate_document_score(
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
            RiskReasonBuilder.document(
                request,
            )
        )

        metadata["agent"] = {
            "name": "Document Agent",
            "version": "1.0",
        }

        metadata["rule_engine"]["type"] = (
            "Document Rules"
        )

        return AgentOutput(

            agent_name="Document Agent",

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