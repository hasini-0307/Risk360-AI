"""
Chief Risk Officer

Combines all specialist AI agents into one
enterprise risk decision.
"""

from __future__ import annotations

from typing import List

from ai_engine.cro.approval_policy import ApprovalPolicy
from ai_engine.cro.conflict_detector import ConflictDetector
from ai_engine.cro.risk_fusion import RiskFusion

from ai_engine.llm.llm_client import LLMClient
from ai_engine.llm.prompts import CHIEF_RISK_OFFICER_SYSTEM_PROMPT
from ai_engine.llm.types import LLMRequest

from ai_engine.schemas.agent_schema import AgentOutput
from ai_engine.schemas.cro_schema import CROOutput
from ai_engine.schemas.request_schema import AIRequest

from ai_engine.utils.json_parser import parse_llm_json
from ai_engine.utils.timer import Timer
from ai_engine.utils.logger import get_logger


logger = get_logger(__name__)


class ChiefRiskOfficer:
    """
    Final decision maker.

    Aggregates all specialist AI agents.
    """

    def __init__(self):

        self.llm = LLMClient()

    ####################################################################
    # Prompt Builder
    ####################################################################

    def build_prompt(
        self,
        request: AIRequest,
        agents: List[AgentOutput],
        overall_risk: str,
        conflicts: List[str],
        approval: str,
    ) -> str:

        return f"""
Generate the executive decision.

ML Probability:

{request.default_probability:.2f}

Model Confidence:

{request.model_confidence:.2f}

Overall Risk:

{overall_risk}

Approval Recommendation:

{approval}

Conflicts:

{conflicts}

Specialist Assessments:

{[a.model_dump() for a in agents]}

Return ONLY JSON.

{{
"executive_summary":"",
"top_risk_drivers":[],
"recommended_actions":[]
}}
"""

    ####################################################################
    # Main
    ####################################################################

    def evaluate(
        self,
        request: AIRequest,
        agents: List[AgentOutput],
    ) -> CROOutput:

        with Timer() as timer:

            overall_risk = RiskFusion.overall_risk(
                request.loan_type,
                agents,
            )

            conflicts = ConflictDetector.detect(
                agents,
            )

            approval = ApprovalPolicy.recommend(
                overall_risk,
                request.default_probability,
                agents,
            )

            average_confidence = sum(
                a.confidence for a in agents
            ) / len(agents)

            prompt = self.build_prompt(
                request,
                agents,
                overall_risk,
                conflicts,
                approval,
            )

            try:

                response = self.llm.generate(

                    LLMRequest(

                        system_prompt=CHIEF_RISK_OFFICER_SYSTEM_PROMPT,

                        user_prompt=prompt,

                        temperature=0.1,
                    )
                )

                parsed = parse_llm_json(
                    response.text
                )

            except Exception as exc:

                logger.exception(exc)

                parsed = {

                    "executive_summary":

                    "Final assessment generated using deterministic banking rules because AI summarization was unavailable.",

                    "top_risk_drivers":[
                        "See specialist agent findings."
                    ],

                    "recommended_actions":[
                        "Perform manual credit review."
                    ]
                }

        return CROOutput(

            overall_risk=overall_risk,

            confidence=round(
                average_confidence,
                2,
            ),

            executive_summary=parsed[
                "executive_summary"
            ],

            top_risk_drivers=parsed[
                "top_risk_drivers"
            ],

            conflicting_assessments=conflicts,

            recommended_actions=parsed[
                "recommended_actions"
            ],

            approval_recommendation=approval,

            processing_time_ms=timer.elapsed_ms,

            metadata={

                "agents_used": len(agents),

                "fusion_engine": "Loan-Type Weighted Confidence",

                "approval_policy": approval,

                "loan_type": request.loan_type,

                "high_risk_agents": sum(
                    1 for a in agents if a.risk_level == "HIGH"
                ),

                "medium_risk_agents": sum(
                    1 for a in agents if a.risk_level == "MEDIUM"
                ),

                "low_risk_agents": sum(
                    1 for a in agents if a.risk_level == "LOW"
                ),

                "average_default_probability":
                    request.default_probability,
            },
        )