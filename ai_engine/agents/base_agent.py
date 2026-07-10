"""
Enterprise Base Agent for all Risk360 AI Agents.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from ai_engine.llm.llm_client import LLMClient
from ai_engine.llm.types import LLMRequest
from ai_engine.schemas.agent_schema import AgentOutput
from ai_engine.schemas.request_schema import AIRequest
from ai_engine.utils.json_parser import parse_llm_json
from ai_engine.utils.timer import Timer

logger = logging.getLogger(__name__)


class BaseAgent(ABC):

    MAX_RETRIES = 2

    def __init__(self, system_prompt: str):

        self.system_prompt = system_prompt
        self.llm = LLMClient()

    @abstractmethod
    def analyze(
        self,
        request: AIRequest,
    ) -> AgentOutput:
        """
        Every specialized agent implements this.
        """
        raise NotImplementedError

    ###########################################################
    # Internal Helpers
    ###########################################################
    
    def resolve_final_risk(
        self,
        rule_risk: str,
        llm_risk: str,
    ) -> str:
        """
        Choose the higher risk level between
        the rule engine and the LLM.
        """

        priority = {
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
        }

        return max(
            rule_risk,
            llm_risk,
            key=lambda r: priority[r],
        )
    
    
    def _validate_response(
        self,
        response: Dict[str, Any],
    ) -> Dict[str, Any]:

        required = [
            "risk_level",
            "summary",
            "key_findings",
            "evidence",
            "recommendations",
        ]

        for key in required:
            response.setdefault(key, [] if key in [
                "key_findings",
                "evidence",
                "recommendations",
            ] else "")

        if response["risk_level"] not in (
            "LOW",
            "MEDIUM",
            "HIGH",
        ):
            response["risk_level"] = "MEDIUM"

        return response

    ###########################################################

    def call_llm(
        self,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> Tuple[Dict[str, Any], float]:

        last_exception = None

        for attempt in range(self.MAX_RETRIES):

            try:

                with Timer() as timer:

                    response = self.llm.generate(
                        LLMRequest(
                            system_prompt=self.system_prompt,
                            user_prompt=user_prompt,
                            temperature=temperature,
                        )
                    )

                parsed = parse_llm_json(response.text)

                parsed = self._validate_response(parsed)

                logger.info(
                    "%s succeeded (Attempt %d)",
                    self.__class__.__name__,
                    attempt + 1,
                )

                return parsed, timer.elapsed_ms

            except Exception as exc:

                last_exception = exc

                logger.exception(
                    "%s failed (Attempt %d)",
                    self.__class__.__name__,
                    attempt + 1,
                )

        raise RuntimeError(
            f"{self.__class__.__name__} failed after "
            f"{self.MAX_RETRIES} attempts."
        ) from last_exception