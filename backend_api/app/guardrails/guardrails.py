"""
Guardrails Module for Atlas-RAG.
Implements input/output validation and hallucination prevention.

All regex patterns and blocked-response messages are loaded from
backend_api/config/guardrails_config.json — edit without rebuilding.
"""
import json
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Load guardrail config (hot-number principle — same as engine.py)
# ──────────────────────────────────────────────────────────────────────────────
_CONFIG_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config")
)
_GR_CFG_PATH = os.path.join(_CONFIG_DIR, "guardrails_config.json")

with open(_GR_CFG_PATH, encoding="utf-8") as _fh:
    _GR_CFG: dict = json.load(_fh)


class RailType(Enum):
    INPUT = "input"
    OUTPUT = "output"
    FACT_CHECK = "fact_check"


class BlockReason(Enum):
    JAILBREAK = "jailbreak_attempt"
    TOXICITY = "toxic_content"
    OFF_TOPIC = "off_topic"
    LOW_CONFIDENCE = "low_retrieval_confidence"
    HALLUCINATION = "potential_hallucination"
    PII_DETECTED = "pii_detected"


@dataclass
class GuardrailResult:
    passed: bool
    blocked_reason: Optional[BlockReason] = None
    message: Optional[str] = None
    confidence: float = 1.0

    @property
    def should_block(self) -> bool:
        return not self.passed


class InputGuardrails:
    """Input validation: jailbreaks, toxicity, off-topic queries."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.jailbreak_re = [
            re.compile(p, re.IGNORECASE) for p in _GR_CFG["jailbreak_patterns"]
        ]
        self.off_topic_re = [
            re.compile(p, re.IGNORECASE) for p in _GR_CFG["off_topic_patterns"]
        ]
        self.toxic_re = [
            re.compile(p, re.IGNORECASE) for p in _GR_CFG["toxic_patterns"]
        ]

    def check(self, user_input: str) -> GuardrailResult:
        if not self.enabled:
            return GuardrailResult(passed=True)

        result = self._check_jailbreak(user_input)
        if result.should_block:
            logger.warning("Jailbreak attempt detected: %s", user_input[:100])
            return result

        result = self._check_toxicity(user_input)
        if result.should_block:
            logger.warning("Toxic content detected: %s", user_input[:100])
            return result

        result = self._check_off_topic(user_input)
        if result.should_block:
            logger.info("Off-topic query detected: %s", user_input[:100])
            return result

        return GuardrailResult(passed=True)

    def _check_jailbreak(self, text: str) -> GuardrailResult:
        for pattern in self.jailbreak_re:
            if pattern.search(text):
                return GuardrailResult(
                    passed=False,
                    blocked_reason=BlockReason.JAILBREAK,
                    message=_GR_CFG["blocked_responses"]["jailbreak_fr"],
                )
        return GuardrailResult(passed=True)

    def _check_toxicity(self, text: str) -> GuardrailResult:
        for pattern in self.toxic_re:
            if pattern.search(text):
                return GuardrailResult(
                    passed=False,
                    blocked_reason=BlockReason.TOXICITY,
                    message=_GR_CFG["blocked_responses"]["toxicity_fr"],
                )
        return GuardrailResult(passed=True)

    def _check_off_topic(self, text: str) -> GuardrailResult:
        for pattern in self.off_topic_re:
            if pattern.search(text):
                is_arabic = bool(re.search(r'[؀-ۿ]', text))
                key = "off_topic_ar" if is_arabic else "off_topic_fr"
                return GuardrailResult(
                    passed=False,
                    blocked_reason=BlockReason.OFF_TOPIC,
                    message=_GR_CFG["blocked_responses"][key],
                )
        return GuardrailResult(passed=True)


class OutputGuardrails:
    """Output validation: hallucination phrases, confidence threshold."""

    def __init__(self, enabled: bool = True, confidence_threshold: float = 0.75):
        self.enabled = enabled
        self.confidence_threshold = confidence_threshold
        self.hallucination_re = [
            re.compile(p, re.IGNORECASE)
            for p in _GR_CFG["hallucination_indicators"]
        ]

    def check(self, response: str, context: str, retrieval_score: float) -> GuardrailResult:
        if not self.enabled:
            return GuardrailResult(passed=True)

        if retrieval_score < self.confidence_threshold:
            logger.info("Low retrieval score: %.3f", retrieval_score)
            return GuardrailResult(
                passed=False,
                blocked_reason=BlockReason.LOW_CONFIDENCE,
                message=_GR_CFG["blocked_responses"]["low_confidence_fr"],
                confidence=retrieval_score,
            )

        for pattern in self.hallucination_re:
            if pattern.search(response):
                return GuardrailResult(
                    passed=False,
                    blocked_reason=BlockReason.HALLUCINATION,
                    message=_GR_CFG["blocked_responses"]["hallucination_fr"],
                )

        return GuardrailResult(passed=True, confidence=retrieval_score)


class Guardrails:
    """Orchestrates input and output validation."""

    def __init__(self, enabled: bool = True, confidence_threshold: float = 0.75):
        self.enabled = enabled
        self.input_rails = InputGuardrails(enabled=enabled)
        self.output_rails = OutputGuardrails(
            enabled=enabled, confidence_threshold=confidence_threshold
        )
        logger.info("Guardrails initialized (enabled=%s, threshold=%s)", enabled, confidence_threshold)

    def check_input(self, user_input: str) -> GuardrailResult:
        return self.input_rails.check(user_input)

    def check_output(self, response: str, context: str, retrieval_score: float) -> GuardrailResult:
        return self.output_rails.check(response, context, retrieval_score)

    def get_blocked_response(self, reason: BlockReason) -> str:
        mapping = {
            BlockReason.JAILBREAK: "jailbreak_fr",
            BlockReason.TOXICITY: "toxicity_fr",
            BlockReason.OFF_TOPIC: "off_topic_fr",
            BlockReason.LOW_CONFIDENCE: "low_confidence_fr",
            BlockReason.HALLUCINATION: "hallucination_fr",
            BlockReason.PII_DETECTED: "pii_detected_fr",
        }
        key = mapping.get(reason)
        return _GR_CFG["blocked_responses"].get(key, "Je ne peux pas traiter cette demande.")


# ── Singleton (re-created when settings differ) ───────────────────────────────

_guardrails_instance: Optional[Guardrails] = None
_guardrails_params: tuple = (None, None)


def get_guardrails(enabled: bool = True, confidence_threshold: float = 0.75) -> Guardrails:
    global _guardrails_instance, _guardrails_params
    params = (enabled, confidence_threshold)
    if _guardrails_instance is None or _guardrails_params != params:
        _guardrails_instance = Guardrails(
            enabled=enabled, confidence_threshold=confidence_threshold
        )
        _guardrails_params = params
    return _guardrails_instance
