"""
Guardrails Module for Atlas-RAG — powered by guardrails-ai
Input and output validation using Guard objects with custom validators.
Public API is unchanged: check_input() / check_output() / get_guardrails().
"""
import logging
from typing import Optional
from enum import Enum
from dataclasses import dataclass

from guardrails import Guard
from guardrails.validator_base import OnFailAction

from .validators import (
    NoJailbreak,
    OnTopic,
    NoHallucinationPhrases,
    MinAnswerLength,
    MAX_INPUT_CHARS,
)

logger = logging.getLogger(__name__)


class BlockReason(Enum):
    JAILBREAK = "jailbreak_attempt"
    TOXICITY = "toxic_content"
    OFF_TOPIC = "off_topic"
    LOW_CONFIDENCE = "low_retrieval_confidence"
    HALLUCINATION = "potential_hallucination"
    PII_DETECTED = "pii_detected"
    SCHEMA_INVALID = "invalid_output_schema"


@dataclass
class GuardrailResult:
    passed: bool
    blocked_reason: Optional[BlockReason] = None
    message: Optional[str] = None
    confidence: float = 1.0
    fixed_value: Optional[str] = None

    @property
    def should_block(self) -> bool:
        return not self.passed


class InputGuardrails:
    """
    Validates user questions before RAG processing.
    Guard: NoJailbreak → OnTopic (both EXCEPTION on fail → caught below).
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._guard = Guard().use_many(
            NoJailbreak(on_fail=OnFailAction.EXCEPTION),
            OnTopic(on_fail=OnFailAction.EXCEPTION),
        )
        logger.info("InputGuardrails ready (guardrails-ai)")

    def check(self, user_input: str) -> GuardrailResult:
        if not self.enabled:
            return GuardrailResult(passed=True)

        try:
            outcome = self._guard.validate(user_input)
            if outcome.validation_passed:
                return GuardrailResult(passed=True)
            # validation_passed=False but no exception (NOOP validators) — treat as blocked
            error_msg = outcome.error or "Validation échouée"
            return self._map_input_error(error_msg)
        except Exception as exc:
            return self._map_input_error(str(exc))

    def _map_input_error(self, msg: str) -> GuardrailResult:
        msg_lower = msg.lower()
        if "jailbreak" in msg_lower:
            return GuardrailResult(
                passed=False,
                blocked_reason=BlockReason.JAILBREAK,
                message="Je peux uniquement vous aider avec les questions relatives au portail ACAPS.",
            )
        if "hors sujet" in msg_lower or "on-topic" in msg_lower or "off_topic" in msg_lower:
            return GuardrailResult(
                passed=False,
                blocked_reason=BlockReason.OFF_TOPIC,
                message="Je suis conçu pour assister avec l'utilisation du portail ACAPS uniquement.",
            )
        logger.warning("InputGuardrails blocked (unmapped reason): %s", msg[:120])
        return GuardrailResult(
            passed=False,
            blocked_reason=BlockReason.OFF_TOPIC,
            message="Je ne peux pas traiter cette demande.",
        )


class OutputGuardrails:
    """
    Validates LLM answers before returning to the user.
    Guard: MinAnswerLength (NOOP) → NoHallucinationPhrases (NOOP).
    NOOP means we inspect outcome.validation_passed without raising.
    """

    def __init__(self, enabled: bool = True, confidence_threshold: float = 0.40):
        self.enabled = enabled
        self.confidence_threshold = confidence_threshold
        self._guard = Guard().use_many(
            MinAnswerLength(min_chars=30, on_fail=OnFailAction.NOOP),
            NoHallucinationPhrases(on_fail=OnFailAction.NOOP),
        )
        logger.info(
            "OutputGuardrails ready (guardrails-ai, threshold=%.2f)", confidence_threshold
        )

    def check(
        self,
        response: str,
        context: str,
        retrieval_score: float,
        section_known: bool = False,
    ) -> GuardrailResult:
        if not self.enabled:
            return GuardrailResult(passed=True)

        # Low retrieval confidence — block before even checking the answer.
        # Skip this check when a section filter was active: the section is
        # guaranteed correct, so a slightly-below-threshold score is still valid.
        if not section_known and 0 < retrieval_score < self.confidence_threshold:
            logger.info("Low retrieval score: %.3f", retrieval_score)
            return GuardrailResult(
                passed=False,
                blocked_reason=BlockReason.LOW_CONFIDENCE,
                message="Je ne trouve pas d'information pertinente dans les documents disponibles.",
                confidence=retrieval_score,
            )

        # Schema + hallucination check via guardrails-ai
        # context passed as metadata — available to validators for grounding checks
        try:
            outcome = self._guard.validate(response, metadata={"context": context[:MAX_INPUT_CHARS]})
        except Exception as exc:
            logger.warning("OutputGuardrails exception: %s", str(exc)[:200])
            return GuardrailResult(
                passed=False,
                blocked_reason=BlockReason.SCHEMA_INVALID,
                message="La réponse générée ne respecte pas les critères de qualité.",
                confidence=retrieval_score,
            )

        if not outcome.validation_passed:
            error_msg = outcome.error or ""
            logger.warning("OutputGuardrails failed: %s", error_msg[:120])

            if "incertitude" in error_msg.lower() or "hallucin" in error_msg.lower():
                return GuardrailResult(
                    passed=False,
                    blocked_reason=BlockReason.HALLUCINATION,
                    message="Je ne peux pas fournir une réponse vérifiée sur base des documents disponibles.",
                    confidence=retrieval_score,
                )
            if "trop courte" in error_msg.lower():
                return GuardrailResult(
                    passed=False,
                    blocked_reason=BlockReason.SCHEMA_INVALID,
                    message="La réponse générée est insuffisante. Veuillez reformuler votre question.",
                    confidence=retrieval_score,
                )
            return GuardrailResult(
                passed=False,
                blocked_reason=BlockReason.SCHEMA_INVALID,
                message="La réponse ne respecte pas les critères de qualité attendus.",
                confidence=retrieval_score,
            )

        validated_answer = outcome.validated_output if outcome.validated_output else response
        return GuardrailResult(
            passed=True,
            confidence=retrieval_score,
            fixed_value=validated_answer,
        )


class Guardrails:
    """Main orchestrator: composes InputGuardrails + OutputGuardrails."""

    def __init__(self, enabled: bool = True, confidence_threshold: float = 0.40):
        self.enabled = enabled
        self.input_rails = InputGuardrails(enabled=enabled)
        self.output_rails = OutputGuardrails(
            enabled=enabled, confidence_threshold=confidence_threshold
        )
        logger.info(
            "Guardrails initialized (enabled=%s, threshold=%.2f)", enabled, confidence_threshold
        )

    def check_input(self, user_input: str) -> GuardrailResult:
        return self.input_rails.check(user_input)

    def check_output(
        self,
        response: str,
        context: str,
        retrieval_score: float,
        section_known: bool = False,
    ) -> GuardrailResult:
        return self.output_rails.check(response, context, retrieval_score, section_known)

    def get_blocked_response(self, reason: BlockReason) -> str:
        responses = {
            BlockReason.JAILBREAK: (
                "Je peux uniquement vous aider avec les questions relatives au portail ACAPS."
            ),
            BlockReason.TOXICITY: (
                "Veuillez reformuler votre question de manière respectueuse."
            ),
            BlockReason.OFF_TOPIC: (
                "Je suis conçu pour assister avec l'utilisation du portail ACAPS uniquement."
            ),
            BlockReason.LOW_CONFIDENCE: (
                "Je ne trouve pas cette information dans les documents disponibles."
            ),
            BlockReason.HALLUCINATION: (
                "Je ne peux pas fournir une réponse vérifiée sur base des documents disponibles."
            ),
            BlockReason.PII_DETECTED: (
                "Je ne peux pas traiter des demandes contenant des données personnelles."
            ),
            BlockReason.SCHEMA_INVALID: (
                "La réponse générée ne respecte pas les critères de qualité."
            ),
        }
        return responses.get(reason, "Je ne peux pas traiter cette demande.")


_guardrails_instance: Optional[Guardrails] = None


def get_guardrails(enabled: bool = True, confidence_threshold: float = 0.40) -> Guardrails:
    global _guardrails_instance
    if _guardrails_instance is None:
        _guardrails_instance = Guardrails(
            enabled=enabled, confidence_threshold=confidence_threshold
        )
    return _guardrails_instance
