"""
Custom Guardrails-AI validators for the ACAPS chatbot.
Each validator focuses on one rule; composed in guardrails.py via Guard.
"""
import re
from guardrails.validator_base import (
    Validator,
    register_validator,
    ValidationResult,
    PassResult,
    FailResult,
    OnFailAction,
)

# Named constants — no magic numbers
MIN_ANSWER_CHARS = 30
MAX_INPUT_CHARS = 2_000


@register_validator(name="acaps/no-jailbreak", data_type="string")
class NoJailbreak(Validator):
    """Blocks prompt-injection / jailbreak attempts in user input."""

    PATTERNS = [
        r"ignore.*(?:previous|your).*instructions",
        r"forget.*(?:previous|your).*instructions",
        r"pretend.*(?:you are|to be)",
        r"act as if",
        r"you are now",
        r"new persona",
        r"bypass.*(?:filters|restrictions)",
        r"\bDAN\b.*mode",
        r"developer.*mode",
        r"system.*prompt",
        r"reveal.*prompt",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.PATTERNS]

    def validate(self, value: str, metadata: dict) -> ValidationResult:
        safe_input = value[:MAX_INPUT_CHARS]
        for pattern in self._compiled:
            if pattern.search(safe_input):
                return FailResult(error_message="Tentative de jailbreak détectée")
        return PassResult()


@register_validator(name="acaps/on-topic", data_type="string")
class OnTopic(Validator):
    """Rejects queries clearly outside the ACAPS portal scope."""

    PATTERNS = [
        r"(?:what(?:'s| is)|quel(?:le)? est|c'est quoi).*(?:weather|météo|temps qu'il fait)",
        r"(?:tell|write|écris?|raconte).*(?:joke|blague|poème|poem|histoire pour enfant)",
        r"(?:who won|score of|résultat du?).*(?:game|match|jeu)",
        r"(?:latest|recent|dernières?).*(?:news|actualités|nouvelles)",
        r"(?:stock|crypto|bourse).*(?:price|prix|cours)",
        r"\b(?:recipe|recette)\b",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.PATTERNS]

    def validate(self, value: str, metadata: dict) -> ValidationResult:
        safe_input = value[:MAX_INPUT_CHARS]
        for pattern in self._compiled:
            if pattern.search(safe_input):
                return FailResult(error_message="Question hors sujet du portail ACAPS")
        return PassResult()


@register_validator(name="acaps/no-hallucination-phrases", data_type="string")
class NoHallucinationPhrases(Validator):
    """
    Rejects LLM answers that contain uncertainty/speculation markers.
    These phrases signal the model is guessing rather than citing the guide.
    """

    PHRASES = [
        # French
        "je pense que", "je crois que", "probablement", "il est possible que",
        "généralement", "je suppose", "à ma connaissance",
        "je ne suis pas certain", "il me semble", "d'habitude",
        "en général", "normalement", "peut-être que",
        # English (model sometimes slips)
        "i think", "i believe", "probably", "might be", "could be",
        "as far as i know", "in my opinion", "generally speaking",
    ]

    def validate(self, value: str, metadata: dict) -> ValidationResult:
        v_lower = value.lower()
        for phrase in self.PHRASES:
            if phrase in v_lower:
                return FailResult(
                    error_message=f"Indicateur d'incertitude détecté: '{phrase}'"
                )
        return PassResult()


@register_validator(name="acaps/min-answer-length", data_type="string")
class MinAnswerLength(Validator):
    """Ensures the answer is substantive, not a one-word refusal or empty string."""

    def __init__(self, min_chars: int = MIN_ANSWER_CHARS, **kwargs):
        super().__init__(**kwargs)
        self.min_chars = min_chars

    def validate(self, value: str, metadata: dict) -> ValidationResult:
        stripped = value.strip()
        if len(stripped) < self.min_chars:
            return FailResult(
                error_message=(
                    f"Réponse trop courte : {len(stripped)} caractères "
                    f"(minimum {self.min_chars})"
                )
            )
        return PassResult()
