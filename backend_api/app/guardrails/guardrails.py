"""
Guardrails Module for Atlas-RAG
Implements input/output validation and hallucination prevention.
"""
import re
import unicodedata
import logging
from typing import Optional, Dict, Any, List, Tuple, Set, Callable
from dataclasses import dataclass
from enum import Enum

from .semantic_scope import SemanticScopeChecker

logger = logging.getLogger(__name__)


# Stop words used by the grounding heuristic. Kept deliberately small and
# focused on high-frequency function words in FR/EN — not a full stoplist.
_STOP_WORDS: Set[str] = {
    # French
    "les", "des", "une", "aux", "que", "qui", "quoi", "dont", "ces", "cet",
    "cette", "son", "ses", "leur", "leurs", "est", "sont", "etre", "avoir",
    "avec", "sans", "pour", "par", "dans", "sur", "sous", "chez", "vers",
    "entre", "mais", "donc", "car", "non", "pas", "plus", "moins", "tres",
    "tout", "tous", "toute", "toutes", "aussi", "encore", "deja", "alors",
    "comme", "autre", "meme", "selon", "afin", "ainsi", "lors", "puis",
    "ici", "cela", "celui", "celle", "ceux",
    # English
    "the", "and", "are", "was", "were", "been", "being", "have", "has",
    "had", "does", "did", "will", "would", "should", "could", "may",
    "might", "can", "this", "that", "these", "those", "with", "from",
    "its", "they", "them", "their", "you", "your", "our", "not", "for",
    "but", "into", "onto", "over", "under", "about", "there", "here",
}


def _normalize(text: str) -> str:
    """Lowercase + strip accents for consistent token matching."""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def _significant_tokens(text: str) -> Set[str]:
    """Extract content tokens (len >= 3, not a stop word) from normalized text."""
    normalized = _normalize(text)
    tokens = re.findall(r"[a-z0-9]{3,}", normalized)
    return {t for t in tokens if t not in _STOP_WORDS}


class RailType(Enum):
    """Types of guardrails."""
    INPUT = "input"
    OUTPUT = "output"
    FACT_CHECK = "fact_check"


class BlockReason(Enum):
    """Reasons for blocking a message."""
    JAILBREAK = "jailbreak_attempt"
    TOXICITY = "toxic_content"
    OFF_TOPIC = "off_topic"
    LOW_CONFIDENCE = "low_retrieval_confidence"
    HALLUCINATION = "potential_hallucination"
    PII_DETECTED = "pii_detected"


@dataclass
class GuardrailResult:
    """Result of guardrail check."""
    passed: bool
    blocked_reason: Optional[BlockReason] = None
    message: Optional[str] = None
    confidence: float = 1.0
    
    @property
    def should_block(self) -> bool:
        return not self.passed


class InputGuardrails:
    """
    Input validation guardrails.
    Checks for jailbreaks, toxicity, and off-topic queries.
    """
    
    # Jailbreak patterns
    JAILBREAK_PATTERNS = [
        r"ignore.*(?:previous|your).*instructions",
        r"forget.*(?:previous|your).*instructions",
        r"pretend.*(?:you are|to be)",
        r"act as if",
        r"you are now",
        r"new persona",
        r"bypass.*(?:filters|restrictions)",
        r"DAN.*mode",
        r"developer.*mode",
    ]
    
    # Off-topic patterns — English + French (ACAPS users write mostly in FR).
    # Patterns target clear entertainment/lifestyle/sports queries; they avoid
    # ACAPS-legitimate words like "président", "capital", "cours" that appear
    # in regulatory context.
    OFF_TOPIC_PATTERNS = [
        # English
        r"(?:what's|what is).*weather",
        r"(?:tell|write).*(?:joke|story|poem)",
        r"(?:who won|score of).*(?:game|match)",
        r"(?:latest|recent).*news",
        r"(?:stock|crypto).*price",
        r"(?:recipe|cook).*",
        r"(?:translate|translation)",
        r"\bbest\s+(?:player|footballer|singer|actor|movie|song)\b",
        r"(?:who is|who's)\s+(?:the\s+)?(?:best|greatest|top)\s+(?:player|footballer|singer|actor)",
        # French — weather
        r"\bm[ée]t[ée]o\b",
        r"(?:quel|quelle).*temps.*(?:fait|qu'il)",
        # French — jokes / stories / poems / songs
        r"(?:raconte|[ée]cris|dis|donne).*(?:blague|histoire\s+dr[ôo]le|po[èe]me|po[ée]sie|chanson)",
        # French — sports / entertainment rankings
        r"(?:meilleur|plus\s+grand|top)\s+(?:joueur|footballeur|sportif|[ée]quipe|chanteur|chanteuse|acteur|actrice|artiste|film|livre|album)",
        r"(?:meilleur|plus\s+grand|top)\s+\w+\s+du\s+monde",
        r"(?:qui est|qui sont)\s+(?:le|la|les)?\s*(?:meilleur|joueur|footballeur|acteur|actrice|chanteur|chanteuse|artiste)",
        # French — cooking / translation / news / crypto
        r"(?:recette\s+de|comment\s+cuisiner)",
        r"\btraduis\s+",
        r"(?:actualit[ée]s?|derni[èe]res?\s+nouvelles?)",
        r"(?:cours|prix|cotation)\s+(?:du\s+|de\s+la\s+|des\s+|de\s+l[''])?(?:bourse|crypto|bitcoin|ethereum)",
    ]
    
    # Toxic patterns (simplified)
    TOXIC_PATTERNS = [
        r"(?:you are|you're).*(?:stupid|idiot|dumb)",
        r"(?:this is|it's).*(?:garbage|trash|useless)",
        r"\b(?:hate|kill|die)\b",
    ]

    # PII patterns — emails, Moroccan phone numbers, CIN, IBAN, credit cards.
    PII_PATTERNS: Dict[str, str] = {
        "email": r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b",
        "phone_ma": r"(?:\+?212|0)\s*[5-7](?:[\s.-]?\d{2}){4}\b",
        "cin_ma": r"\b[A-Z]{1,2}\d{5,7}\b",
        "iban": r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
        "credit_card": r"\b(?:\d{4}[\s-]?){3}\d{4}\b",
    }

    def __init__(
        self,
        enabled: bool = True,
        enable_pii_detection: bool = True,
        semantic_scope: Optional[SemanticScopeChecker] = None,
    ):
        self.enabled = enabled
        self.enable_pii_detection = enable_pii_detection
        self.semantic_scope = semantic_scope
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        self.jailbreak_re = [re.compile(p, re.IGNORECASE) for p in self.JAILBREAK_PATTERNS]
        self.off_topic_re = [re.compile(p, re.IGNORECASE) for p in self.OFF_TOPIC_PATTERNS]
        self.toxic_re = [re.compile(p, re.IGNORECASE) for p in self.TOXIC_PATTERNS]
        self.pii_re = {name: re.compile(p) for name, p in self.PII_PATTERNS.items()}

    def check(self, user_input: str) -> GuardrailResult:
        """
        Check user input against all input guardrails.

        Args:
            user_input: The user's message

        Returns:
            GuardrailResult indicating pass/fail
        """
        if not self.enabled:
            return GuardrailResult(passed=True)

        # Check jailbreak
        result = self._check_jailbreak(user_input)
        if result.should_block:
            logger.warning(f"Jailbreak attempt detected: {user_input[:100]}")
            return result

        # Check toxicity
        result = self._check_toxicity(user_input)
        if result.should_block:
            logger.warning(f"Toxic content detected: {user_input[:100]}")
            return result

        # Check PII
        if self.enable_pii_detection:
            result = self._check_pii(user_input)
            if result.should_block:
                logger.warning(f"PII detected in input: {result.message}")
                return result

        # Check off-topic (regex — fast, precise, catches obvious cases)
        result = self._check_off_topic(user_input)
        if result.should_block:
            logger.info(f"Off-topic query detected (regex): {user_input[:100]}")
            return result

        # Check off-topic (semantic — catches paraphrases the regex misses)
        if self.semantic_scope is not None:
            result = self._check_semantic_scope(user_input)
            if result.should_block:
                logger.info(
                    f"Off-topic query detected (semantic, sim={result.confidence:.2f}): "
                    f"{user_input[:100]}"
                )
                return result

        return GuardrailResult(passed=True)

    def _check_semantic_scope(self, text: str) -> GuardrailResult:
        """Use embedding similarity against ACAPS anchors to detect off-topic."""
        try:
            in_scope, similarity = self.semantic_scope.check(text)
        except Exception as e:
            logger.warning(f"Semantic scope check failed, skipping: {e}")
            return GuardrailResult(passed=True)

        if not in_scope:
            return GuardrailResult(
                passed=False,
                blocked_reason=BlockReason.OFF_TOPIC,
                message="Je suis spécialisé dans la réglementation ACAPS, les règles internes et l'utilisation du site. Je ne peux pas traiter de sujets en dehors de ce périmètre.",
                confidence=similarity,
            )
        return GuardrailResult(passed=True, confidence=similarity)

    def _check_pii(self, text: str) -> GuardrailResult:
        """Check for personally identifiable information in input."""
        for name, pattern in self.pii_re.items():
            if pattern.search(text):
                return GuardrailResult(
                    passed=False,
                    blocked_reason=BlockReason.PII_DETECTED,
                    message=f"Votre message semble contenir des données personnelles ({name}). Merci de les retirer avant de renvoyer votre question."
                )
        return GuardrailResult(passed=True)
    
    def _check_jailbreak(self, text: str) -> GuardrailResult:
        """Check for jailbreak attempts."""
        for pattern in self.jailbreak_re:
            if pattern.search(text):
                return GuardrailResult(
                    passed=False,
                    blocked_reason=BlockReason.JAILBREAK,
                    message="Je ne peux répondre qu'aux questions concernant la documentation et la réglementation ACAPS."
                )
        return GuardrailResult(passed=True)

    def _check_toxicity(self, text: str) -> GuardrailResult:
        """Check for toxic content."""
        for pattern in self.toxic_re:
            if pattern.search(text):
                return GuardrailResult(
                    passed=False,
                    blocked_reason=BlockReason.TOXICITY,
                    message="Merci de reformuler votre question de manière respectueuse."
                )
        return GuardrailResult(passed=True)

    def _check_off_topic(self, text: str) -> GuardrailResult:
        """Check for off-topic queries."""
        for pattern in self.off_topic_re:
            if pattern.search(text):
                return GuardrailResult(
                    passed=False,
                    blocked_reason=BlockReason.OFF_TOPIC,
                    message="Je suis spécialisé dans la réglementation ACAPS, les règles internes et l'utilisation du site. Je ne peux pas traiter de sujets en dehors de ce périmètre."
                )
        return GuardrailResult(passed=True)


class OutputGuardrails:
    """
    Output validation guardrails.
    Checks for hallucinations and ensures factual consistency.
    """

    # Subjective / uncertainty markers. Narrow on purpose — a legitimate
    # grounded answer should not open with "I think" or "à mon avis".
    HALLUCINATION_INDICATORS = [
        r"\bI think\b",
        r"\bI believe\b",
        r"\bin my opinion\b",
        r"\bas far as I know\b",
        r"\bje pense que\b",
        r"\bje crois que\b",
        r"\bà mon avis\b",
        r"\bil me semble que\b",
    ]

    # Fallback / system responses the grounding check must not penalize.
    _SYSTEM_RESPONSE_PREFIXES = (
        "i cannot find",
        "je ne trouve pas",
        "je ne peux pas",
        "**note:**",
        "bonjour",
    )

    def __init__(
        self,
        enabled: bool = True,
        min_confidence_threshold: float = 0.35,
        groundedness_threshold: float = 0.50,
    ):
        self.enabled = enabled
        self.min_confidence_threshold = min_confidence_threshold
        self.groundedness_threshold = groundedness_threshold
        self.hallucination_re = [
            re.compile(p, re.IGNORECASE) for p in self.HALLUCINATION_INDICATORS
        ]

    def check(
        self,
        response: str,
        context: str,
        retrieval_score: float
    ) -> GuardrailResult:
        """
        Check model output for hallucinations.

        Args:
            response: The model's response
            context: The retrieved context used
            retrieval_score: Similarity score from retrieval

        Returns:
            GuardrailResult indicating pass/fail
        """
        if not self.enabled:
            return GuardrailResult(passed=True)

        # Check retrieval confidence
        if retrieval_score < self.min_confidence_threshold:
            logger.info(f"Low retrieval score: {retrieval_score}")
            return GuardrailResult(
                passed=False,
                blocked_reason=BlockReason.LOW_CONFIDENCE,
                message="Je ne trouve pas d'information pertinente dans les documents disponibles.",
                confidence=retrieval_score
            )

        # Check for hallucination indicators
        result = self._check_hallucination_phrases(response)
        if result.should_block:
            return result

        # Check if response references content not in context
        result = self._check_grounding(response, context)
        if result.should_block:
            return result

        return GuardrailResult(passed=True, confidence=retrieval_score)

    def _check_hallucination_phrases(self, response: str) -> GuardrailResult:
        """Check for phrases that indicate uncertainty/hallucination."""
        for pattern in self.hallucination_re:
            if pattern.search(response):
                return GuardrailResult(
                    passed=False,
                    blocked_reason=BlockReason.HALLUCINATION,
                    message="Je ne peux pas fournir de réponse définitive sur la base des documents disponibles."
                )
        return GuardrailResult(passed=True)

    def _check_grounding(self, response: str, context: str) -> GuardrailResult:
        """
        Check that the response is lexically grounded in the retrieved context.

        Token-overlap heuristic: extract content tokens (non-stopword, length
        >= 3) from both response and context, and require at least
        `groundedness_threshold` of the response tokens to appear in the
        context. Cheap, language-agnostic, and catches the most common
        hallucination mode where the LLM invents entities or numbers absent
        from the sources.
        """
        # No context to verify against (greetings, no-retrieval paths) — skip.
        if not context or not context.strip():
            return GuardrailResult(passed=True)

        # Canned system/fallback messages are not LLM generations — skip.
        normalized_response = response.lstrip().lower()
        if normalized_response.startswith(self._SYSTEM_RESPONSE_PREFIXES):
            return GuardrailResult(passed=True)

        response_tokens = _significant_tokens(response)
        if len(response_tokens) < 4:
            # Too few content tokens to produce a reliable ratio.
            return GuardrailResult(passed=True)

        context_tokens = _significant_tokens(context)
        overlap = response_tokens & context_tokens
        ratio = len(overlap) / len(response_tokens)

        if ratio < self.groundedness_threshold:
            ungrounded_sample = list(response_tokens - context_tokens)[:10]
            logger.warning(
                f"Ungrounded response: ratio={ratio:.2f} "
                f"threshold={self.groundedness_threshold} "
                f"ungrounded_sample={ungrounded_sample}"
            )
            return GuardrailResult(
                passed=False,
                blocked_reason=BlockReason.HALLUCINATION,
                message="Je ne peux pas vérifier cette information dans les documents disponibles.",
                confidence=ratio,
            )

        return GuardrailResult(passed=True, confidence=ratio)


class Guardrails:
    """
    Main guardrails orchestrator.
    Combines input and output validation.
    """

    def __init__(
        self,
        enabled: bool = True,
        min_confidence_threshold: float = 0.35,
        groundedness_threshold: float = 0.50,
        enable_pii_detection: bool = True,
        enable_semantic_scope: bool = True,
        semantic_scope_threshold: float = 0.35,
        embedder_provider: Optional[Callable] = None,
    ):
        self.enabled = enabled

        semantic_scope: Optional[SemanticScopeChecker] = None
        if enabled and enable_semantic_scope and embedder_provider is not None:
            semantic_scope = SemanticScopeChecker(
                embedder_provider=embedder_provider,
                threshold=semantic_scope_threshold,
            )

        self.input_rails = InputGuardrails(
            enabled=enabled,
            enable_pii_detection=enable_pii_detection,
            semantic_scope=semantic_scope,
        )
        self.output_rails = OutputGuardrails(
            enabled=enabled,
            min_confidence_threshold=min_confidence_threshold,
            groundedness_threshold=groundedness_threshold,
        )
        logger.info(
            f"Guardrails initialized (enabled={enabled}, "
            f"min_confidence={min_confidence_threshold}, "
            f"groundedness={groundedness_threshold}, "
            f"pii={enable_pii_detection}, "
            f"semantic_scope={semantic_scope is not None}@{semantic_scope_threshold})"
        )
    
    def check_input(self, user_input: str) -> GuardrailResult:
        """Check user input."""
        return self.input_rails.check(user_input)
    
    def check_output(
        self,
        response: str,
        context: str,
        retrieval_score: float
    ) -> GuardrailResult:
        """Check model output."""
        return self.output_rails.check(response, context, retrieval_score)
    
    def get_blocked_response(self, reason: BlockReason) -> str:
        """Get appropriate response for blocked content."""
        responses = {
            BlockReason.JAILBREAK: "Je ne peux répondre qu'aux questions concernant la documentation et la réglementation ACAPS.",
            BlockReason.TOXICITY: "Merci de reformuler votre question de manière respectueuse.",
            BlockReason.OFF_TOPIC: "Je suis spécialisé dans la réglementation ACAPS, les règles internes et l'utilisation du site. Je ne peux pas traiter de sujets en dehors de ce périmètre.",
            BlockReason.LOW_CONFIDENCE: "Je ne trouve pas cette information dans les documents disponibles.",
            BlockReason.HALLUCINATION: "Je ne peux pas fournir de réponse vérifiée sur la base des documents disponibles.",
            BlockReason.PII_DETECTED: "Je ne peux pas traiter une demande contenant des informations personnelles.",
        }
        return responses.get(reason, "Je ne peux pas traiter cette requête.")


# Singleton instance
_guardrails_instance: Optional[Guardrails] = None


def get_guardrails(
    enabled: bool = True,
    min_confidence_threshold: float = 0.35,
    groundedness_threshold: float = 0.50,
    enable_pii_detection: bool = True,
    enable_semantic_scope: bool = True,
    semantic_scope_threshold: float = 0.35,
    embedder_provider: Optional[Callable] = None,
) -> Guardrails:
    """Get or create guardrails instance."""
    global _guardrails_instance
    if _guardrails_instance is None:
        _guardrails_instance = Guardrails(
            enabled=enabled,
            min_confidence_threshold=min_confidence_threshold,
            groundedness_threshold=groundedness_threshold,
            enable_pii_detection=enable_pii_detection,
            enable_semantic_scope=enable_semantic_scope,
            semantic_scope_threshold=semantic_scope_threshold,
            embedder_provider=embedder_provider,
        )
    return _guardrails_instance

