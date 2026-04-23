"""
Semantic greeting / small-talk detection for Atlas-RAG.

A user query is compared against a small set of canonical greeting and
small-talk anchor phrases using the same BGE-M3 embedder used by the
retrieval path. If the maximum cosine similarity exceeds a threshold,
the query is treated as conversational and the RAG pipeline is bypassed.

Complements the fast keyword/regex pre-filter in RAGEngine._is_greeting:
the regex catches the common cases cheaply; this router catches typos,
paraphrases, and variants that the regex misses (e.g. "salut, ça roule ?",
"bonjour comment vas-tu mon ami", "kif dayer", "buenos días").
"""
import logging
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Canonical greeting / small-talk phrases. Kept short and representative —
# semantic similarity generalizes across paraphrases and minor typos.
GREETING_ANCHORS: List[str] = [
    # French
    "bonjour",
    "salut, comment vas-tu ?",
    "bonsoir, comment ça va ?",
    "coucou, ça roule ?",
    "comment allez-vous aujourd'hui ?",
    "quoi de neuf ?",
    "tout va bien ?",
    # English
    "hello, how are you?",
    "hi there, how's it going?",
    "good morning",
    "good evening, what's up?",
    # Arabic / Darija (transliterated)
    "salam, kif dayer ?",
    "marhaba, labas ?",
    # Spanish (occasional)
    "hola, cómo estás?",
]


class SemanticGreetingRouter:
    """
    Embedding-based greeting detector. Lazy-loads anchor embeddings on
    first use, then performs a single dot-product against the anchor
    matrix per query (cheap relative to the LLM call it avoids).
    """

    def __init__(
        self,
        embedder_provider: Callable,
        threshold: float = 0.70,
        anchors: Optional[List[str]] = None,
    ):
        """
        Args:
            embedder_provider: zero-arg callable returning the shared embedder
                instance. Lazy — no model is loaded until the first check.
            threshold: cosine similarity above which a query is a greeting.
                BGE-M3 on short related phrases scores roughly 0.6-0.9 for
                semantic matches; 0.70 is a conservative default.
            anchors: optional override of the default GREETING_ANCHORS list.
        """
        self._embedder_provider = embedder_provider
        self.threshold = threshold
        self.anchors = anchors if anchors is not None else GREETING_ANCHORS
        self._anchor_matrix: Optional[np.ndarray] = None

    def _ensure_anchors_ready(self) -> bool:
        """Lazy-compute anchor embeddings. Returns False if embedder is mock."""
        if self._anchor_matrix is not None:
            return True

        try:
            embedder = self._embedder_provider()
        except Exception as e:
            logger.warning(f"Greeting router: embedder unavailable ({e}); skipping check")
            return False

        # Mock embeddings are random noise — the check would be meaningless.
        model_name = getattr(embedder, "model_name", "")
        if "mock" in model_name.lower():
            logger.info("Greeting router: mock embedder detected, semantic check disabled")
            return False

        logger.info(f"Greeting router: embedding {len(self.anchors)} anchor phrases")
        vectors = [embedder.embed_text(phrase).embedding for phrase in self.anchors]
        self._anchor_matrix = np.array(vectors, dtype=np.float32)
        return True

    def check(self, query: str) -> Tuple[bool, float]:
        """
        Evaluate a query against the greeting anchors.

        Returns:
            (is_greeting, max_similarity). is_greeting=True means the query
            is semantically close to at least one anchor phrase.
        """
        if not self._ensure_anchors_ready():
            # Fail-closed for greeting detection: if we can't check, assume
            # it's a real question and let RAG handle it.
            return False, 0.0

        embedder = self._embedder_provider()
        query_vec = np.array(embedder.embed_text(query).embedding, dtype=np.float32)

        # EmbeddingGenerator L2-normalizes by default, so dot product == cosine.
        similarities = self._anchor_matrix @ query_vec
        max_sim = float(similarities.max())

        return max_sim >= self.threshold, max_sim
