"""
Semantic scope detection for Atlas-RAG guardrails.

Complements regex-based OFF_TOPIC_PATTERNS with a language-agnostic
embedding similarity check. A user query is compared against a set of
ACAPS domain anchor phrases; if the maximum cosine similarity falls
below a threshold, the query is flagged as out-of-scope.

Uses the same embedder instance as the RAG retrieval path, so no extra
model is loaded into memory.
"""
import logging
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Anchor phrases covering the ACAPS documentation domain. They do not need
# to be exhaustive — semantic similarity generalizes across paraphrases.
# Keep them short and representative of real questions users would ask.
ACAPS_ANCHORS: List[str] = [
    "Obligations des compagnies d'assurance",
    "Déclaration de sinistre et indemnisation",
    "Délais de paiement des indemnités aux assurés",
    "Exigences prudentielles pour les assureurs",
    "Capital minimum des sociétés d'assurance",
    "Contrôle et supervision des entreprises d'assurance",
    "Prévoyance sociale et retraite au Maroc",
    "Règlement intérieur et organisation de l'ACAPS",
    "Gouvernance des organismes assureurs",
    "Lutte contre le blanchiment d'argent dans l'assurance",
    "Protection des assurés et des bénéficiaires",
    "Recours et réclamations auprès de l'ACAPS",
    "Suivre une réclamation en ligne sur le site",
    "Agrément des intermédiaires d'assurance",
    "Réassurance et rétrocession",
    "Provisions techniques des compagnies d'assurance",
    "Comptes annuels et reporting des assureurs",
    "Contrats d'assurance et clauses obligatoires",
    "Rôle de l'autorité de contrôle des assurances",
    "Insurance regulation and compliance in Morocco",
    "Claims handling and policyholder rights",
]


class SemanticScopeChecker:
    """
    Checks whether a query lies within the ACAPS domain using embedding
    similarity to a set of anchor phrases.
    """

    def __init__(
        self,
        embedder_provider: Callable,
        threshold: float = 0.35,
        anchors: Optional[List[str]] = None,
    ):
        """
        Args:
            embedder_provider: zero-arg callable returning the shared embedder
                instance. Called lazily on the first check, so no model is
                loaded until needed.
            threshold: cosine similarity below which a query is OFF-TOPIC.
            anchors: optional override of the default ACAPS_ANCHORS list.
        """
        self._embedder_provider = embedder_provider
        self.threshold = threshold
        self.anchors = anchors if anchors is not None else ACAPS_ANCHORS
        self._anchor_matrix: Optional[np.ndarray] = None

    def _ensure_anchors_ready(self) -> bool:
        """Lazy-compute anchor embeddings. Returns False if embedder is mock."""
        if self._anchor_matrix is not None:
            return True

        try:
            embedder = self._embedder_provider()
        except Exception as e:
            logger.warning(f"Semantic scope: embedder unavailable ({e}); skipping check")
            return False

        # Skip semantic check in mock/debug mode — random embeddings are noise.
        model_name = getattr(embedder, "model_name", "")
        if "mock" in model_name.lower():
            logger.info("Semantic scope: mock embedder detected, check disabled")
            return False

        logger.info(f"Semantic scope: embedding {len(self.anchors)} anchor phrases")
        vectors = [embedder.embed_text(phrase).embedding for phrase in self.anchors]
        self._anchor_matrix = np.array(vectors, dtype=np.float32)
        return True

    def check(self, query: str) -> Tuple[bool, float]:
        """
        Evaluate a query against the anchor set.

        Returns:
            (in_scope, max_similarity) — in_scope=True means the query is
            semantically close to at least one anchor; max_similarity is the
            highest cosine similarity across all anchors.
        """
        if not self._ensure_anchors_ready():
            # Fail-open: if we cannot check, don't block.
            return True, 1.0

        embedder = self._embedder_provider()
        query_vec = np.array(embedder.embed_text(query).embedding, dtype=np.float32)

        # Embeddings are L2-normalized by EmbeddingGenerator, so dot product
        # equals cosine similarity.
        similarities = self._anchor_matrix @ query_vec
        max_sim = float(similarities.max())

        in_scope = max_sim >= self.threshold
        return in_scope, max_sim
