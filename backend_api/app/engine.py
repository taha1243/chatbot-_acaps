"""
RAG Engine for Atlas-RAG
Pipeline : embed question → hybrid search → LLM generation.
Configuration loaded from backend_api/config/*.json at startup.
"""
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .config import get_settings, Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _load_json(filename: str) -> Any:
    path = _CONFIG_DIR / filename
    with open(path, encoding="utf-8") as f:
        return json.load(f)


try:
    _PROMPTS = _load_json("prompts.json")
    _ENGINE_CFG = _load_json("engine_config.json")
    logger.info("Engine config loaded from %s", _CONFIG_DIR)
except Exception as exc:
    logger.critical("Failed to load engine config JSON: %s", exc)
    raise


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    title: str
    url: str
    score: float
    text_snippet: str


@dataclass
class RAGResponse:
    answer: str
    citations: List[Citation]
    confidence: float
    context_used: str
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------

class RAGEngine:
    """
    RAG pipeline: embed → hybrid search → LLM answer.
    No keyword routing or section filtering — pure semantic retrieval.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        use_mock: bool = False,
    ):
        self.settings = settings or get_settings()
        self.use_mock = use_mock
        self._llm_client = None
        self._vector_store = None
        self._embedder = None
        logger.info("RAGEngine initialized (mock=%s)", use_mock)

    # ------------------------------------------------------------------ #
    #  Lazy-loaded components                                               #
    # ------------------------------------------------------------------ #

    @property
    def llm_client(self):
        if self._llm_client is None:
            if self.use_mock:
                self._llm_client = MockLLMClient()
            else:
                from openai import OpenAI
                import httpx
                default_headers = None
                if self.settings.llm_provider == "openrouter":
                    default_headers = {}
                    if self.settings.openrouter_site_url:
                        default_headers["HTTP-Referer"] = self.settings.openrouter_site_url
                    if self.settings.openrouter_title:
                        default_headers["X-Title"] = self.settings.openrouter_title
                self._llm_client = OpenAI(
                    base_url=self.settings.vllm_url,
                    api_key=self.settings.vllm_api_key,
                    default_headers=default_headers,
                    timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
                )
        return self._llm_client

    @property
    def vector_store(self):
        if self._vector_store is None:
            if self.use_mock:
                from data_ingestion.vector_store import MockVectorStore
                self._vector_store = MockVectorStore(
                    table_name=self.settings.vector_table,
                    embedding_dimension=self.settings.embedding_dimension,
                )
            else:
                from data_ingestion.vector_store import VectorStore
                self._vector_store = VectorStore(
                    dsn=self.settings.database_url,
                    table_name=self.settings.vector_table,
                    embedding_dimension=self.settings.embedding_dimension,
                )
        return self._vector_store

    @property
    def embedder(self):
        if self._embedder is None:
            if self.use_mock:
                from data_ingestion.embedder import MockEmbeddingGenerator
                self._embedder = MockEmbeddingGenerator(
                    dimension=self.settings.embedding_dimension
                )
            else:
                from data_ingestion.embedder import EmbeddingGenerator
                self._embedder = EmbeddingGenerator(
                    model_name=self.settings.embedding_model
                )
        return self._embedder

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _is_greeting(self, text: str) -> bool:
        clean = text.strip().lower()
        # Word-boundary match to avoid "hi" matching inside "fichiers", etc.
        return any(
            re.search(r'(?<![a-zàâäéèêëîïôùûüç])' + re.escape(kw) + r'(?![a-zàâäéèêëîïôùûüç])', clean)
            for kw in _ENGINE_CFG["greeting_keywords"]
        )

    def _sanitize_response(self, text: str) -> str:
        """Strip Cyrillic characters that the 3b model occasionally injects."""
        return re.sub(r'[Ѐ-ӿ]+', '', text).strip()

    def _post_validate(self, answer: str) -> bool:
        """Return False if the answer is empty or contains hallucination markers."""
        if len(answer.strip()) < _ENGINE_CFG["min_answer_length"]:
            return False
        answer_lower = answer.lower()
        for indicator in _ENGINE_CFG["hallucination_indicators"]:
            if indicator in answer_lower:
                logger.warning("Hallucination indicator detected: '%s'", indicator)
                return False
        return True

    # ------------------------------------------------------------------ #
    #  Main RAG pipeline                                                    #
    # ------------------------------------------------------------------ #

    def query(self, question: str) -> RAGResponse:
        logger.info("Processing query: %s...", question[:100])

        # 0. Greeting fast-path
        if self._is_greeting(question):
            logger.info("Detected greeting")
            prompt = _PROMPTS["greeting_prompt"].format(question=question)
            return RAGResponse(
                answer=self._generate_conversational(prompt),
                citations=[],
                confidence=1.0,
                context_used="",
                metadata={"greeting": True},
            )

        # 1. Embed the raw question
        query_embedding = self.embedder.embed_query(question)

        # 2. Hybrid search across the full guide
        search_results = self.vector_store.hybrid_search(
            query_embedding=query_embedding,
            query_text=question,
            top_k=self.settings.top_k_results,
            score_threshold=self.settings.similarity_threshold,
            keyword_boost=_ENGINE_CFG["keyword_boost"],
            file_name=_ENGINE_CFG["portal_file"],
        )

        # No results → redirect
        if not search_results:
            logger.info("No relevant chunks found — redirecting")
            prompt = _PROMPTS["general_prompt"].format(question=question)
            return RAGResponse(
                answer=self._generate_conversational(prompt),
                citations=[],
                confidence=0.0,
                context_used="",
                metadata={"retrieval_count": 0, "source": "general_redirect"},
            )

        # 3. Build context
        context_parts = []
        citations: List[Citation] = []
        for result in search_results:
            context_parts.append(f"[Source: {result.header_path}]\n{result.text}")
            url = result.full_url if result.full_url and not result.full_url.startswith("#") else ""
            snippet = result.text[:400] + "..." if len(result.text) > 400 else result.text
            citations.append(Citation(
                title=result.header_path,
                url=url,
                score=result.score,
                text_snippet=snippet,
            ))

        context = "\n\n---\n\n".join(context_parts)
        avg_score = sum(r.score for r in search_results) / len(search_results)

        # 4. LLM generates answer from context
        prompt = _PROMPTS["system_prompt"].format(context=context, question=question)
        answer = self._generate(prompt, context=context)

        # 5. Post-validation
        if not self._post_validate(answer):
            logger.warning("Post-validation failed — falling back to general_prompt")
            fallback_prompt = _PROMPTS["general_prompt"].format(question=question)
            answer = self._generate_conversational(fallback_prompt)
            citations = []

        logger.info("Answer generated with %d citations (avg_score=%.3f)", len(citations), avg_score)

        return RAGResponse(
            answer=answer,
            citations=citations,
            confidence=avg_score,
            context_used=context,
            metadata={
                "retrieval_count": len(search_results),
                "top_score": search_results[0].score,
                "model": self.settings.vllm_model,
            },
        )

    # ------------------------------------------------------------------ #
    #  LLM helpers                                                          #
    # ------------------------------------------------------------------ #

    def _generate(self, prompt: str, context: str = "") -> str:
        if self.use_mock:
            return self.llm_client.generate(prompt)
        try:
            response = self.llm_client.chat.completions.create(
                model=self.settings.vllm_model,
                messages=[
                    {"role": "system", "content": _PROMPTS["system_message_rag"]},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )
            return self._sanitize_response(response.choices[0].message.content)
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            if self.settings.llm_fallback_enabled and context:
                return self._generate_fallback_response(context)
            return _PROMPTS["fallback_unavailable"]

    def _generate_conversational(self, prompt: str) -> str:
        if self.use_mock:
            return _PROMPTS["conversational_fallback"]
        try:
            response = self.llm_client.chat.completions.create(
                model=self.settings.vllm_model,
                messages=[
                    {"role": "system", "content": _PROMPTS["system_message_conversational"]},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=self.settings.max_tokens,
            )
            return self._sanitize_response(response.choices[0].message.content)
        except Exception as e:
            logger.error("Error generating conversational response: %s", e)
            return _PROMPTS["greeting_fallback"]

    def _generate_fallback_response(self, context: str) -> str:
        if "[Source:" in context:
            parts = context.split("---")
            first_source = parts[0].strip() if parts else context[:500]
        else:
            first_source = context[:500]
        return (
            _PROMPTS["fallback_context_header"]
            + first_source
            + _PROMPTS["fallback_context_footer"]
        )

    def health_check(self) -> Dict[str, Any]:
        health = {"vector_store": False, "llm": False, "embedder": False}
        try:
            health["vector_store"] = self.vector_store.health_check()
        except Exception as e:
            logger.error("Vector store health check failed: %s", e)
        try:
            self.embedder.embed_query("test")
            health["embedder"] = True
        except Exception as e:
            logger.error("Embedder health check failed: %s", e)
        try:
            if self.use_mock:
                health["llm"] = True
            else:
                self.llm_client.models.list()
                health["llm"] = True
        except Exception as e:
            logger.error("LLM health check failed: %s", e)
        health["overall"] = all(health.values())
        return health


class MockLLMClient:
    def generate(self, prompt: str) -> str:
        if "réclamation" in prompt.lower() or "soumettre" in prompt.lower():
            return "Pour soumettre une réclamation, accédez au portail ACAPS et cliquez sur 'Nouvelle réclamation'."
        return "Basé sur la documentation disponible, veuillez consulter le guide du portail ACAPS."


# Singleton
_engine_instance: Optional[RAGEngine] = None


def get_engine(use_mock: bool = False) -> RAGEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngine(use_mock=use_mock)
    return _engine_instance
