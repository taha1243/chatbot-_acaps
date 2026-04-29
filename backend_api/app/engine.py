"""
RAG Engine for Atlas-RAG — pure semantic architecture.
No keyword routing. No intent lists.
embed_query → hybrid_search → Ollama/qwen2.5:7b → validate.
"""
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from langdetect import DetectorFactory, LangDetectException
from langdetect import detect as _langdetect_detect

from .config import Settings, get_settings

DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

# ── Config loaded once at import time ────────────────────────────────────────
_CONFIG_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config")
)


def _load_json(filename: str) -> dict:
    path = os.path.join(_CONFIG_DIR, filename)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


_ENGINE_CFG: dict = _load_json("engine_config.json")
_PROMPTS: dict = _load_json("prompts.json")

_GREETING_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"(?<![a-zàâäéèêëîïôùûüç])" + re.escape(kw) + r"(?![a-zàâäéèêëîïôùûüç])",
        re.IGNORECASE,
    )
    for kw in _ENGINE_CFG["greeting_keywords"]
]


# ── Data classes ──────────────────────────────────────────────────────────────

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


# ── Ollama LLM client (OpenAI-compatible API) ─────────────────────────────────

class OllamaLLMClient:
    """Ollama inference via OpenAI-compatible REST API."""

    def __init__(self, model: str, base_url: str, api_key: str):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._client = None
        logger.info("OllamaLLMClient created — model: %s base_url: %s", model, base_url)

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def generate(self, messages: list, max_new_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.8,
        )
        return response.choices[0].message.content.strip()

    def generate_stream(self, messages: list, max_new_tokens: int = 1024) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.8,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def health_check(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False


# ── Mock LLM client ──────────────────────────────────────────────────────────

class MockLLMClient:
    def generate(self, messages: list, max_new_tokens: int = 600) -> str:
        content = messages[-1]["content"].lower() if messages else ""
        if "reclamation" in content or "réclamation" in content or "مطالبة" in content:
            return _PROMPTS["mock_reclamation_fr"]
        return _PROMPTS["mock_fallback_fr"]

    def generate_stream(self, messages: list, max_new_tokens: int = 600) -> Iterator[str]:
        yield self.generate(messages)

    def health_check(self) -> bool:
        return True


# ── Main RAG Engine ───────────────────────────────────────────────────────────

class RAGEngine:
    """
    Pure semantic RAG: embed → search → Ollama → validate.
    No keyword lists. No intent routing. Confidence-based grounding only.
    """

    def __init__(self, settings: Optional[Settings] = None, use_mock: bool = False):
        self.settings = settings or get_settings()
        self.use_mock = use_mock
        self._llm_client = None
        self._vector_store = None
        self._embedder = None
        logger.info("RAGEngine initialized (mock=%s)", use_mock)

    # ── Lazy components ───────────────────────────────────────────────────────

    @property
    def llm_client(self):
        if self._llm_client is None:
            if self.use_mock:
                self._llm_client = MockLLMClient()
            else:
                self._llm_client = OllamaLLMClient(
                    model=self.settings.llm_model,
                    base_url=self.settings.llm_base_url,
                    api_key=self.settings.llm_api_key,
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

    # ── Language detection ────────────────────────────────────────────────────

    def _detect_language(self, text: str) -> str:
        if not text:
            return "fr"
        arabic_chars = sum(1 for c in text if "؀" <= c <= "ۿ")
        if arabic_chars > len(text) * 0.30:
            return "ar"
        try:
            lang = _langdetect_detect(text)
            return "ar" if lang == "ar" else "fr"
        except LangDetectException:
            return "fr"

    # ── Greeting detection ────────────────────────────────────────────────────

    def _is_greeting(self, text: str) -> bool:
        clean = text.strip().lower()
        return any(pat.search(clean) for pat in _GREETING_PATTERNS)

    # ── Pure semantic retrieval ───────────────────────────────────────────────

    def _retrieve(self, question: str) -> Tuple[List[Any], str]:
        """Embed question → hybrid_search. No keyword routing."""
        lang = self._detect_language(question)
        portal_file = (
            _ENGINE_CFG["portal_file_ar"]
            if lang == "ar"
            else _ENGINE_CFG["portal_file_fr"]
        )
        query_embedding = self.embedder.embed_query(question)
        search_results = self.vector_store.hybrid_search(
            query_embedding=query_embedding,
            query_text=question,
            top_k=self.settings.top_k_results,
            score_threshold=self.settings.similarity_threshold,
            file_name=portal_file,
        )
        logger.info("Retrieved %d chunks (lang=%s)", len(search_results), lang)
        return search_results, lang

    # ── Grounding check ───────────────────────────────────────────────────────

    def _is_grounded_query(self, search_results: List[Any]) -> Tuple[bool, str]:
        if not search_results:
            return False, "no_results"
        avg_score = sum(r.score for r in search_results) / len(search_results)
        if avg_score >= _ENGINE_CFG["min_confidence_threshold"]:
            return True, "confident_retrieval"
        return False, "low_confidence"

    # ── Post-validation ───────────────────────────────────────────────────────

    _GROUNDING_TOKEN_RE = re.compile(r"[a-zàâäéèêëîïôùûüç؀-ۿ]{4,}")
    _GROUNDING_STOPWORDS = frozenset({
        "alors", "ainsi", "avec", "aussi", "avant", "apres", "après",
        "cela", "cette", "celui", "celle", "comme", "comment", "dans",
        "donc", "elle", "elles", "etait", "était", "etre", "être",
        "faut", "leur", "leurs", "mais", "meme", "même", "moins",
        "nous", "pour", "plus", "puis", "quand", "quelle", "quelles",
        "quelqu", "quelque", "quelques", "quels", "quoi", "sans",
        "sont", "sous", "tres", "très", "tout", "tous", "toute",
        "toutes", "vous", "votre", "voici", "voila", "voilà",
        "هذا", "هذه", "ذلك", "تلك", "الذي", "التي", "الذين",
        "كان", "كانت", "يكون", "تكون", "مثل", "حيث", "حول",
        "بعد", "قبل", "عند", "لكن", "غير", "بين", "أيضا", "أيضًا",
    })

    def _grounding_tokens(self, text: str) -> set:
        return {
            tok for tok in self._GROUNDING_TOKEN_RE.findall(text.lower())
            if tok not in self._GROUNDING_STOPWORDS
        }

    def _is_answer_grounded(self, answer: str, context: str) -> bool:
        """
        Verify the answer's content words actually appear in the retrieved
        context. Catches cases where the LLM falls back to its world
        knowledge (e.g., "La capitale du Maroc est Rabat") despite a
        system prompt restricting it to the guide.
        """
        if not context.strip():
            return False
        answer_tokens = self._grounding_tokens(answer)
        if len(answer_tokens) < _ENGINE_CFG["answer_grounding_min_tokens"]:
            return True
        context_tokens = self._grounding_tokens(context)
        overlap = answer_tokens & context_tokens
        ratio = len(overlap) / len(answer_tokens)
        if ratio < _ENGINE_CFG["answer_grounding_min_ratio"]:
            logger.warning(
                "Answer not grounded — overlap %.2f (%d/%d tokens). Missing: %s",
                ratio, len(overlap), len(answer_tokens),
                sorted(answer_tokens - context_tokens)[:10],
            )
            return False
        return True

    def _post_validate(self, answer: str, confidence: float, context: str) -> bool:
        stripped = answer.strip()
        if not stripped:
            return False
        if len(stripped) < _ENGINE_CFG["min_answer_length"]:
            return False
        return self._is_answer_grounded(stripped, context)

    # ── Context & citations ───────────────────────────────────────────────────

    def _build_context_and_citations(
        self, search_results: List[Any]
    ) -> Tuple[str, List[Citation]]:
        max_chars = _ENGINE_CFG["max_chunk_chars_for_prompt"]
        context_parts: List[str] = []
        citations: List[Citation] = []
        for result in search_results:
            trimmed = result.text[:max_chars]
            if len(result.text) > max_chars:
                trimmed += "..."
            context_parts.append(trimmed)
            url = (
                result.full_url
                if result.full_url and not result.full_url.startswith("#")
                else ""
            )
            snippet = result.text[:400] + "..." if len(result.text) > 400 else result.text
            citations.append(
                Citation(title=result.header_path, url=url, score=result.score, text_snippet=snippet)
            )
        return "\n\n---\n\n".join(context_parts), citations

    # ── Response builders ─────────────────────────────────────────────────────

    def _document_only_response(self, question: str) -> str:
        lang = self._detect_language(question)
        key = "document_only_ar" if lang == "ar" else "document_only_fr"
        return _PROMPTS[key]

    def _sys_msg(self, lang: str) -> str:
        return _PROMPTS["sys_msg_ar"] if lang == "ar" else _PROMPTS["sys_msg_fr"]

    # ── LLM generation ────────────────────────────────────────────────────────

    def _generate(self, lang: str, context: str, question: str) -> str:
        prompt_key = "system_prompt_ar" if lang == "ar" else "system_prompt_fr"
        user_content = _PROMPTS[prompt_key].format(context=context, question=question)
        messages = [
            {"role": "system", "content": self._sys_msg(lang)},
            {"role": "user", "content": user_content},
        ]
        try:
            return self.llm_client.generate(messages, max_new_tokens=self.settings.max_tokens)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return _PROMPTS["fallback_response_fr"].format(context=context[:500])

    def _generate_conversational(self, question: str) -> str:
        prompt = _PROMPTS["greeting_prompt"].format(question=question)
        messages = [
            {"role": "system", "content": _PROMPTS["greeting_sys_msg"]},
            {"role": "user", "content": prompt},
        ]
        try:
            return self.llm_client.generate(messages, max_new_tokens=200)
        except Exception:
            return _PROMPTS["mock_greeting_fr"]

    def _stream_generate(self, lang: str, context: str, question: str) -> Iterator[str]:
        prompt_key = "system_prompt_ar" if lang == "ar" else "system_prompt_fr"
        user_content = _PROMPTS[prompt_key].format(context=context, question=question)
        messages = [
            {"role": "system", "content": self._sys_msg(lang)},
            {"role": "user", "content": user_content},
        ]
        yield from self.llm_client.generate_stream(
            messages, max_new_tokens=self.settings.max_tokens
        )

    # ── Public query ──────────────────────────────────────────────────────────

    def query(self, question: str) -> RAGResponse:
        logger.info("Processing query: %s...", question[:100])

        if self._is_greeting(question):
            return RAGResponse(
                answer=self._generate_conversational(question),
                citations=[],
                confidence=1.0,
                context_used="",
                metadata={"greeting": True},
            )

        search_results, lang = self._retrieve(question)
        grounded, reason = self._is_grounded_query(search_results)

        if not grounded:
            logger.info("Not grounded (%s)", reason)
            return RAGResponse(
                answer=self._document_only_response(question),
                citations=[],
                confidence=0.0,
                context_used="",
                metadata={"retrieval_count": len(search_results), "grounding_reason": reason},
            )

        context, citations = self._build_context_and_citations(search_results)
        avg_score = sum(r.score for r in search_results) / len(search_results)

        answer = self._generate(lang=lang, context=context, question=question)

        if not self._post_validate(answer, avg_score, context):
            logger.warning("Post-validation failed")
            return RAGResponse(
                answer=self._document_only_response(question),
                citations=[],
                confidence=0.0,
                context_used="",
                metadata={"retrieval_count": len(search_results), "source": "post_validate_failed"},
            )

        logger.info("Answer ready — %d citations (avg=%.3f)", len(citations), avg_score)
        return RAGResponse(
            answer=answer,
            citations=citations,
            confidence=avg_score,
            context_used=context,
            metadata={
                "retrieval_count": len(search_results),
                "top_score": search_results[0].score,
                "model": self.settings.llm_model,
            },
        )

    # ── Streaming query ───────────────────────────────────────────────────────

    def stream_query(self, question: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
        if self._is_greeting(question):
            yield "meta", {"citations": [], "confidence": 1.0, "metadata": {"greeting": True}}
            yield "token", {"text": self._generate_conversational(question)}
            yield "done", {}
            return

        search_results, lang = self._retrieve(question)
        grounded, reason = self._is_grounded_query(search_results)

        if not grounded:
            yield "meta", {"citations": [], "confidence": 0.0,
                           "metadata": {"grounding_reason": reason}}
            yield "token", {"text": self._document_only_response(question)}
            yield "done", {}
            return

        context, citations = self._build_context_and_citations(search_results)
        avg_score = sum(r.score for r in search_results) / len(search_results)

        yield "meta", {
            "citations": [
                {"title": c.title, "url": c.url, "score": c.score, "snippet": c.text_snippet}
                for c in citations
            ],
            "confidence": avg_score,
            "metadata": {
                "retrieval_count": len(search_results),
                "top_score": search_results[0].score,
                "model": self.settings.llm_model,
            },
        }

        collected: List[str] = []
        try:
            for chunk in self._stream_generate(lang=lang, context=context, question=question):
                collected.append(chunk)
                yield "token", {"text": chunk}
        except Exception as exc:
            logger.error("Streaming failed: %s", exc)
            yield "token", {"text": _PROMPTS["fallback_response_fr"].format(context=context[:500])}
            yield "done", {}
            return

        full_answer = "".join(collected)
        if not self._post_validate(full_answer, avg_score, context):
            yield "token", {"text": self._document_only_response(question), "replace": True}

        yield "done", {}

    # ── Health check ──────────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        health: Dict[str, Any] = {"vector_store": False, "llm": False, "embedder": False}
        try:
            health["vector_store"] = self.vector_store.health_check()
        except Exception as exc:
            logger.error("Vector store health check failed: %s", exc)
        try:
            self.embedder.embed_query("test")
            health["embedder"] = True
        except Exception as exc:
            logger.error("Embedder health check failed: %s", exc)
        try:
            health["llm"] = self.llm_client.health_check()
        except Exception as exc:
            logger.error("LLM health check failed: %s", exc)
        health["overall"] = all(health.values())
        return health


# ── Singleton ─────────────────────────────────────────────────────────────────

_engine_instance: Optional[RAGEngine] = None


def get_engine(use_mock: bool = False) -> RAGEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngine(use_mock=use_mock)
    return _engine_instance
