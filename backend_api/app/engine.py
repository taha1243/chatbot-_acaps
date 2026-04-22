"""
RAG Engine for Atlas-RAG
PostgreSQL + pgvector retrieval and OpenAI-compatible generation orchestration.
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .config import get_settings, Settings

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A citation for a source document."""
    title: str
    url: str
    score: float
    text_snippet: str


@dataclass
class RAGResponse:
    """Response from the RAG engine."""
    answer: str
    citations: List[Citation]
    confidence: float
    context_used: str
    metadata: Dict[str, Any]


class RAGEngine:
    """
    Main RAG engine orchestrating retrieval and generation.
    Uses pgvector-backed retrieval and an OpenAI-compatible API for generation.
    """

    PORTAL_FILE = "guide_portal_acaps.md"

    SYSTEM_PROMPT = """Tu es l'assistant officiel du portail ACAPS.

OBJECTIF :
Aider l'utilisateur à utiliser le portail (soumettre, suivre, gérer une réclamation).

RÈGLES :
1. Utilise uniquement les informations du guide fourni ci-dessous.
2. Reformule et interprète la question si elle est mal exprimée (français, arabe ou darija).
3. Si la question est ambiguë ou incomplète, pose une question de clarification courte.
4. Donne des réponses simples, structurées et orientées action.
5. Si plusieurs étapes sont nécessaires, réponds étape par étape.
6. Si l'information n'existe pas dans le guide, dis-le clairement sans inventer.
7. Ne jamais inventer d'informations absentes du guide.

COMPORTEMENT :
- Si l'utilisateur veut faire une action → guide-le étape par étape.
- Si l'utilisateur a un problème → propose une solution basée sur le guide.
- Si la question est partiellement liée → donne les éléments utiles du guide.

LANGUE :
- Réponds dans la langue de l'utilisateur : français, arabe, ou darija.

Guide du portail ACAPS :
{context}

Question : {question}

Réponse :"""

    GENERAL_PROMPT = """Tu es l'assistant du portail ACAPS.

La question ne correspond pas directement au guide du portail.

Réponds de manière utile et naturelle :
1. Explique brièvement que tu es spécialisé dans l'utilisation du portail ACAPS.
2. Propose ce que tu peux faire :
   - Aider à soumettre une réclamation (Section 1)
   - Aider à suivre une réclamation (Section 2)
   - Expliquer comment clôturer ou réouvrir une réclamation (Section 3)
   - Donner des informations sur le questionnaire de satisfaction (Section 4)
3. Invite l'utilisateur à reformuler sa demande.

Question : {question}

Réponse :"""

    # Darija / Arabic → French keyword mapping for query normalization
    DARIJA_MAP = {
        # Verbs
        "ndir": "faire", "n9der": "je peux", "n9eder": "je peux",
        "nkdar": "je peux", "bghit": "je veux", "bghina": "nous voulons",
        "sifet": "envoyer", "tsifet": "envoyé", "mcha": "parti",
        # Question words
        "fin": "où", "wach": "est-ce que", "kifach": "comment",
        "kif": "comment", "chno": "quoi", "3lach": "pourquoi",
        "mnin": "depuis quand",
        # Nouns
        "chikaya": "réclamation", "chikayah": "réclamation",
        "chikayat": "réclamation", "plainte": "réclamation",
        "mouchkil": "problème", "mochkil": "problème",
        "jawab": "réponse", "rdoud": "réponse",
        "numero": "numéro", "ref": "référence",
        # Arabic
        "شكاية": "réclamation", "شكوى": "réclamation",
        "كيفاش": "comment", "ندير": "faire", "وين": "où",
        "واش": "est-ce que", "ضاعت": "perdu",
        "متابعة": "suivi", "تقديم": "soumettre",
        "إغلاق": "clôturer", "مرجع": "référence",
        "مشكل": "problème", "رد": "réponse",
    }

    # Keywords → intent mapping (section hint for retrieval boost)
    INTENT_KEYWORDS: Dict[str, List[str]] = {
        "submit": [
            "soumettre", "déposer", "créer", "nouvelle réclamation", "nouveau",
            "ouvrir", "faire une réclamation", "comment faire", "ndir",
            "bghit ndir", "submit", "plainte", "chikaya", "شكاية",
            "envoyer réclamation", "déposer réclamation",
        ],
        "track": [
            "suivre", "suivi", "statut", "état", "avancement", "ma réclamation",
            "référence", "où en est", "réponse", "délai", "combien de temps",
            "متابعة", "jawab", "rdoud", "ma réf", "mon numéro",
        ],
        "close": [
            "clôturer", "fermer", "réouvrir", "rouvrir", "clôture",
            "fermé", "close", "إغلاق", "réouverture", "réactiver",
        ],
        "satisfaction": [
            "satisfaction", "questionnaire", "avis", "évaluation",
            "noter", "note", "sondage", "enquête",
        ],
    }

    SECTION_HINTS = {
        "submit": "Soumettre une réclamation",
        "track": "Suivre une réclamation",
        "close": "Clôturer réouvrir réclamation",
        "satisfaction": "Questionnaire satisfaction",
    }

    HALLUCINATION_INDICATORS = [
        "je pense que", "je crois que", "probablement", "il est possible que",
        "généralement", "je suppose", "à ma connaissance",
        "je ne suis pas certain", "il me semble", "d'habitude",
    ]

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
        logger.info(f"RAGEngine initialized (mock={use_mock})")

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
    #  Pre-processing                                                       #
    # ------------------------------------------------------------------ #

    def _is_greeting(self, text: str) -> bool:
        greeting_keywords = [
            "bonjour", "salut", "hello", "hi", "coucou", "salam", "holla",
            "bonsoir", "good morning", "good afternoon", "good evening",
            "hey", "yo", "hola", "ahlan", "مرحبا", "السلام",
        ]
        clean = text.strip().lower()
        return any(kw in clean for kw in greeting_keywords)

    def _normalize_query(self, question: str) -> str:
        """
        Append French equivalents for darija/Arabic tokens so the embedding
        captures intent even when the user writes in mixed language.
        """
        q_lower = question.lower()
        extras = [french for darija, french in self.DARIJA_MAP.items() if darija in q_lower]
        if extras:
            normalized = f"{question} {' '.join(extras)}"
            logger.info("Query normalized: '%s' → '%s'", question[:80], normalized[:120])
            return normalized
        return question

    def _classify_intent(self, question: str) -> Optional[str]:
        """Return intent name or None, using keyword matching."""
        q_lower = question.lower()
        for intent, keywords in self.INTENT_KEYWORDS.items():
            if any(kw in q_lower for kw in keywords):
                logger.info("Intent classified: %s", intent)
                return intent
        return None

    def _post_validate(self, answer: str, confidence: float) -> bool:
        """Return False if the answer looks hallucinated or empty."""
        if len(answer.strip()) < 20:
            return False
        answer_lower = answer.lower()
        for indicator in self.HALLUCINATION_INDICATORS:
            if indicator in answer_lower:
                logger.warning("Hallucination indicator detected: '%s'", indicator)
                return False
        return True

    # ------------------------------------------------------------------ #
    #  Main query pipeline                                                  #
    # ------------------------------------------------------------------ #

    def query(self, question: str) -> RAGResponse:
        logger.info("Processing query: %s...", question[:100])

        # 0. Greeting fast-path
        if self._is_greeting(question):
            logger.info("Detected greeting, returning conversational response")
            prompt = (
                "You are the official assistant for the ACAPS portal in Morocco. "
                "The user has just greeted you. Respond politely in the user's language "
                "(French, Arabic or Darija), welcome them and offer help with the portal. "
                f"User message: '{question}'"
            )
            return RAGResponse(
                answer=self._generate_conversational(prompt),
                citations=[],
                confidence=1.0,
                context_used="",
                metadata={"greeting": True, "model": "conversational-llm"},
            )

        # 1. Pre-process: normalize darija/mixed language
        normalized_question = self._normalize_query(question)

        # 2. Intent classification → section hint for retrieval boost
        intent = self._classify_intent(question)
        query_text_for_search = normalized_question
        if intent:
            hint = self.SECTION_HINTS.get(intent, "")
            if hint:
                query_text_for_search = f"{normalized_question} {hint}"
                logger.info("Search query augmented with hint: '%s'", hint)

        # 3. Embed & retrieve
        query_embedding = self.embedder.embed_query(normalized_question)
        search_results = self.vector_store.hybrid_search(
            query_embedding=query_embedding,
            query_text=query_text_for_search,
            top_k=self.settings.top_k_results,
            score_threshold=self.settings.similarity_threshold,
            keyword_boost=0.3,
            file_name=self.PORTAL_FILE,
        )

        # No results → intelligent redirect (not a rejection)
        if not search_results:
            logger.info("No relevant chunks found — redirecting via GENERAL_PROMPT")
            prompt = self.GENERAL_PROMPT.format(question=question)
            answer = self._generate_conversational(prompt)
            return RAGResponse(
                answer=answer,
                citations=[],
                confidence=0.0,
                context_used="",
                metadata={"retrieval_count": 0, "source": "general_redirect", "intent": intent},
            )

        # 4. Build context
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

        # 5. Generate answer
        prompt = self.SYSTEM_PROMPT.format(context=context, question=question)
        answer = self._generate(prompt, context=context)

        # 6. Post-validation: fall back to GENERAL_PROMPT if response looks bad
        if not self._post_validate(answer, avg_score):
            logger.warning("Post-validation failed — falling back to GENERAL_PROMPT")
            fallback_prompt = self.GENERAL_PROMPT.format(question=question)
            answer = self._generate_conversational(fallback_prompt)
            citations = []

        logger.info("Generated answer with %d citations (avg_score=%.3f)", len(citations), avg_score)

        return RAGResponse(
            answer=answer,
            citations=citations,
            confidence=avg_score,
            context_used=context,
            metadata={
                "retrieval_count": len(search_results),
                "top_score": search_results[0].score,
                "model": self.settings.vllm_model,
                "intent": intent,
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
                    {
                        "role": "system",
                        "content": (
                            "Tu es un assistant officiel ACAPS. "
                            "Réponds strictement dans la langue de la question (français, arabe ou darija). "
                            "N'invente aucune information absente du guide."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            if self.settings.llm_fallback_enabled and context:
                return self._generate_fallback_response(context)
            return "Je ne peux pas générer une réponse pour le moment. Veuillez réessayer plus tard."

    def _generate_conversational(self, prompt: str) -> str:
        if self.use_mock:
            return "Bonjour! Je suis l'assistant ACAPS. Comment puis-je vous aider aujourd'hui?"
        try:
            response = self.llm_client.chat.completions.create(
                model=self.settings.vllm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu es l'assistant officiel du portail ACAPS. "
                            "Réponds dans la langue de l'utilisateur (français, arabe ou darija). "
                            "Sois utile, naturel et concis."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=self.settings.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Error generating conversational response: %s", e)
            return "Bonjour! Je suis l'assistant ACAPS. Comment puis-je vous aider ?"

    def _generate_fallback_response(self, context: str) -> str:
        """Return raw context when LLM is unavailable."""
        if "[Source:" in context:
            parts = context.split("---")
            first_source = parts[0].strip() if parts else context[:500]
        else:
            first_source = context[:500]
        return (
            "**Note :** Le serveur LLM est temporairement indisponible. "
            "Voici les informations trouvées dans le guide :\n\n"
            f"{first_source}\n\n"
            "*Pour une réponse complète, veuillez réessayer dans quelques instants.*"
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
