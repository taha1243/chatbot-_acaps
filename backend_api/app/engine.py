"""
RAG Engine for Atlas-RAG.
PostgreSQL + pgvector retrieval and OpenAI-compatible generation orchestration.
"""
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from .config import Settings, get_settings

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
    PORTAL_FILE_AR = "guide_portal_acaps_ar.md"

    SYSTEM_PROMPT_FR = """Tu es l'assistant officiel du portail ACAPS. Tu reponds TOUJOURS en francais uniquement.

REGLE DE FORMAT :
Pas d'emoji, pas d'icone, pas de symbole decoratif.
Texte brut. Listes numerotees (1. 2. 3.) et tirets (-) autorises.
Markdown gras **...** autorise pour les noms de boutons ou champs.

REGLES :
1. Utilise uniquement les informations du guide ci-dessous.
2. Interprete la question si elle est mal exprimee (tolere les fautes de frappe).
3. Ne demande JAMAIS de clarification sur une date, une periode ou un numero si l'utilisateur ne t'en a pas parle.
4. Reponds directement a l'intention de la question, de maniere simple et orientee action.
5. Si plusieurs etapes : UNE ligne par etape, liste les sous-champs d'un coup sans les re-expliquer.
6. Sois CONCIS : maximum 10 lignes au total. Pas de paraphrase.
7. Si l'information n'existe pas dans le guide, dis-le clairement sans inventer.

Guide du portail ACAPS :
{context}

Question : {question}

Reponse (en francais) :"""

    SYSTEM_PROMPT_AR = """أنت المساعد الرسمي لبوابة ACAPS. أنت تجيب دائمًا بالعربية الفصحى فقط.

قواعد التنسيق:
لا تستخدم أي رمز تعبيري أو أيقونة أو رمز زخرفي.
نص عادي فقط. القوائم المرقمة (1. 2. 3.) والشرطات (-) مسموح بها.
الكتابة بخط عريض **...** مسموح بها لأسماء الأزرار أو الحقول.

القواعد:
1. استخدم فقط المعلومات الواردة في الدليل أدناه.
2. فسّر السؤال إذا كان معبراً عنه بشكل سيء (تسامح مع الأخطاء الإملائية).
3. لا تطلب أبداً توضيحاً بشأن تاريخ أو فترة أو رقم إذا لم يذكره المستخدم.
4. أجب مباشرة على قصد السؤال، بطريقة بسيطة وموجهة نحو الفعل.
5. إذا كانت هناك عدة خطوات: سطر واحد لكل خطوة، اذكر الحقول الفرعية دفعة واحدة دون إعادة شرحها.
6. كن مختصراً: 10 أسطر كحد أقصى. لا إعادة صياغة.
7. إذا لم تكن المعلومة في الدليل، قل ذلك بوضوح دون اختراع.

دليل بوابة ACAPS:
{context}

السؤال: {question}

الجواب (بالعربية):"""

    DARIJA_MAP = {
        "ndir": "faire",
        "n9der": "je peux",
        "n9eder": "je peux",
        "nkdar": "je peux",
        "bghit": "je veux",
        "bghina": "nous voulons",
        "sifet": "envoyer",
        "tsifet": "envoye",
        "mcha": "parti",
        "fin": "ou",
        "wach": "est-ce que",
        "kifach": "comment",
        "kif": "comment",
        "chno": "quoi",
        "3lach": "pourquoi",
        "mnin": "depuis quand",
        "chikaya": "reclamation",
        "chikayah": "reclamation",
        "chikayat": "reclamation",
        "plainte": "reclamation",
        "mouchkil": "probleme",
        "mochkil": "probleme",
        "jawab": "reponse",
        "rdoud": "reponse",
        "numero": "numero",
        "ref": "reference",
        "شكاية": "reclamation",
        "شكوى": "reclamation",
        "كيفاش": "comment",
        "ندير": "faire",
        "وين": "ou",
        "واش": "est-ce que",
        "ضاعت": "perdu",
        "متابعة": "suivi",
        "تقديم": "soumettre",
        "إغلاق": "cloturer",
        "مرجع": "reference",
        "مشكل": "probleme",
        "رد": "reponse",
    }

    INTENT_KEYWORDS: Dict[str, List[str]] = {
        "submit": [
            "soumettre",
            "deposer",
            "déposer",
            "creer",
            "créer",
            "nouvelle reclamation",
            "nouvelle réclamation",
            "nouveau",
            "ouvrir",
            "faire une reclamation",
            "faire une réclamation",
            "comment faire",
            "ndir",
            "bghit ndir",
            "submit",
            "plainte",
            "chikaya",
            "شكاية",
            "envoyer reclamation",
            "envoyer réclamation",
            "deposer reclamation",
            "déposer réclamation",
            "تقديم",
        ],
        "track": [
            "suivre",
            "suivi",
            "statut",
            "etat",
            "état",
            "avancement",
            "ma reclamation",
            "ma réclamation",
            "reference",
            "référence",
            "ou en est",
            "où en est",
            "reponse",
            "réponse",
            "delai",
            "délai",
            "combien de temps",
            "متابعة",
            "jawab",
            "rdoud",
            "ma ref",
            "mon numero",
            "mon numéro",
        ],
        "close": [
            "cloturer",
            "clôturer",
            "fermer",
            "reouvrir",
            "réouvrir",
            "rouvrir",
            "cloture",
            "clôture",
            "ferme",
            "fermé",
            "close",
            "إغلاق",
            "reouverture",
            "réouverture",
            "reactiver",
            "réactiver",
        ],
        "satisfaction": [
            "satisfaction",
            "questionnaire",
            "avis",
            "evaluation",
            "évaluation",
            "noter",
            "note",
            "sondage",
            "enquete",
            "enquête",
        ],
    }

    SECTION_HINTS = {
        "submit": "Soumettre une reclamation",
        "track": "Suivre une reclamation",
        "close": "Cloturer reouvrir reclamation",
        "satisfaction": "Questionnaire satisfaction",
    }

    SECTION_HINTS_AR = {
        "submit": "تقديم شكاية",
        "track": "متابعة شكاية",
        "close": "إغلاق وإعادة فتح شكاية",
        "satisfaction": "استبيان الرضا",
    }

    INTENT_SECTION_MARKERS = {
        "submit": ["Section 1", "القسم 1"],
        "track": ["Section 2", "القسم 2"],
        "close": ["Section 3", "القسم 3"],
        "satisfaction": ["Section 4", "القسم 4"],
    }

    MAX_CHUNK_CHARS_FOR_PROMPT = 800

    HALLUCINATION_INDICATORS = [
        "je pense que",
        "je crois que",
        "probablement",
        "il est possible que",
        "generalement",
        "généralement",
        "je suppose",
        "a ma connaissance",
        "à ma connaissance",
        "je ne suis pas certain",
        "il me semble",
        "d'habitude",
        "i think",
        "i believe",
        "probably",
        "might be",
        "could be",
        "as far as i know",
    ]

    PORTAL_SCOPE_KEYWORDS = {
        "acaps",
        "portail",
        "portal",
        "reclamation",
        "réclamation",
        "reclamations",
        "réclamations",
        "plainte",
        "plaintes",
        "soumettre",
        "deposer",
        "déposer",
        "suivre",
        "suivi",
        "statut",
        "reference",
        "référence",
        "references",
        "références",
        "reponse",
        "réponse",
        "reouvrir",
        "réouvrir",
        "cloturer",
        "clôturer",
        "questionnaire",
        "satisfaction",
        "email",
        "e-mail",
        "telephone",
        "téléphone",
        "document",
        "documents",
        "piece",
        "pièce",
        "pieces",
        "pièces",
        "jointe",
        "jointes",
        "chikaya",
        "chikayat",
        "شكاية",
        "شكايات",
        "متابعة",
        "تقديم",
        "إغلاق",
        "اغلاق",
        "إعادة",
        "استبيان",
        "الرضا",
        "مرجع",
        "وثيقة",
        "وثائق",
    }
    STRICT_GROUNDING_SCORE = 0.45

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

    @property
    def llm_client(self):
        if self._llm_client is None:
            if self.use_mock:
                self._llm_client = MockLLMClient()
            else:
                import httpx
                from openai import OpenAI

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

                self._embedder = MockEmbeddingGenerator(dimension=self.settings.embedding_dimension)
            else:
                from data_ingestion.embedder import EmbeddingGenerator

                self._embedder = EmbeddingGenerator(model_name=self.settings.embedding_model)
        return self._embedder

    def _detect_language(self, text: str) -> str:
        """Route to the Arabic guide when Arabic script dominates the question."""
        if not text:
            return "fr"
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        return "ar" if arabic_chars > len(text) * 0.3 else "fr"

    def _portal_file(self, question: str) -> str:
        return self.PORTAL_FILE_AR if self._detect_language(question) == "ar" else self.PORTAL_FILE

    def _is_greeting(self, text: str) -> bool:
        greeting_keywords = [
            "bonjour",
            "salut",
            "hello",
            "hi",
            "coucou",
            "salam",
            "bonsoir",
            "hey",
            "مرحبا",
            "السلام",
        ]
        clean = text.strip().lower()
        return any(keyword in clean for keyword in greeting_keywords)

    def _normalize_query(self, question: str) -> str:
        """
        Append French equivalents for Darija and Arabic terms so embeddings
        capture portal intent from mixed-language queries.
        """
        question_lower = question.lower()
        extras = [
            french
            for darija, french in self.DARIJA_MAP.items()
            if darija in question_lower
        ]
        if extras:
            normalized = f"{question} {' '.join(extras)}"
            logger.info("Query normalized: '%s' -> '%s'", question[:80], normalized[:120])
            return normalized
        return question

    def _classify_intent(self, question: str) -> Optional[str]:
        question_lower = question.lower()
        for intent, keywords in self.INTENT_KEYWORDS.items():
            if any(keyword in question_lower for keyword in keywords):
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

    def _extract_terms(self, text: str) -> Set[str]:
        terms = set()
        for match in re.findall(r"[\wÀ-ÿ\u0600-\u06FF]+", text.lower()):
            if len(match) < 3:
                continue
            terms.add(match)
        return terms

    def _question_has_portal_scope(self, question: str, intent: Optional[str]) -> bool:
        if intent:
            return True
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in self.PORTAL_SCOPE_KEYWORDS)

    def _has_context_overlap(self, question: str, search_results: List[Any]) -> bool:
        if not search_results:
            return False

        question_terms = self._extract_terms(self._normalize_query(question))
        if not question_terms:
            return False

        context_blob = " ".join(
            f"{result.header_path} {result.text[:300]}"
            for result in search_results
        )
        context_terms = self._extract_terms(context_blob)
        overlap = question_terms.intersection(context_terms)
        return bool(overlap)

    def _is_grounded_query(
        self,
        question: str,
        intent: Optional[str],
        search_results: List[Any],
    ) -> Tuple[bool, str]:
        if not search_results:
            return False, "no_search_results"

        if self._question_has_portal_scope(question, intent):
            return True, "portal_scope"

        avg_score = sum(result.score for result in search_results) / len(search_results)
        if avg_score >= self.STRICT_GROUNDING_SCORE and self._has_context_overlap(question, search_results):
            return True, "strong_retrieval_overlap"

        return False, "outside_portal_scope"

    def _build_document_only_response(self, question: str) -> str:
        lang = self._detect_language(question)
        if lang == "ar":
            return (
                "لا أجد هذه المعلومة في الوثائق المدمجة الخاصة ببوابة ACAPS. "
                "أنا أجيب فقط اعتمادا على هذه الوثائق. "
                "إذا كان سؤالك يتعلق بتقديم شكاية أو تتبعها أو إغلاقها أو إعادة فتحها "
                "أو استبيان الرضا، فأعد صياغته في هذا النطاق."
            )

        return (
            "Je ne trouve pas cette information dans les documents ingestes du portail ACAPS. "
            "Je reponds uniquement a partir de ces documents. "
            "Si votre question concerne le depot, le suivi, la cloture ou la reouverture "
            "d'une reclamation, ou le questionnaire de satisfaction, reformulez-la dans ce cadre."
        )

    def query(self, question: str) -> RAGResponse:
        logger.info("Processing query: %s...", question[:100])

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

        normalized_question = self._normalize_query(question)
        intent = self._classify_intent(question)
        lang = self._detect_language(question)

        query_text_for_search = normalized_question
        if intent:
            hints_map = self.SECTION_HINTS_AR if lang == "ar" else self.SECTION_HINTS
            hint = hints_map.get(intent, "")
            if hint:
                query_text_for_search = f"{normalized_question} {hint}"
                logger.info("Search query augmented with hint (%s): '%s'", lang, hint)

        portal_file = self._portal_file(question)
        query_embedding = self.embedder.embed_query(normalized_question)
        search_results = self.vector_store.hybrid_search(
            query_embedding=query_embedding,
            query_text=query_text_for_search,
            top_k=self.settings.top_k_results,
            score_threshold=self.settings.similarity_threshold,
            keyword_boost=0.3,
            file_name=portal_file,
        )

        if intent:
            markers = self.INTENT_SECTION_MARKERS.get(intent) or []
            if markers:
                scoped = [
                    result
                    for result in (search_results or [])
                    if any(marker in result.header_path for marker in markers)
                ]
                if scoped:
                    logger.info(
                        "Intent-scoped filter (%s -> %s): kept %d/%d chunks",
                        intent,
                        markers,
                        len(scoped),
                        len(search_results),
                    )
                    search_results = scoped
                else:
                    marker = next(
                        (
                            m
                            for m in markers
                            if (lang == "ar") == ("القسم" in m)
                        ),
                        markers[0],
                    )
                    kw_results = self.vector_store.search_by_keyword(
                        keyword=marker,
                        field="header_path",
                        top_k=self.settings.top_k_results,
                    )
                    kw_results = [result for result in kw_results if result.file_name == portal_file]
                    if kw_results:
                        logger.info(
                            "Intent-scoped fallback (%s -> %s): pulled %d chunks by header_path",
                            intent,
                            marker,
                            len(kw_results),
                        )
                        search_results = kw_results

        grounded_query, grounding_reason = self._is_grounded_query(
            question=question,
            intent=intent,
            search_results=search_results,
        )
        if not grounded_query:
            logger.info("Question outside grounded document scope (%s)", grounding_reason)
            return RAGResponse(
                answer=self._build_document_only_response(question),
                citations=[],
                confidence=0.0,
                context_used="",
                metadata={
                    "retrieval_count": len(search_results),
                    "source": "document_only_no_match",
                    "intent": intent,
                    "grounding_reason": grounding_reason,
                },
            )

        context_parts = []
        citations: List[Citation] = []
        for result in search_results:
            trimmed = result.text[: self.MAX_CHUNK_CHARS_FOR_PROMPT]
            if len(result.text) > self.MAX_CHUNK_CHARS_FOR_PROMPT:
                trimmed += "..."
            context_parts.append(f"[Source: {result.header_path}]\n{trimmed}")
            url = result.full_url if result.full_url and not result.full_url.startswith("#") else ""
            snippet = result.text[:400] + "..." if len(result.text) > 400 else result.text
            citations.append(
                Citation(
                    title=result.header_path,
                    url=url,
                    score=result.score,
                    text_snippet=snippet,
                )
            )

        context = "\n\n---\n\n".join(context_parts)
        avg_score = sum(result.score for result in search_results) / len(search_results)

        prompt_template = self.SYSTEM_PROMPT_AR if lang == "ar" else self.SYSTEM_PROMPT_FR
        prompt = prompt_template.format(context=context, question=question)
        answer = self._generate(prompt, lang=lang, context=context)

        if not self._post_validate(answer, avg_score):
            logger.warning("Post-validation failed -> returning strict document-only answer")
            return RAGResponse(
                answer=self._build_document_only_response(question),
                citations=[],
                confidence=0.0,
                context_used="",
                metadata={
                    "retrieval_count": len(search_results),
                    "source": "document_only_unverified",
                    "intent": intent,
                },
            )

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

    def stream_query(self, question: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """
        Token-by-token variant of `query`.

        Yields `(event, payload)` tuples:
          - ("meta",  {"citations": [...], "confidence": float, "metadata": {...}})
          - ("token", {"text": str})
          - ("done",  {})
          - ("error", {"message": str})

        Retrieval and grounding logic mirror `query` so the streamed answer is
        built from the same context. Non-streaming paths (greetings, off-scope,
        fallback) emit the whole answer as a single token event.
        """
        logger.info("Processing streaming query: %s...", question[:100])

        if self._is_greeting(question):
            prompt = (
                "You are the official assistant for the ACAPS portal in Morocco. "
                "The user has just greeted you. Respond politely in the user's language "
                "(French, Arabic or Darija), welcome them and offer help with the portal. "
                f"User message: '{question}'"
            )
            yield "meta", {
                "citations": [],
                "confidence": 1.0,
                "metadata": {"greeting": True, "model": "conversational-llm"},
            }
            for chunk in self._stream_conversational(prompt):
                yield "token", {"text": chunk}
            yield "done", {}
            return

        normalized_question = self._normalize_query(question)
        intent = self._classify_intent(question)
        lang = self._detect_language(question)

        query_text_for_search = normalized_question
        if intent:
            hints_map = self.SECTION_HINTS_AR if lang == "ar" else self.SECTION_HINTS
            hint = hints_map.get(intent, "")
            if hint:
                query_text_for_search = f"{normalized_question} {hint}"

        portal_file = self._portal_file(question)
        query_embedding = self.embedder.embed_query(normalized_question)
        search_results = self.vector_store.hybrid_search(
            query_embedding=query_embedding,
            query_text=query_text_for_search,
            top_k=self.settings.top_k_results,
            score_threshold=self.settings.similarity_threshold,
            keyword_boost=0.3,
            file_name=portal_file,
        )

        if intent:
            markers = self.INTENT_SECTION_MARKERS.get(intent) or []
            if markers:
                scoped = [
                    result
                    for result in (search_results or [])
                    if any(marker in result.header_path for marker in markers)
                ]
                if scoped:
                    search_results = scoped
                else:
                    marker = next(
                        (m for m in markers if (lang == "ar") == ("القسم" in m)),
                        markers[0],
                    )
                    kw_results = self.vector_store.search_by_keyword(
                        keyword=marker,
                        field="header_path",
                        top_k=self.settings.top_k_results,
                    )
                    kw_results = [r for r in kw_results if r.file_name == portal_file]
                    if kw_results:
                        search_results = kw_results

        grounded_query, grounding_reason = self._is_grounded_query(
            question=question,
            intent=intent,
            search_results=search_results,
        )
        if not grounded_query:
            yield "meta", {
                "citations": [],
                "confidence": 0.0,
                "metadata": {
                    "retrieval_count": len(search_results),
                    "source": "document_only_no_match",
                    "intent": intent,
                    "grounding_reason": grounding_reason,
                },
            }
            yield "token", {"text": self._build_document_only_response(question)}
            yield "done", {}
            return

        context_parts = []
        citations: List[Citation] = []
        for result in search_results:
            trimmed = result.text[: self.MAX_CHUNK_CHARS_FOR_PROMPT]
            if len(result.text) > self.MAX_CHUNK_CHARS_FOR_PROMPT:
                trimmed += "..."
            context_parts.append(f"[Source: {result.header_path}]\n{trimmed}")
            url = result.full_url if result.full_url and not result.full_url.startswith("#") else ""
            snippet = result.text[:400] + "..." if len(result.text) > 400 else result.text
            citations.append(
                Citation(
                    title=result.header_path,
                    url=url,
                    score=result.score,
                    text_snippet=snippet,
                )
            )

        context = "\n\n---\n\n".join(context_parts)
        avg_score = sum(result.score for result in search_results) / len(search_results)

        prompt_template = self.SYSTEM_PROMPT_AR if lang == "ar" else self.SYSTEM_PROMPT_FR
        prompt = prompt_template.format(context=context, question=question)

        yield "meta", {
            "citations": [
                {
                    "title": c.title,
                    "url": c.url,
                    "score": c.score,
                    "snippet": c.text_snippet,
                }
                for c in citations
            ],
            "confidence": avg_score,
            "metadata": {
                "retrieval_count": len(search_results),
                "top_score": search_results[0].score,
                "model": self.settings.vllm_model,
                "intent": intent,
            },
        }

        collected = []
        try:
            for chunk in self._stream_generate(prompt, lang=lang):
                collected.append(chunk)
                yield "token", {"text": chunk}
        except Exception as exc:
            logger.error("Streaming generation failed: %s", exc)
            if self.settings.llm_fallback_enabled and context:
                yield "token", {"text": self._generate_fallback_response(context)}
                yield "done", {}
                return
            yield "error", {"message": "llm_unavailable"}
            return

        full_answer = "".join(collected)
        if not self._post_validate(full_answer, avg_score):
            logger.warning("Post-validation failed on streamed answer")
            yield "meta", {
                "citations": [],
                "confidence": 0.0,
                "metadata": {
                    "retrieval_count": len(search_results),
                    "source": "document_only_unverified",
                    "intent": intent,
                    "replace_answer": True,
                },
            }
            yield "token", {"text": self._build_document_only_response(question), "replace": True}

        yield "done", {}

    SYS_MSG_FR = (
        "Tu es l'assistant officiel du portail ACAPS. "
        "Tu reponds UNIQUEMENT en francais. Ne jamais utiliser une autre langue. "
        "Pas d'emoji. Texte brut avec listes numerotees ou tirets. "
        "Utilise uniquement les informations du guide fourni."
    )
    SYS_MSG_AR = (
        "أنت المساعد الرسمي لبوابة ACAPS. "
        "تجيب فقط بالعربية الفصحى. لا تستخدم أي لغة أخرى أبداً. "
        "لا تستخدم الرموز التعبيرية. نص عادي مع قوائم مرقمة أو شرطات. "
        "استخدم فقط المعلومات الواردة في الدليل المقدم."
    )

    def _sys_msg(self, lang: str) -> str:
        return self.SYS_MSG_AR if lang == "ar" else self.SYS_MSG_FR

    def _stream_generate(self, prompt: str, lang: str = "fr") -> Iterator[str]:
        if self.use_mock:
            yield self.llm_client.generate(prompt)
            return

        stream = self.llm_client.chat.completions.create(
            model=self.settings.vllm_model,
            messages=[
                {"role": "system", "content": self._sys_msg(lang)},
                {"role": "user", "content": prompt},
            ],
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            stream=True,
        )
        for event in stream:
            delta = event.choices[0].delta.content if event.choices else None
            if delta:
                yield delta

    def _stream_conversational(self, prompt: str) -> Iterator[str]:
        if self.use_mock:
            yield "Bonjour, je suis l'assistant ACAPS. Comment puis-je vous aider avec le portail ?"
            return

        try:
            stream = self.llm_client.chat.completions.create(
                model=self.settings.vllm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "CRITICAL LANGUAGE RULE: Detect the language of the user's question "
                            "and answer ENTIRELY in that same language. "
                            "If Arabic, answer in Arabic script. If French, answer in French. "
                            "If Darija, answer in Darija. Never translate, never switch language.\n\n"
                            "FORMAT RULE: NO emojis, NO icons, NO decorative symbols. Plain text only.\n\n"
                            "You are the official ACAPS portal assistant. Be concise and helpful."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=self.settings.max_tokens,
                stream=True,
            )
            for event in stream:
                delta = event.choices[0].delta.content if event.choices else None
                if delta:
                    yield delta
        except Exception as exc:
            logger.error("Error streaming conversational response: %s", exc)
            yield "Bonjour, je suis l'assistant ACAPS. Comment puis-je vous aider avec le portail ?"

    def _generate(self, prompt: str, lang: str = "fr", context: str = "") -> str:
        if self.use_mock:
            return self.llm_client.generate(prompt)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.settings.vllm_model,
                messages=[
                    {"role": "system", "content": self._sys_msg(lang)},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            if self.settings.llm_fallback_enabled and context:
                return self._generate_fallback_response(context)
            return "Je ne peux pas generer une reponse pour le moment. Veuillez reessayer plus tard."

    def _generate_conversational(self, prompt: str) -> str:
        if self.use_mock:
            return "Bonjour, je suis l'assistant ACAPS. Comment puis-je vous aider avec le portail ?"

        try:
            response = self.llm_client.chat.completions.create(
                model=self.settings.vllm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "CRITICAL LANGUAGE RULE: Detect the language of the user's question "
                            "and answer ENTIRELY in that same language. "
                            "If Arabic, answer in Arabic script. If French, answer in French. "
                            "If Darija, answer in Darija. Never translate, never switch language.\n\n"
                            "FORMAT RULE: NO emojis, NO icons, NO decorative symbols. Plain text only.\n\n"
                            "You are the official ACAPS portal assistant. Be concise and helpful."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=self.settings.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as exc:
            logger.error("Error generating conversational response: %s", exc)
            return "Bonjour, je suis l'assistant ACAPS. Comment puis-je vous aider avec le portail ?"

    def _generate_fallback_response(self, context: str) -> str:
        """Return raw context when the LLM is unavailable."""
        if "[Source:" in context:
            parts = context.split("---")
            first_source = parts[0].strip() if parts else context[:500]
        else:
            first_source = context[:500]
        return (
            "Note : le serveur LLM est temporairement indisponible. "
            "Voici les informations trouvees dans le guide :\n\n"
            f"{first_source}\n\n"
            "Pour une reponse complete, veuillez reessayer dans quelques instants."
        )

    def health_check(self) -> Dict[str, Any]:
        health = {"vector_store": False, "llm": False, "embedder": False}

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
            if self.use_mock:
                health["llm"] = True
            else:
                self.llm_client.models.list()
                health["llm"] = True
        except Exception as exc:
            logger.error("LLM health check failed: %s", exc)

        health["overall"] = all(health.values())
        return health


class MockLLMClient:
    def generate(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "sick leave" in prompt_lower or "vacation" in prompt_lower or "cong" in prompt_lower:
            return "Article 5 states that employees are entitled to 22 days of annual leave."
        if "reclamation" in prompt_lower or "soumettre" in prompt_lower or "réclamation" in prompt_lower:
            return "Pour soumettre une reclamation, accedez au portail ACAPS et cliquez sur 'Nouvelle reclamation'."
        return "Base sur la documentation disponible, veuillez consulter le guide du portail ACAPS."


_engine_instance: Optional[RAGEngine] = None


def get_engine(use_mock: bool = False) -> RAGEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngine(use_mock=use_mock)
    return _engine_instance
