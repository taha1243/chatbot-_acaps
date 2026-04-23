"""
RAG Engine for Atlas-RAG
PostgreSQL + pgvector retrieval and OpenAI-compatible generation orchestration.
"""
import logging
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
    
    SYSTEM_PROMPT = """Tu es un assistant de référence pour l'ACAPS (Autorité de Contrôle des Assurances et de la Prévoyance Sociale) spécialisé dans la documentation interne.

INSTRUCTIONS STRICTES :

Tu dois choisir EXACTEMENT UN des deux modes de réponse ci-dessous — JAMAIS les deux à la fois :

▸ MODE A — Le contexte contient l'information demandée :
  • Réponds uniquement à partir du contexte.
  • Sois précis, factuel et structuré.
  • Cite la source via le chemin d'en-tête fourni (ex. : "Selon [Section 1 — Soumettre une Réclamation > Accès]…").
  • N'AJOUTE JAMAIS de phrase de refus, de disclaimer, ni de mention « je ne trouve pas » — même partielle, même en fin de réponse.

▸ MODE B — Le contexte NE contient PAS l'information demandée :
  • Réponds EXACTEMENT et UNIQUEMENT cette phrase, sans rien ajouter :
    « Je ne trouve pas cette information dans les documents disponibles. »
  • N'invente rien. N'inclus aucune information du contexte.

Règles transverses :
  • Ne fais jamais de suppositions ni d'inventions.
  • Détermine la langue de la question et réponds exclusivement dans cette langue. Si plusieurs langues sont mélangées, privilégie le français.

EXEMPLES :

Exemple 1 — information présente (MODE A) :
Contexte : [Source: Section 1 — Soumettre une Réclamation > Accès]
Pour soumettre une réclamation : 1) Scroller la page principale. 2) Cliquer sur "Soumettre une réclamation".
Question : Comment déposer une réclamation ?
Réponse attendue :
Pour déposer une réclamation :
1. Scroller jusqu'au bas de la page principale.
2. Cliquer sur « Soumettre une réclamation ».
(Source : Section 1 — Soumettre une Réclamation > Accès.)

Exemple 2 — information absente (MODE B) :
Contexte : [Source: Section 3] Les horaires d'ouverture sont de 9h à 17h.
Question : Quel est le montant maximal d'indemnisation ?
Réponse attendue :
Je ne trouve pas cette information dans les documents disponibles.

---

Contexte des documents :
{context}

Question : {question}

Réponse :"""
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        use_mock: bool = False
    ):
        """
        Initialize RAG engine.
        
        Args:
            settings: Configuration settings
            use_mock: Use mock components for testing
        """
        self.settings = settings or get_settings()
        self.use_mock = use_mock
        self._llm_client = None
        self._vector_store = None
        self._embedder = None
        self._greeting_router = None

        logger.info(f"RAGEngine initialized (mock={use_mock})")
    
    @property
    def llm_client(self):
        """Lazy load LLM client."""
        if self._llm_client is None:
            if self.use_mock:
                self._llm_client = MockLLMClient()
            else:
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
                    default_headers=default_headers
                )
        return self._llm_client
    
    @property
    def vector_store(self):
        """Lazy load vector store."""
        if self._vector_store is None:
            if self.use_mock:
                from data_ingestion.vector_store import MockVectorStore
                self._vector_store = MockVectorStore(
                    table_name=self.settings.vector_table,
                    embedding_dimension=self.settings.embedding_dimension
                )
            else:
                from data_ingestion.vector_store import VectorStore
                self._vector_store = VectorStore(
                    dsn=self.settings.database_url,
                    table_name=self.settings.vector_table,
                    embedding_dimension=self.settings.embedding_dimension
                )
        return self._vector_store
    
    @property
    def embedder(self):
        """Lazy load embedder."""
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

    @property
    def greeting_router(self):
        """Lazy load the semantic greeting router."""
        if self._greeting_router is None:
            from .greeting_router import SemanticGreetingRouter
            self._greeting_router = SemanticGreetingRouter(
                embedder_provider=lambda: self.embedder,
                threshold=self.settings.greeting_similarity_threshold,
            )
        return self._greeting_router
    
    def _is_greeting(self, text: str) -> bool:
        """
        Check if the input text is a greeting or small-talk pleasantry
        (e.g. "bonjour", "comment vas-tu ?", "ça va ?", "how are you").
        Tolerates common typos (e.g. "comments vas tu").
        """
        import re

        clean_text = text.strip().lower()

        # Whole-word greetings (avoid matching inside larger words like "hire")
        greeting_words = [
            "bonjour", "salut", "hello", "hi", "coucou", "salam", "holla",
            "bonsoir", "hey", "yo", "hola", "marhaba", "ahlan", "cava",
        ]
        tokens = set(re.findall(r"[\w'’çàâäéèêëîïôöùûüÿñ]+", clean_text))
        if any(word in tokens for word in greeting_words):
            return True

        # Regex patterns for small-talk — tolerant to typos and punctuation
        smalltalk_patterns = [
            r"\bgood\s+(morning|afternoon|evening|night)\b",
            r"\bhow\s+are\s+(you|u)\b",
            r"\bhow'?s\s+it\s+going\b",
            r"\bwhat'?s\s+up\b",
            r"\bcomments?\s+(vas|allez|ca\s+va|ça\s+va)\b",  # "comment(s) vas/allez/ça va"
            r"\bça\s+va\b",
            r"\bca\s+va\b",
            r"\bquoi\s+de\s+neuf\b",
            r"\btout\s+va\s+bien\b",
        ]
        return any(re.search(p, clean_text) for p in smalltalk_patterns)

    def query(self, question: str) -> RAGResponse:
        """
        Process a user question through the RAG pipeline.

        Args:
            question: User's question

        Returns:
            RAGResponse with answer and citations
        """
        logger.info(f"Processing query: {question[:100]}...")

        # --- Greeting / small-talk routing -------------------------------
        # Two-stage: a fast regex pre-filter for obvious cases, then an
        # embedding-similarity fallback that catches paraphrases and typos.
        greeting_detected = False
        greeting_source = None
        greeting_score = 1.0

        if self._is_greeting(question):
            greeting_detected = True
            greeting_source = "keyword"
        elif self.settings.enable_semantic_greeting:
            try:
                is_greet, sim = self.greeting_router.check(question)
                if is_greet:
                    greeting_detected = True
                    greeting_source = "semantic"
                    greeting_score = sim
                    logger.info(f"Semantic greeting match (sim={sim:.3f})")
            except Exception as e:
                # Never let the router break the main pipeline.
                logger.warning(f"Greeting router failed, continuing to RAG: {e}")

        if greeting_detected:
            logger.info(f"Detected greeting ({greeting_source}), returning conversational response")
            greeting_prompt = (
                "You are an AI assistant for ACAPS (Autorité de Contrôle des Assurances et de la Prévoyance Sociale) in Morocco. "
                "The user has just greeted you. "
                "Respond politely in French, welcoming them and offering your help with the internal documentation. "
                "Keep it brief and professional. "
                f"User greeting: '{question}'"
            )
            answer = self._generate_conversational(greeting_prompt)
            return RAGResponse(
                answer=answer,
                citations=[],
                confidence=1.0,
                context_used="",
                metadata={
                    "greeting": True,
                    "greeting_source": greeting_source,
                    "greeting_score": greeting_score,
                    "model": "conversational-llm",
                },
            )

        # Step 1: Embed the query
        query_embedding = self.embedder.embed_query(question)
        
        # Step 2: Retrieve relevant documents using HYBRID search
        # This combines semantic search with keyword matching for specific references
        search_results = self.vector_store.hybrid_search(
            query_embedding=query_embedding,
            query_text=question,
            top_k=self.settings.top_k_results,
            score_threshold=self.settings.similarity_threshold,
            keyword_boost=0.3
        )
        
        if not search_results:
            logger.info("No relevant documents found")
            return RAGResponse(
                answer="I cannot find relevant information in the available documents.",
                citations=[],
                confidence=0.0,
                context_used="",
                metadata={"retrieval_count": 0}
            )
        
        # Step 3: Build context from retrieved documents
        context_parts = []
        citations = []
        
        for result in search_results:
            context_parts.append(
                f"[Source: {result.header_path}]\n{result.text}"
            )
            
            # Use full_url if available, otherwise indicate no link
            url = result.full_url if result.full_url and not result.full_url.startswith("#") else ""
            
            # Longer snippets for better context (400 chars)
            snippet_length = 400
            text_snippet = result.text[:snippet_length] + "..." if len(result.text) > snippet_length else result.text
            
            citations.append(Citation(
                title=result.header_path,
                url=url,
                score=result.score,
                text_snippet=text_snippet
            ))
        
        context = "\n\n---\n\n".join(context_parts)
        avg_score = sum(r.score for r in search_results) / len(search_results)
        
        # Step 4: Generate answer
        prompt = self.SYSTEM_PROMPT.format(context=context, question=question)
        answer = self._generate(prompt, context=context)

        # Step 5: Citation gating + parasitic-refusal cleaning.
        #   • Pure refusal → drop citations (showing sources next to "Je ne
        #     trouve pas..." misleads the user into thinking they back the
        #     answer when they don't).
        #   • Parasitic refusal sentence appended to a real answer → strip it
        #     and keep the citations. The few-shot SYSTEM_PROMPT should
        #     already prevent this, but smaller models still slip occasionally.
        cleaned_answer, is_pure_refusal = self._clean_refusal(answer)

        if is_pure_refusal:
            logger.info("LLM emitted no-info fallback; dropping citations")
            return RAGResponse(
                answer=cleaned_answer,
                citations=[],
                confidence=0.0,
                context_used=context,
                metadata={
                    "retrieval_count": len(search_results),
                    "top_score": search_results[0].score if search_results else 0,
                    "model": self.settings.vllm_model,
                    "citations_gated": True,
                },
            )

        cleaned_flag = cleaned_answer != answer
        if cleaned_flag:
            logger.info("Stripped parasitic refusal sentence from LLM answer")

        logger.info(f"Generated answer with {len(citations)} citations")

        return RAGResponse(
            answer=cleaned_answer,
            citations=citations,
            confidence=avg_score,
            context_used=context,
            metadata={
                "retrieval_count": len(search_results),
                "top_score": search_results[0].score if search_results else 0,
                "model": self.settings.vllm_model,
                "refusal_cleaned": cleaned_flag,
            }
        )

    # Sentences that look like the canned "no information" refusal — in FR/EN
    # and common paraphrases the LLM produces. Used both for pure-refusal
    # detection and for stripping parasitic trailing sentences.
    _REFUSAL_PATTERNS = [
        r"je ne trouve pas cette information[^.!?]*",
        r"je ne dispose pas (?:de|d')[^.!?]*",
        r"aucune information (?:n'est )?disponible[^.!?]*",
        r"(?:l')?information n'est pas (?:présente|disponible|fournie)[^.!?]*",
        r"i (?:cannot|can't|do not|don't) find[^.!?]*",
        r"no (?:relevant )?information (?:is )?(?:available|found)[^.!?]*",
    ]

    @classmethod
    def _clean_refusal(cls, answer: str):
        """
        Inspect the LLM answer for canned refusal sentences.

        Returns (cleaned_answer, is_pure_refusal):
          • is_pure_refusal=True when the answer is essentially only a refusal
            (citations should be dropped by the caller).
          • Otherwise, parasitic refusal sentences are stripped from the
            answer and the cleaned text is returned with is_pure_refusal=False.
        """
        import re

        if not answer or not answer.strip():
            return answer or "", True

        # Split into sentences while preserving non-whitespace content.
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
        if not sentences:
            return answer.strip(), True

        kept = []
        dropped = []
        for sentence in sentences:
            lower = sentence.lower()
            if any(re.search(p, lower) for p in cls._REFUSAL_PATTERNS):
                dropped.append(sentence)
            else:
                kept.append(sentence)

        kept_text = " ".join(kept).strip()

        # Pure refusal: nothing else of substance remains.
        # Threshold of 30 chars filters out filler like "Voici :" or "Bien sûr."
        if not kept_text or len(kept_text) < 30:
            # Return the original (single canonical sentence is fine UX-wise).
            return answer.strip(), True

        return kept_text, False
    
    def _generate(self, prompt: str, context: str = "") -> str:
        """
        Generate response using LLM.
        
        Args:
            prompt: Full prompt with context
            context: The retrieved context (for fallback response)
            
        Returns:
            Generated text
        """
        if self.use_mock:
            return self.llm_client.generate(prompt)
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.settings.vllm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu es un assistant utile et professionnel pour la documentation ACAPS. "
                            "Analyse la langue de la question fournie dans le message de l'utilisateur et réponds strictement dans cette langue. "
                            "Si la langue semble ambiguë, privilégie le français. N'introduis pas d'anglais dans une réponse attendue en français."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            
            # Fallback: Return context summary if LLM is unavailable
            if self.settings.llm_fallback_enabled and context:
                return self._generate_fallback_response(context)
            
            return "Je ne peux pas générer une réponse pour le moment car le serveur LLM n'est pas disponible. Veuillez réessayer plus tard ou contacter l'administrateur."
    
    def _generate_fallback_response(self, context: str) -> str:
        """
        Generate a fallback response when LLM is unavailable.
        Returns the retrieved context with a disclaimer.
        
        Args:
            context: The retrieved document context
            
        Returns:
            Formatted fallback response
        """
        # Extract the most relevant part (first source)
        if "[Source:" in context:
            parts = context.split("---")
            first_source = parts[0].strip() if parts else context[:500]
        else:
            first_source = context[:500]
        
        return f"""**Note:** Le serveur LLM n'est pas disponible. Voici les informations trouvées dans les documents:

{first_source}

*Pour une réponse plus détaillée, veuillez contacter l'administrateur pour activer le serveur LLM.*"""

    def _generate_conversational(self, prompt: str) -> str:
        """
        Generate a conversational response without RAG context.
        
        Args:
            prompt: Prompt for the LLM
            
        Returns:
            Generated text
        """
        if self.use_mock:
            return "Bonjour! Je suis l'assistant AI de l'ACAPS. Comment puis-je vous aider aujourd'hui?"
            
        try:
            response = self.llm_client.chat.completions.create(
                model=self.settings.vllm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful, professional AI assistant for ACAPS (Morocco Insurance Authority). Always answer in French."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Slightly higher temperature for natural conversation
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating conversational response: {e}")
            return "Bonjour! Je suis l'assistant ACAPS. Comment puis-je vous aider ?"
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all components."""
        health = {
            "vector_store": False,
            "llm": False,
            "embedder": False
        }
        
        try:
            health["vector_store"] = self.vector_store.health_check()
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
        
        try:
            # Simple embedding test
            self.embedder.embed_text("test")
            health["embedder"] = True
        except Exception as e:
            logger.error(f"Embedder health check failed: {e}")
        
        try:
            if self.use_mock:
                health["llm"] = True
            else:
                # Test LLM connection
                response = self.llm_client.models.list()
                health["llm"] = True
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
        
        health["overall"] = all(health.values())
        return health


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def generate(self, prompt: str) -> str:
        """Generate a mock response based on the prompt."""
        if "sick leave" in prompt.lower():
            return "According to Article 5, employees must inform their supervisor within 24 hours in case of sickness and provide a medical certificate within 48 hours."
        elif "vacation" in prompt.lower() or "congés" in prompt.lower():
            return "According to Article 4, each employee is entitled to 22 working days of annual paid leave."
        elif "working hours" in prompt.lower() or "horaires" in prompt.lower():
            return "According to Article 3, working hours are Monday to Friday from 8:30 AM to 4:30 PM, with a lunch break from 12:00 PM to 1:00 PM."
        else:
            return "Based on the available documentation, the information requested can be found in the relevant sections of the internal regulations."


# Singleton instance
_engine_instance: Optional[RAGEngine] = None


def get_engine(use_mock: bool = False) -> RAGEngine:
    """Get or create RAG engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngine(use_mock=use_mock)
    return _engine_instance

