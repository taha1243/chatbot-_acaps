"""
Embedding Generator for Atlas-RAG
Uses AutoTokenizer + AutoModel from transformers for explicit cache control.
"""
import os
import logging
from typing import List, Optional, Union
from dataclasses import dataclass

import warnings
import numpy as np

# Suppress noisy deprecation warnings from older huggingface_hub versions
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

logger = logging.getLogger(__name__)

# Explicit cache directory — matches docker-compose HF_HOME volume
_DEFAULT_CACHE_DIR = os.getenv("HF_HOME", "/app/.cache/huggingface")


def _mean_pooling(model_output, attention_mask):
    """Mean pooling over token embeddings, weighted by attention mask."""
    import torch
    token_embeddings = model_output[0]  # (batch, seq_len, hidden)
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)


@dataclass
class EmbeddingResult:
    text: str
    embedding: List[float]
    model: str
    dimension: int


class EmbeddingGenerator:
    """
    Generates sentence embeddings using AutoTokenizer + AutoModel.
    Explicit cache_dir so the model is always stored in the persisted Docker volume.
    """

    # Models that require asymmetric "query: " / "passage: " prefixes
    _E5_PREFIX_MODELS = ("intfloat/multilingual-e5", "intfloat/e5-", "microsoft/e5-")

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        normalize: bool = True,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._device = device
        self._tokenizer = None
        self._model = None
        self._use_e5_prefix = any(model_name.startswith(p) for p in self._E5_PREFIX_MODELS)

        logger.info("Initializing EmbeddingGenerator — model: %s  cache: %s", model_name, self.cache_dir)

    def _load(self):
        """Load tokenizer and model into memory (called once)."""
        import torch
        from transformers import AutoTokenizer, AutoModel

        logger.info("Loading model from HuggingFace: %s", self.model_name)

        os.makedirs(self.cache_dir, exist_ok=True)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        self._model.eval()

        # Resolve device
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)

        logger.info("Model loaded on %s. Hidden size: %d", self._device, self._model.config.hidden_size)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load()
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.config.hidden_size

    def _encode_batch(self, texts: List[str]) -> "np.ndarray":
        """Tokenize, forward-pass, mean-pool, optionally L2-normalize. Returns np array (N, D)."""
        import torch
        import torch.nn.functional as F

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self.model(**encoded)

        embeddings = _mean_pooling(output, encoded["attention_mask"])

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    # ── Public API ────────────────────────────────────────────────────

    def embed_text(self, text: str) -> EmbeddingResult:
        vec = self._encode_batch([text])[0]
        return EmbeddingResult(
            text=text,
            embedding=vec.tolist(),
            model=self.model_name,
            dimension=len(vec),
        )

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[EmbeddingResult]:
        if not texts:
            return []

        logger.info("Embedding %d texts with batch_size=%d", len(texts), batch_size)
        results: List[EmbeddingResult] = []

        if self._use_e5_prefix:
            texts = [f"passage: {t}" for t in texts]

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs = self._encode_batch(batch)
            for text, vec in zip(batch, vecs):
                results.append(EmbeddingResult(
                    text=text,
                    embedding=vec.tolist(),
                    model=self.model_name,
                    dimension=len(vec),
                ))
            if show_progress:
                logger.info("  Embedded %d / %d", min(i + batch_size, len(texts)), len(texts))

        return results

    def embed_query(self, query: str) -> List[float]:
        """Embed a query for semantic search, applying model-specific prefixes."""
        if self._use_e5_prefix:
            query = f"query: {query}"
        return self.embed_text(query).embedding


class MockEmbeddingGenerator:
    """Deterministic mock for unit tests — no network, no GPU."""

    def __init__(self, dimension: int = 384, seed: int = 42):
        self.model_name = "mock-embedding-model"
        self._dimension = dimension
        self._seed = seed
        logger.info("MockEmbeddingGenerator initialized (dim=%d)", dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> EmbeddingResult:
        np.random.seed(hash(text) % (2**32))
        vec = np.random.randn(self._dimension)
        vec = vec / np.linalg.norm(vec)
        return EmbeddingResult(text=text, embedding=vec.tolist(), model=self.model_name, dimension=self._dimension)

    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[EmbeddingResult]:
        return [self.embed_text(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_text(query).embedding


def get_embedding_generator(
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    use_mock: bool = False,
    dimension: int = 384,
) -> Union[EmbeddingGenerator, MockEmbeddingGenerator]:
    if use_mock:
        return MockEmbeddingGenerator(dimension=dimension)
    return EmbeddingGenerator(model_name=model_name)
