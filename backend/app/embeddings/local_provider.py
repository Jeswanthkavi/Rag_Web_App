"""Local embedding provider using SentenceTransformers."""

import logging
import threading
from typing import Any, Dict, List, Optional
from backend.app.embeddings.base import BaseEmbeddingProvider, EmbeddingProviderError
from backend.app.embeddings.config import EmbeddingConfig

logger = logging.getLogger("app.embeddings.local")

# Process-level model cache to avoid repeated model loading
_GLOBAL_EMBEDDING_CACHE: Dict[str, Any] = {}
_GLOBAL_EMBEDDING_LOCK = threading.Lock()


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Local inference embedding provider powered by SentenceTransformers."""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig.from_settings()
        self._model = None

    @property
    def provider_name(self) -> str:
        return "local"

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def dimension(self) -> int:
        return self.config.dimension

    def _get_model(self):
        """Lazily loads and caches the SentenceTransformer model instance globally."""
        if self._model is not None:
            return self._model
        model_name = self.config.model_name
        with _GLOBAL_EMBEDDING_LOCK:
            if model_name not in _GLOBAL_EMBEDDING_CACHE:
                try:
                    from sentence_transformers import SentenceTransformer
                    logger.info(f"Loading local embedding model: {model_name}")
                    _GLOBAL_EMBEDDING_CACHE[model_name] = SentenceTransformer(model_name)
                except Exception as e:
                    sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
                    logger.error(f"Failed to load embedding model '{model_name}': {sanitized}")
                    raise EmbeddingProviderError(
                        f"Failed to load embedding model '{model_name}': {sanitized}"
                    ) from e
            self._model = _GLOBAL_EMBEDDING_CACHE[model_name]
            return self._model

    def embed_text(self, text: str) -> List[float]:
        """Generates a dense vector embedding for a single text."""
        vectors = self.embed_texts([text])
        if not vectors:
            raise EmbeddingProviderError("Failed to generate embedding for input text.")
        return vectors[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates dense vector embeddings for a list of text strings in batch.
        
        Preserves strict 1:1 positional alignment between inputs and returned vectors.
        """
        if not texts:
            return []

        model = self._get_model()

        try:
            # Handle empty/whitespace strings deterministically
            processed_texts = [t if (t and t.strip()) else " " for t in texts]

            raw_embeddings = model.encode(
                sentences=processed_texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Convert numpy array to standard Python float list
            result_vectors: List[List[float]] = [
                [float(val) for val in row] for row in raw_embeddings
            ]

            # Validate generated vectors
            self.validate_vectors(result_vectors, expected_count=len(texts))
            return result_vectors

        except EmbeddingProviderError:
            raise
        except Exception as e:
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.error(f"Local embedding generation failed: {sanitized}")
            raise EmbeddingProviderError(f"Embedding generation failed: {sanitized}") from e
