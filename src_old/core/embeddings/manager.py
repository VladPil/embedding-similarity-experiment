"""
Embedding model manager with Redis caching.
Combines model management and caching in a single async interface.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional, Dict, Any
import torch
from loguru import logger

from server.config import settings
from server.system.cache import get_redis_cache


class EmbeddingManager:
    """
    Manages embedding models with Redis caching.
    Async-first design for FastAPI integration.
    """

    AVAILABLE_MODELS = {
        # E5 модели разных размерностей
        'multilingual-e5-large': {
            'name': 'intfloat/multilingual-e5-large',
            'dimensions': 1024,
            'description': 'Самая точная E5 модель (1024 dim)'
        },
        'multilingual-e5-base': {
            'name': 'intfloat/multilingual-e5-base',
            'dimensions': 768,
            'description': 'Баланс скорости и качества (768 dim)'
        },
        'multilingual-e5-small': {
            'name': 'intfloat/multilingual-e5-small',
            'dimensions': 384,
            'description': 'Быстрая E5 модель (384 dim)'
        },

        # LaBSE модель
        'labse': {
            'name': 'sentence-transformers/LaBSE',
            'dimensions': 768,
            'description': 'Многоязычная модель LaBSE (768 dim)'
        },

        # Русскоязычные модели
        'rubert-tiny': {
            'name': 'cointegrated/rubert-tiny2',
            'dimensions': 312,
            'description': 'Сверхбыстрая русская модель (312 dim)'
        },
        'rubert-base': {
            'name': 'cointegrated/rubert-base-cased-sentence',
            'dimensions': 768,
            'description': 'Русская BERT модель (768 dim)'
        },

        # MiniLM модели (легкие и быстрые)
        'minilm-l6': {
            'name': 'sentence-transformers/all-MiniLM-L6-v2',
            'dimensions': 384,
            'description': 'Легкая универсальная модель (384 dim)'
        },
        'minilm-l12': {
            'name': 'sentence-transformers/all-MiniLM-L12-v2',
            'dimensions': 384,
            'description': 'Улучшенная MiniLM (384 dim)'
        },

        # MPNet модели (высокое качество)
        'mpnet-base': {
            'name': 'sentence-transformers/all-mpnet-base-v2',
            'dimensions': 768,
            'description': 'Высококачественная MPNet (768 dim)'
        },

        # Distilled модели (быстрые)
        'distilbert': {
            'name': 'sentence-transformers/all-distilroberta-v1',
            'dimensions': 768,
            'description': 'Дистиллированная модель (768 dim)'
        }
    }

    def __init__(self, model_key: str = None):
        """
        Initialize embedding manager.

        Args:
            model_key: Model key from AVAILABLE_MODELS (None for default from settings)
        """
        self.model_key = model_key or settings.models_default_model

        if self.model_key not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model: {self.model_key}. "
                f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )

        self.model_name = self.AVAILABLE_MODELS[self.model_key]['name']
        self.dimensions = self.AVAILABLE_MODELS[self.model_key]['dimensions']
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        logger.info(f"Loading model {self.model_name}...")

        self.model = SentenceTransformer(
            self.model_name,
            cache_folder=settings.models_cache_dir
        )

        # Use GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Model loaded on CPU")

    async def get_embedding(
        self,
        text: str,
        text_id: Optional[str] = None,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Get embedding for text with automatic caching.

        Args:
            text: Text to encode
            text_id: Optional text identifier for caching
            use_cache: Whether to use Redis cache

        Returns:
            Embedding vector
        """
        if not use_cache:
            return self._encode_text(text)

        # Try to get from cache
        cache = await get_redis_cache()
        cache_key = cache.generate_key(
            "embedding",
            self.model_key,
            cache.generate_hash_key(text)
        )

        cached = await cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for text (model: {self.model_key})")
            return cached

        # Compute embedding
        embedding = self._encode_text(text)

        # Save to cache
        await cache.set(cache_key, embedding, ttl=settings.cache_ttl)
        logger.debug(f"Cached embedding (model: {self.model_key})")

        return embedding

    def _encode_text(self, text: str, show_progress: bool = False) -> np.ndarray:
        """
        Encode text into embedding vector (synchronous).

        Args:
            text: Text to encode
            show_progress: Show progress bar

        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
        return embedding

    async def get_embeddings_batch(
        self,
        texts: List[str],
        text_ids: Optional[List[str]] = None,
        use_cache: bool = True,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts with caching.

        Args:
            texts: List of texts to encode
            text_ids: Optional text identifiers for caching
            use_cache: Whether to use Redis cache
            show_progress: Show progress bar

        Returns:
            List of embedding vectors
        """
        if not use_cache:
            return list(self._encode_texts(texts, show_progress))

        embeddings = []
        to_compute = []
        to_compute_indices = []

        cache = await get_redis_cache()

        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = cache.generate_key(
                "embedding",
                self.model_key,
                cache.generate_hash_key(text)
            )

            cached = await cache.get(cache_key)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                to_compute.append(text)
                to_compute_indices.append(i)

        # Compute missing embeddings
        if to_compute:
            logger.info(
                f"Computing {len(to_compute)}/{len(texts)} embeddings "
                f"({len(texts) - len(to_compute)} cached)"
            )
            computed = self._encode_texts(to_compute, show_progress)

            # Update embeddings list and cache
            for idx, embedding in zip(to_compute_indices, computed):
                embeddings[idx] = embedding

                # Cache the new embedding
                text = texts[idx]
                cache_key = cache.generate_key(
                    "embedding",
                    self.model_key,
                    cache.generate_hash_key(text)
                )
                await cache.set(cache_key, embedding, ttl=settings.cache_ttl)

        return embeddings

    def _encode_texts(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode multiple texts into embedding vectors (synchronous).

        Args:
            texts: List of texts to encode
            show_progress: Show progress bar

        Returns:
            Array of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=32
        )
        return embeddings

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = None,
        overlap: int = None
    ) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to split
            chunk_size: Size of each chunk (None for settings default)
            overlap: Overlap between chunks (None for settings default)

        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or settings.analysis_chunk_size
        overlap = overlap or settings.analysis_chunk_overlap

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            # Move to next chunk with overlap
            start = end - overlap

        return chunks

    @staticmethod
    def adaptive_chunk_count(
        text: str,
        min_chunk_size: int = 500,
        max_chunk_size: int = 5000
    ) -> int:
        """
        Calculate adaptive chunk count based on text length.

        Args:
            text: Text to analyze
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk

        Returns:
            Recommended number of chunks
        """
        text_length = len(text)

        if text_length <= max_chunk_size:
            return 1

        optimal_chunk_size = min(
            max_chunk_size,
            max(min_chunk_size, text_length // 100)
        )
        chunk_count = max(1, text_length // optimal_chunk_size)

        return chunk_count

    async def encode_text_by_chunks(
        self,
        text: str,
        num_chunks: Optional[int] = None,
        aggregation: str = 'mean',
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Encode text by splitting into chunks and aggregating embeddings.

        Args:
            text: Text to encode
            num_chunks: Number of chunks (None for adaptive)
            aggregation: Aggregation method ('mean', 'max', 'weighted')
            use_cache: Whether to use cache

        Returns:
            Aggregated embedding vector
        """
        # Determine chunk count
        if num_chunks is None:
            num_chunks = self.adaptive_chunk_count(text)

        # Calculate chunk size
        chunk_size = max(500, len(text) // num_chunks)
        overlap = min(100, chunk_size // 10)

        # Split into chunks
        chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        # Encode chunks
        chunk_embeddings = await self.get_embeddings_batch(
            chunks,
            use_cache=use_cache,
            show_progress=False
        )

        # Convert to numpy array for aggregation
        chunk_array = np.array(chunk_embeddings)

        # Aggregate
        if aggregation == 'mean':
            return np.mean(chunk_array, axis=0)
        elif aggregation == 'max':
            return np.max(chunk_array, axis=0)
        elif aggregation == 'weighted':
            weights = self._get_position_weights(len(chunks))
            return np.average(chunk_array, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    @staticmethod
    def _get_position_weights(num_chunks: int) -> np.ndarray:
        """
        Generate position-based weights (U-shape: beginning and end more important).

        Args:
            num_chunks: Number of chunks

        Returns:
            Array of weights
        """
        positions = np.arange(num_chunks)
        # U-shaped curve
        weights = 1.0 + 0.5 * (
            np.abs(positions - num_chunks / 2) / (num_chunks / 2)
        )
        # Normalize
        weights = weights / np.sum(weights)
        return weights

    @classmethod
    def list_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all available embedding models."""
        return cls.AVAILABLE_MODELS.copy()

    @classmethod
    def get_model_info(cls, model_key: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        return cls.AVAILABLE_MODELS.get(model_key)
