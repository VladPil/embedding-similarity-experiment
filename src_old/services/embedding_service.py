"""
Service for embedding management with caching.
Implements Strategy pattern for different models.
"""

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
import numpy as np
import logging

from server.services.base import BaseService
from server.repositories.embedding_repository import EmbeddingRepository
from server.repositories.text_repository import TextRepository
from server.core.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class EmbeddingService(BaseService):
    """
    Service for embedding operations.
    Manages models and caching strategy.
    """

    def __init__(self, db: Session):
        """Initialize embedding service."""
        super().__init__(db)
        self.embedding_repo = EmbeddingRepository(db)
        self.text_repo = TextRepository(db)
        self.models: Dict[str, EmbeddingManager] = {}

    async def initialize(self):
        """Initialize service and warmup cache."""
        # Warmup Redis cache with recent embeddings
        await self.embedding_repo.warmup_redis_cache(limit=50)
        self.log_info("Embedding service initialized, cache warmed up")

    async def cleanup(self):
        """Cleanup service resources."""
        # Clear model instances
        self.models.clear()
        self.log_info("Embedding service cleaned up")

    def get_model(self, model_key: str) -> EmbeddingManager:
        """
        Get or create embedding model instance (lazy loading).

        Args:
            model_key: Model key

        Returns:
            EmbeddingManager instance
        """
        if model_key not in self.models:
            self.log_info(f"Loading model: {model_key}")
            self.models[model_key] = EmbeddingManager(model_key)

        return self.models[model_key]

    async def get_embedding(
        self,
        text_id: str,
        model_key: str = "multilingual-e5-base",
        use_cache: bool = True
    ) -> Optional[np.ndarray]:
        """
        Get embedding for text with caching.

        Args:
            text_id: Text ID
            model_key: Model to use
            use_cache: Whether to use cache

        Returns:
            Embedding array or None
        """
        try:
            # Try to get from cache first
            if use_cache:
                cached = await self.embedding_repo.get_embedding(text_id, model_key)
                if cached is not None:
                    self.log_debug(f"Cache hit: {text_id}/{model_key}")
                    return cached

            # Get text content
            content = await self.text_repo.get_content(text_id)
            if not content:
                self.log_error(f"Text not found: {text_id}")
                return None

            # Generate embedding
            self.log_info(f"Generating embedding: {text_id}/{model_key}")
            model = self.get_model(model_key)
            embedding = await model.get_embedding(
                content,
                text_id=text_id,
                use_cache=False  # We handle caching ourselves
            )

            # Store in cache
            if use_cache and embedding is not None:
                await self.embedding_repo.store_embedding(
                    text_id, model_key, embedding
                )

            return embedding

        except Exception as e:
            self.log_error(f"Error getting embedding for {text_id}", e)
            return None

    async def get_embeddings_batch(
        self,
        text_ids: List[str],
        model_key: str = "multilingual-e5-base",
        use_cache: bool = True
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Get embeddings for multiple texts.

        Args:
            text_ids: List of text IDs
            model_key: Model to use
            use_cache: Whether to use cache

        Returns:
            Dictionary mapping text_id to embedding
        """
        results = {}

        for text_id in text_ids:
            embedding = await self.get_embedding(text_id, model_key, use_cache)
            results[text_id] = embedding

        return results

    async def precompute_embeddings(
        self,
        text_id: str,
        model_keys: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Precompute embeddings for all models.

        Args:
            text_id: Text ID
            model_keys: Models to use (None for all)

        Returns:
            Dictionary mapping model to success status
        """
        if model_keys is None:
            model_keys = list(EmbeddingManager.AVAILABLE_MODELS.keys())

        results = {}

        for model_key in model_keys:
            try:
                embedding = await self.get_embedding(
                    text_id, model_key, use_cache=True
                )
                results[model_key] = embedding is not None
            except Exception as e:
                self.log_error(f"Error precomputing {model_key} for {text_id}", e)
                results[model_key] = False

        return results

    async def clear_text_cache(self, text_id: str) -> int:
        """
        Clear all embeddings for a text.

        Args:
            text_id: Text ID

        Returns:
            Number of cleared embeddings
        """
        return await self.embedding_repo.delete_text_embeddings(text_id)

    async def clear_model_cache(self, model_key: str) -> int:
        """
        Clear all embeddings for a model.

        Args:
            model_key: Model key

        Returns:
            Number of cleared embeddings
        """
        # Also remove from loaded models
        if model_key in self.models:
            del self.models[model_key]

        return await self.embedding_repo.delete_model_embeddings(model_key)

    async def clear_all_cache(self) -> int:
        """
        Clear all embedding cache.

        Returns:
            Number of cleared embeddings
        """
        # Clear loaded models
        self.models.clear()

        return await self.embedding_repo.clear_all_cache()

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Cache stats dictionary
        """
        stats = await self.embedding_repo.get_cache_stats()
        stats["loaded_models"] = list(self.models.keys())
        return stats

    async def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available models.

        Returns:
            Dictionary of model information
        """
        return EmbeddingManager.AVAILABLE_MODELS

    async def check_embedding_exists(
        self,
        text_id: str,
        model_key: str
    ) -> bool:
        """
        Check if embedding exists in cache.

        Args:
            text_id: Text ID
            model_key: Model key

        Returns:
            True if exists
        """
        return await self.embedding_repo.has_embedding(text_id, model_key)