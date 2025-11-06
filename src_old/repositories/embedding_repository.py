"""
Repository for embedding cache management.
Integrates with Redis for fast access and PostgreSQL for persistence.
"""

import json
import numpy as np
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_
import logging
import pickle

from server.repositories.base import BaseRepository
from server.db.models import EmbeddingCache
from server.system.cache import get_redis_cache

logger = logging.getLogger(__name__)


class EmbeddingRepository(BaseRepository[EmbeddingCache]):
    """
    Repository for embedding cache.
    Two-tier caching: Redis (L1) + PostgreSQL (L2).
    """

    def __init__(self, db: Session):
        """Initialize embedding repository."""
        super().__init__(EmbeddingCache, db)
        self.redis = None  # Will be initialized lazily on first use

    async def _get_redis(self):
        """Get Redis cache instance (lazy initialization)."""
        if self.redis is None:
            self.redis = await get_redis_cache()
        return self.redis

    async def get_embedding(
        self,
        text_id: str,
        model_name: str
    ) -> Optional[np.ndarray]:
        """
        Get embedding from cache (Redis first, then DB).

        Args:
            text_id: Text ID
            model_name: Model name

        Returns:
            Embedding array or None
        """
        # Try Redis first (L1 cache)
        redis = await self._get_redis()
        redis_key = self._get_redis_key(text_id, model_name)
        redis_value = await redis.get(redis_key)

        if redis_value is not None:
            logger.debug(f"Embedding cache hit (Redis): {text_id}/{model_name}")
            # Value is already unpickled by RedisCache
            return redis_value

        # Try database (L2 cache)
        db_cache = self.db.query(EmbeddingCache).filter(
            and_(
                EmbeddingCache.text_id == text_id,
                EmbeddingCache.model_name == model_name
            )
        ).first()

        if db_cache:
            logger.debug(f"Embedding cache hit (DB): {text_id}/{model_name}")
            # Convert JSON array to numpy
            embedding = np.array(db_cache.embedding, dtype=np.float32)

            # Store in Redis for next time
            await self._store_in_redis(text_id, model_name, embedding)

            return embedding

        logger.debug(f"Embedding cache miss: {text_id}/{model_name}")
        return None

    async def store_embedding(
        self,
        text_id: str,
        model_name: str,
        embedding: np.ndarray
    ) -> EmbeddingCache:
        """
        Store embedding in both Redis and DB.

        Args:
            text_id: Text ID
            model_name: Model name
            embedding: Embedding array

        Returns:
            Created cache entry
        """
        # Check if already exists
        existing = await self.get_embedding(text_id, model_name)
        if existing is not None:
            logger.info(f"Embedding already cached: {text_id}/{model_name}")
            return self.db.query(EmbeddingCache).filter(
                and_(
                    EmbeddingCache.text_id == text_id,
                    EmbeddingCache.model_name == model_name
                )
            ).first()

        # Store in database
        cache_entry = self.create(
            text_id=text_id,
            model_name=model_name,
            embedding=embedding.tolist(),  # Convert to JSON-serializable list
            dimensions=len(embedding)
        )

        # Store in Redis
        await self._store_in_redis(text_id, model_name, embedding)

        logger.info(
            f"Stored embedding: {text_id}/{model_name}, "
            f"dimensions: {len(embedding)}"
        )
        return cache_entry

    async def delete_text_embeddings(self, text_id: str) -> int:
        """
        Delete all embeddings for a text (from both Redis and DB).

        Args:
            text_id: Text ID

        Returns:
            Number of deleted entries
        """
        # Get all embeddings for this text
        embeddings = self.db.query(EmbeddingCache).filter(
            EmbeddingCache.text_id == text_id
        ).all()

        count = len(embeddings)

        # Delete from Redis
        redis = await self._get_redis()
        for emb in embeddings:
            redis_key = self._get_redis_key(text_id, emb.model_name)
            await redis.delete(redis_key)

        # Delete from DB
        self.db.query(EmbeddingCache).filter(
            EmbeddingCache.text_id == text_id
        ).delete()
        self.db.commit()

        logger.info(f"Deleted {count} embeddings for text: {text_id}")
        return count

    async def delete_model_embeddings(self, model_name: str) -> int:
        """
        Delete all embeddings for a specific model.

        Args:
            model_name: Model name

        Returns:
            Number of deleted entries
        """
        # Get all embeddings for this model
        embeddings = self.db.query(EmbeddingCache).filter(
            EmbeddingCache.model_name == model_name
        ).all()

        count = len(embeddings)

        # Delete from Redis
        redis = await self._get_redis()
        for emb in embeddings:
            redis_key = self._get_redis_key(emb.text_id, model_name)
            await redis.delete(redis_key)

        # Delete from DB
        self.db.query(EmbeddingCache).filter(
            EmbeddingCache.model_name == model_name
        ).delete()
        self.db.commit()

        logger.info(f"Deleted {count} embeddings for model: {model_name}")
        return count

    async def clear_all_cache(self) -> int:
        """
        Clear all embedding cache (both Redis and DB).

        Returns:
            Number of deleted entries
        """
        # Count entries
        count = self.db.query(EmbeddingCache).count()

        # Clear Redis (pattern matching)
        redis = await self._get_redis()
        pattern = "embedding:*"
        cursor = 0
        while True:
            cursor, keys = await redis.client.scan(cursor, match=pattern, count=100)
            if keys:
                await redis.client.delete(*keys)
            if cursor == 0:
                break

        # Clear DB
        self.db.query(EmbeddingCache).delete()
        self.db.commit()

        logger.info(f"Cleared all embedding cache: {count} entries")
        return count

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Statistics dictionary
        """
        # DB stats
        total_count = self.db.query(EmbeddingCache).count()

        # Group by model
        model_stats = {}
        models = self.db.query(
            EmbeddingCache.model_name
        ).distinct().all()

        for (model,) in models:
            count = self.db.query(EmbeddingCache).filter(
                EmbeddingCache.model_name == model
            ).count()
            model_stats[model] = count

        # Redis stats
        redis = await self._get_redis()
        redis_info = await redis.client.info()
        redis_keys = await redis.client.dbsize()

        return {
            "total_embeddings": total_count,
            "by_model": model_stats,
            "redis": {
                "keys": redis_keys,
                "memory_used": redis_info.get("used_memory_human", "N/A"),
                "hits": redis_info.get("keyspace_hits", 0),
                "misses": redis_info.get("keyspace_misses", 0)
            }
        }

    async def has_embedding(self, text_id: str, model_name: str) -> bool:
        """
        Check if embedding exists in cache.

        Args:
            text_id: Text ID
            model_name: Model name

        Returns:
            True if exists
        """
        # Check Redis first
        redis = await self._get_redis()
        redis_key = self._get_redis_key(text_id, model_name)
        if await redis.exists(redis_key):
            return True

        # Check DB
        return self.db.query(
            self.db.query(EmbeddingCache).filter(
                and_(
                    EmbeddingCache.text_id == text_id,
                    EmbeddingCache.model_name == model_name
                )
            ).exists()
        ).scalar()

    async def warmup_redis_cache(self, limit: int = 100):
        """
        Warmup Redis cache with recent embeddings from DB.

        Args:
            limit: Maximum number of embeddings to load
        """
        recent = self.db.query(EmbeddingCache).order_by(
            EmbeddingCache.created_at.desc()
        ).limit(limit).all()

        count = 0
        for cache in recent:
            embedding = np.array(cache.embedding, dtype=np.float32)
            await self._store_in_redis(
                cache.text_id,
                cache.model_name,
                embedding
            )
            count += 1

        logger.info(f"Warmed up Redis cache with {count} embeddings")

    def _get_redis_key(self, text_id: str, model_name: str) -> str:
        """Generate Redis key for embedding."""
        return f"embedding:{text_id}:{model_name}"

    async def _store_in_redis(
        self,
        text_id: str,
        model_name: str,
        embedding: np.ndarray,
        ttl: int = 3600
    ):
        """
        Store embedding in Redis.

        Args:
            text_id: Text ID
            model_name: Model name
            embedding: Embedding array
            ttl: Time to live in seconds (default 1 hour)
        """
        redis = await self._get_redis()
        redis_key = self._get_redis_key(text_id, model_name)
        await redis.set(redis_key, embedding, ttl=ttl)