"""
Redis-based cache manager for embeddings and analysis results.
Uses async redis-py for FastAPI compatibility.
"""

import json
import hashlib
import pickle
from typing import Optional, Any
from datetime import timedelta

import redis.asyncio as redis
from loguru import logger

from server.config import settings


class RedisCache:
    """Redis cache manager for storing embeddings and analysis results."""

    def __init__(self, redis_client: redis.Redis):
        """
        Initialize Redis cache manager.

        Args:
            redis_client: Async Redis client instance
        """
        self.client = redis_client
        self.default_ttl = settings.cache_ttl

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        try:
            value = await self.client.get(key)
            if value is None:
                return None

            # Try to unpickle first (for numpy arrays and complex objects)
            try:
                return pickle.loads(value)
            except:
                # Fallback to JSON
                return json.loads(value.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: from config)

        Returns:
            True if successful
        """
        try:
            ttl = ttl or self.default_ttl

            # Try to pickle first (for numpy arrays)
            try:
                serialized = pickle.dumps(value)
            except:
                # Fallback to JSON
                serialized = json.dumps(value).encode('utf-8')

            await self.client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if successful
        """
        try:
            await self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if exists
        """
        try:
            return await self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "embedding:*")

        Returns:
            Number of deleted keys
        """
        try:
            keys = []
            async for key in self.client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                return await self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error clearing cache pattern {pattern}: {e}")
            return 0

    @staticmethod
    def generate_key(*parts: str) -> str:
        """
        Generate cache key from parts.

        Args:
            *parts: Key parts

        Returns:
            Cache key
        """
        key = ":".join(str(p) for p in parts)
        return key

    @staticmethod
    def generate_hash_key(data: str) -> str:
        """
        Generate hash-based cache key.

        Args:
            data: Data to hash

        Returns:
            MD5 hash key
        """
        return hashlib.md5(data.encode()).hexdigest()


# Global Redis client and cache manager
_redis_client: Optional[redis.Redis] = None
_redis_cache: Optional[RedisCache] = None


async def init_redis() -> redis.Redis:
    """Initialize Redis client."""
    global _redis_client

    if _redis_client is None:
        _redis_client = await redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=False,  # We handle decoding ourselves
            max_connections=50,
        )
        logger.info(f"Redis connected: {settings.redis_host}:{settings.redis_port}")

    return _redis_client


async def close_redis():
    """Close Redis connection."""
    global _redis_client, _redis_cache

    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        _redis_cache = None
        logger.info("Redis connection closed")


async def get_redis_cache() -> RedisCache:
    """
    Get Redis cache manager instance.

    Returns:
        RedisCache instance
    """
    global _redis_cache

    if _redis_cache is None:
        client = await init_redis()
        _redis_cache = RedisCache(client)

    return _redis_cache
