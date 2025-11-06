"""Cache system for embeddings and analysis results."""

from .redis_cache import RedisCache, get_redis_cache, init_redis, close_redis

__all__ = ["RedisCache", "get_redis_cache", "init_redis", "close_redis"]
