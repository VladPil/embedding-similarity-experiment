"""
FastStream application with Redis broker.
"""

from faststream import FastStream
from faststream.redis import RedisBroker
from functools import lru_cache

from .config import get_redis_url


# Global broker instance
_broker: RedisBroker | None = None
_app: FastStream | None = None


@lru_cache()
def get_broker() -> RedisBroker:
    """
    Get or create Redis broker instance.

    Returns:
        RedisBroker: Configured Redis broker
    """
    global _broker

    if _broker is None:
        redis_url = get_redis_url()
        _broker = RedisBroker(
            url=redis_url,
            # Connection pool settings
            max_connections=50,
            decode_responses=False,
        )

    return _broker


@lru_cache()
def get_faststream_app() -> FastStream:
    """
    Get or create FastStream application.

    Returns:
        FastStream: Configured FastStream app
    """
    global _app

    if _app is None:
        broker = get_broker()
        _app = FastStream(
            broker,
            title="Book Analysis Task Queue",
            description="FastStream-based task queue for book analysis and comparison",
        )

    return _app


async def start_broker():
    """Start the broker connection."""
    broker = get_broker()
    await broker.start()


async def stop_broker():
    """Stop the broker connection."""
    broker = get_broker()
    await broker.close()
