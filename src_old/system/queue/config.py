"""
FastStream and Redis configuration for task queue.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class QueueSettings(BaseSettings):
    """Queue configuration settings."""

    # Redis connection
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None

    # Queue names
    analysis_queue: str = "book_analysis_tasks"
    comparison_queue: str = "book_comparison_tasks"

    # Task settings
    task_timeout: int = 3600  # 1 hour
    max_retries: int = 3
    retry_delay: int = 5  # seconds

    # Progress tracking
    progress_channel: str = "task_progress"
    status_ttl: int = 86400  # 24 hours

    class Config:
        env_prefix = "QUEUE_"
        case_sensitive = False


@lru_cache()
def get_queue_settings() -> QueueSettings:
    """Get cached queue settings instance."""
    return QueueSettings()


def get_redis_url() -> str:
    """Get Redis connection URL."""
    settings = get_queue_settings()

    if settings.redis_password:
        return (
            f"redis://:{settings.redis_password}@"
            f"{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
        )

    return f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
