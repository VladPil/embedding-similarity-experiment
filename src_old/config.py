"""
Application configuration using pydantic-settings.
All settings loaded from environment variables with EMB__ prefix.
"""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Main application settings - single flat class."""

    # App
    debug: bool = False
    development_mode: bool = False
    app_name: str = "Embedding Similarity Experiment"
    version: str = "2.0.0"

    # Server
    run_host: str = "0.0.0.0"
    run_port: int = 8000

    # Database (PostgreSQL)
    database_host: str = "localhost"
    database_port: int = 5433
    database_name: str = "embedding_db"
    database_user: str = "embedding_user"
    database_password: str = "embedding_password"
    database_echo: bool = False  # SQL logging

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_cache_ttl: int = 3600

    # Models
    models_cache_dir: str = str(BASE_DIR / "models_cache")
    models_default_model: str = "multilingual-e5-small"
    models_max_sequence_length: int = 512

    # Tasks
    tasks_max_workers: int = 3
    tasks_task_timeout: int = 3600

    # Cache
    cache_enabled: bool = True
    cache_type: str = "redis"
    cache_ttl: int = 86400

    # LLM
    llm_device: str = "cuda"
    llm_max_memory_gb: int = 28
    llm_default_model: str = "qwen2.5-0.5b"

    # Analysis
    analysis_chunk_size: int = 1000
    analysis_chunk_overlap: int = 200
    analysis_default_segments: int = 10

    # Loguru
    loguru_enabled: bool = True
    loguru_level: str = "INFO"
    loguru_console_level: str = "INFO"
    loguru_console_format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    loguru_colorize: bool = True
    loguru_write_log_file: bool = False
    loguru_path: str = str(BASE_DIR / "logs" / "app.log")
    loguru_rotation: str = "1 day"
    loguru_retention: str = "1 week"
    loguru_compression: str = "zip"
    loguru_serialize: bool = False

    # CORS
    cors_allow_origins: List[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]

    # Storage
    data_texts_dir: str = str(BASE_DIR / "data" / "texts")  # For file storage
    storage_texts_dir: str = str(BASE_DIR / "data" / "texts")
    storage_fb2_upload_dir: str = str(BASE_DIR / "data" / "uploads")

    # API
    api_prefix: str = "/api"
    api_v1_prefix: str = "/v1"

    model_config = SettingsConfigDict(
        env_file=(BASE_DIR / ".env.example", BASE_DIR / ".env"),
        case_sensitive=False,
        env_nested_delimiter="__",
        env_prefix="EMB__",
        extra="allow",
    )

    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


# Global settings instance
settings = Settings()

# Create directories
Path(settings.models_cache_dir).mkdir(parents=True, exist_ok=True)
Path(settings.storage_texts_dir).mkdir(parents=True, exist_ok=True)
Path(settings.storage_fb2_upload_dir).mkdir(parents=True, exist_ok=True)
if settings.loguru_write_log_file:
    Path(settings.loguru_path).parent.mkdir(parents=True, exist_ok=True)
