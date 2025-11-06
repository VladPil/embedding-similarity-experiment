"""
Конфигурация приложения
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Настройки приложения из .env файла"""

    # Приложение
    APP_NAME: str = "Embedding Similarity Experiment"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # API
    API_V2_PREFIX: str = "/api/v2"
    CORS_ORIGINS: list = ["http://localhost:5173", "http://localhost:3000"]

    # База данных PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5433  # Docker использует 5433
    POSTGRES_USER: str = "embedding_user"
    POSTGRES_PASSWORD: str = "embedding_password"
    POSTGRES_DB: str = "embedding_db"

    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def sync_database_url(self) -> str:
        """Для Alembic миграций"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    @property
    def redis_url(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # Кэширование
    REDIS_TTL_EMBEDDINGS: int = 86400  # 24 часа
    REDIS_TTL_ANALYSIS: int = 3600     # 1 час

    # Пути
    DATA_DIR: str = "data"
    TEXTS_DIR: str = "data/texts"
    UPLOADS_DIR: str = "data/uploads"
    INDEXES_DIR: str = "data/indexes"
    EXPORTS_DIR: str = "data/exports"
    MODELS_CACHE_DIR: str = "models_cache"

    # Лимиты
    MAX_TEXTS_PER_SESSION: int = 5
    MAX_CONCURRENT_LLM_TASKS: int = 2
    MAX_CONCURRENT_EMBEDDING_TASKS: int = 4
    MAX_UPLOAD_SIZE_MB: int = 100

    # Модели по умолчанию
    DEFAULT_LLM_MODEL: str = "Qwen/Qwen2.5-3B-Instruct"
    DEFAULT_EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"

    # GPU
    USE_GPU: bool = True
    GPU_DEVICE_ID: int = 0
    MAX_GPU_MEMORY_GB: float = 20.0

    # Очередь задач
    TASK_TIMEOUT_SECONDS: int = 3600
    MAX_TASK_RETRIES: int = 3

    # Логирование
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Игнорировать дополнительные поля из .env


# Singleton instance
settings = Settings()
