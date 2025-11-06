"""
Модели базы данных SQLAlchemy
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    String, Integer, Float, Boolean, Text, DateTime, JSON,
    ForeignKey, Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base
from src.common.utils import now_utc


# ===== ТЕКСТЫ =====

class TextModel(Base):
    """Модель текста"""
    __tablename__ = "texts"

    # Основные поля
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    text_type: Mapped[str] = mapped_column(String(20), nullable=False)  # 'plain' or 'fb2'
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    language: Mapped[Optional[str]] = mapped_column(String(10))

    # Хранение текста
    storage_type: Mapped[str] = mapped_column(String(20), nullable=False)  # 'database' or 'file'
    content: Mapped[Optional[str]] = mapped_column(Text)  # Для коротких текстов
    file_path: Mapped[Optional[str]] = mapped_column(String(500))  # Для длинных текстов

    # Метаданные
    length: Mapped[Optional[int]] = mapped_column(Integer)  # Длина в символах
    text_metadata: Mapped[dict] = mapped_column(JSON, default={})

    # FB2 специфичные поля
    author: Mapped[Optional[str]] = mapped_column(String(200))
    genre: Mapped[Optional[list]] = mapped_column(JSON)
    year: Mapped[Optional[int]] = mapped_column(Integer)
    publisher: Mapped[Optional[str]] = mapped_column(String(200))

    # Временные метки
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc, onupdate=now_utc)

    # Связи
    embeddings: Mapped[list["EmbeddingCacheModel"]] = relationship(
        back_populates="text", cascade="all, delete-orphan"
    )
    session_texts: Mapped[list["SessionTextModel"]] = relationship(
        back_populates="text", cascade="all, delete-orphan"
    )
    analysis_results: Mapped[list["AnalysisResultModel"]] = relationship(
        back_populates="text", cascade="all, delete-orphan"
    )

    # Индексы
    __table_args__ = (
        Index('idx_texts_created_at', 'created_at'),
        Index('idx_texts_title', 'title'),
        Index('idx_texts_type', 'text_type'),
    )


# ===== СТРАТЕГИИ ЧАНКОВКИ =====

class ChunkingStrategyModel(Base):
    """Модель стратегии разбиения на чанки"""
    __tablename__ = "chunking_strategies"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)

    # Параметры
    base_chunk_size: Mapped[int] = mapped_column(Integer, default=2000)
    min_chunk_size: Mapped[int] = mapped_column(Integer, default=500)
    max_chunk_size: Mapped[int] = mapped_column(Integer, default=4000)
    overlap_percentage: Mapped[float] = mapped_column(Float, default=0.1)

    # Адаптивность
    use_sentence_boundaries: Mapped[bool] = mapped_column(Boolean, default=True)
    use_paragraph_boundaries: Mapped[bool] = mapped_column(Boolean, default=True)
    balance_chunks: Mapped[bool] = mapped_column(Boolean, default=True)

    is_default: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc, onupdate=now_utc)


# ===== СЕССИИ АНАЛИЗА =====

class AnalysisSessionModel(Base):
    """Модель сессии анализа"""
    __tablename__ = "analysis_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)

    # Режим анализа
    mode: Mapped[str] = mapped_column(String(20), default='full_text')
    chunking_strategy_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey('chunking_strategies.id')
    )
    chunked_comparison_strategy: Mapped[Optional[str]] = mapped_column(String(20))

    # FAISS настройки
    use_faiss_search: Mapped[bool] = mapped_column(Boolean, default=False)
    faiss_index_id: Mapped[Optional[str]] = mapped_column(String(64))
    similarity_top_k: Mapped[int] = mapped_column(Integer, default=10)
    similarity_threshold: Mapped[float] = mapped_column(Float, default=0.7)

    # Прогресс
    progress: Mapped[int] = mapped_column(Integer, default=0)
    progress_message: Mapped[Optional[str]] = mapped_column(Text)

    # Результаты
    result: Mapped[Optional[dict]] = mapped_column(JSON)
    error: Mapped[Optional[str]] = mapped_column(Text)

    # Временные метки
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)
    queued_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Пользователь (опционально)
    user_id: Mapped[Optional[str]] = mapped_column(String(64))

    # Связи
    texts: Mapped[list["SessionTextModel"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    analyzers: Mapped[list["SessionAnalyzerModel"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    analysis_results: Mapped[list["AnalysisResultModel"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    comparison_matrix: Mapped[Optional["ComparisonMatrixModel"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )

    # Индексы
    __table_args__ = (
        Index('idx_session_status', 'status'),
        Index('idx_session_created_at', 'created_at'),
        Index('idx_session_user', 'user_id'),
    )


class SessionTextModel(Base):
    """Связь сессии с текстами (M:N)"""
    __tablename__ = "session_texts"

    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey('analysis_sessions.id', ondelete='CASCADE'), primary_key=True
    )
    text_id: Mapped[str] = mapped_column(
        String(64), ForeignKey('texts.id', ondelete='CASCADE'), primary_key=True
    )
    position: Mapped[int] = mapped_column(Integer)  # Порядок в сессии

    # Связи
    session: Mapped["AnalysisSessionModel"] = relationship(back_populates="texts")
    text: Mapped["TextModel"] = relationship(back_populates="session_texts")


class SessionAnalyzerModel(Base):
    """Связь сессии с анализаторами (M:N)"""
    __tablename__ = "session_analyzers"

    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey('analysis_sessions.id', ondelete='CASCADE'), primary_key=True
    )
    analyzer_name: Mapped[str] = mapped_column(String(50), primary_key=True)
    position: Mapped[int] = mapped_column(Integer)  # Порядок выполнения

    # Связи
    session: Mapped["AnalysisSessionModel"] = relationship(back_populates="analyzers")


# ===== РЕЗУЛЬТАТЫ АНАЛИЗОВ =====

class AnalysisResultModel(Base):
    """Модель результата анализа"""
    __tablename__ = "analysis_results"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey('analysis_sessions.id', ondelete='CASCADE')
    )
    text_id: Mapped[str] = mapped_column(
        String(64), ForeignKey('texts.id', ondelete='CASCADE')
    )
    analyzer_name: Mapped[str] = mapped_column(String(50), nullable=False)

    # Результаты
    result_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    interpretation: Mapped[Optional[str]] = mapped_column(Text)  # Человекочитаемый вид
    execution_time_ms: Mapped[Optional[float]] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)

    # Связи
    session: Mapped["AnalysisSessionModel"] = relationship(back_populates="analysis_results")
    text: Mapped["TextModel"] = relationship(back_populates="analysis_results")

    # Индексы
    __table_args__ = (
        Index('idx_result_session', 'session_id'),
        Index('idx_result_text_analyzer', 'text_id', 'analyzer_name'),
    )


class ComparisonMatrixModel(Base):
    """Модель матрицы сравнений"""
    __tablename__ = "comparison_matrices"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey('analysis_sessions.id', ondelete='CASCADE'), unique=True
    )

    # Данные матрицы
    matrix_data: Mapped[dict] = mapped_column(JSON, nullable=False)  # Полная матрица N x N
    aggregated_scores: Mapped[Optional[dict]] = mapped_column(JSON)  # Агрегированные скоры

    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)

    # Связи
    session: Mapped["AnalysisSessionModel"] = relationship(back_populates="comparison_matrix")


# ===== ПРОМПТЫ =====

class PromptTemplateModel(Base):
    """Модель промпт-шаблона"""
    __tablename__ = "prompt_templates"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    analyzer_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Промпты
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    user_prompt_template: Mapped[str] = mapped_column(Text, nullable=False)

    # Параметры генерации
    temperature: Mapped[float] = mapped_column(Float, default=0.7)
    max_tokens: Mapped[int] = mapped_column(Integer, default=1000)
    output_schema: Mapped[Optional[dict]] = mapped_column(JSON)

    is_default: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc, onupdate=now_utc)

    # Индексы
    __table_args__ = (
        Index('idx_prompt_analyzer_type', 'analyzer_type'),
    )


# ===== МОДЕЛИ =====

class ModelConfigModel(Base):
    """Модель конфигурации модели"""
    __tablename__ = "model_configs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_type: Mapped[str] = mapped_column(String(20), nullable=False)  # 'llm' or 'embedding'
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)

    # Пути и параметры
    model_path: Mapped[Optional[str]] = mapped_column(Text)
    quantization: Mapped[Optional[str]] = mapped_column(String(20))
    max_memory_gb: Mapped[Optional[float]] = mapped_column(Float)
    dimensions: Mapped[Optional[int]] = mapped_column(Integer)  # Для embedding
    batch_size: Mapped[int] = mapped_column(Integer, default=32)

    # Общее
    device: Mapped[str] = mapped_column(String(20), default='cuda')
    priority: Mapped[int] = mapped_column(Integer, default=0)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Статистика
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    avg_inference_time_ms: Mapped[Optional[float]] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc, onupdate=now_utc)

    # Связи
    metrics: Mapped[list["ModelMetricsModel"]] = relationship(
        back_populates="model_config", cascade="all, delete-orphan"
    )

    # Индексы
    __table_args__ = (
        UniqueConstraint('model_type', 'model_name', name='uq_model_type_name'),
        Index('idx_model_type', 'model_type'),
        Index('idx_model_enabled', 'is_enabled'),
    )


class ModelMetricsModel(Base):
    """Модель метрик использования модели"""
    __tablename__ = "model_metrics"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_config_id: Mapped[str] = mapped_column(
        String(64), ForeignKey('model_configs.id', ondelete='CASCADE')
    )

    timestamp: Mapped[datetime] = mapped_column(DateTime, default=now_utc)

    # Ресурсы
    gpu_memory_used_mb: Mapped[Optional[float]] = mapped_column(Float)
    gpu_utilization_percent: Mapped[Optional[float]] = mapped_column(Float)
    inference_time_ms: Mapped[Optional[float]] = mapped_column(Float)

    # Контекст
    task_id: Mapped[Optional[str]] = mapped_column(String(64))
    task_type: Mapped[Optional[str]] = mapped_column(String(50))
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer)

    # Статус
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[Optional[str]] = mapped_column(Text)

    # Связи
    model_config: Mapped["ModelConfigModel"] = relationship(back_populates="metrics")

    # Индексы
    __table_args__ = (
        Index('idx_metrics_model_timestamp', 'model_config_id', 'timestamp'),
        Index('idx_metrics_task', 'task_id'),
    )


# ===== FAISS ИНДЕКСЫ =====

class FaissIndexModel(Base):
    """Модель FAISS индекса"""
    __tablename__ = "faiss_indexes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)

    # Тип и параметры
    index_type: Mapped[str] = mapped_column(String(20), nullable=False)
    nlist: Mapped[Optional[int]] = mapped_column(Integer)  # Для IVF
    nprobe: Mapped[Optional[int]] = mapped_column(Integer)
    hnsw_m: Mapped[Optional[int]] = mapped_column(Integer)  # Для HNSW
    pq_m: Mapped[Optional[int]] = mapped_column(Integer)  # Для PQ
    pq_nbits: Mapped[Optional[int]] = mapped_column(Integer)

    # Метаданные
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    total_vectors: Mapped[int] = mapped_column(Integer, default=0)
    file_path: Mapped[Optional[str]] = mapped_column(Text)

    # GPU
    use_gpu: Mapped[bool] = mapped_column(Boolean, default=True)
    gpu_id: Mapped[int] = mapped_column(Integer, default=0)

    last_rebuilt: Mapped[Optional[datetime]] = mapped_column(DateTime)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc, onupdate=now_utc)

    # Связи
    vector_mappings: Mapped[list["FaissVectorMappingModel"]] = relationship(
        back_populates="index", cascade="all, delete-orphan"
    )


class FaissVectorMappingModel(Base):
    """Модель маппинга векторов FAISS → text_id"""
    __tablename__ = "faiss_vector_mappings"

    index_id: Mapped[str] = mapped_column(
        String(64), ForeignKey('faiss_indexes.id', ondelete='CASCADE'), primary_key=True
    )
    position: Mapped[int] = mapped_column(Integer, primary_key=True)  # Позиция в индексе
    text_id: Mapped[str] = mapped_column(
        String(64), ForeignKey('texts.id', ondelete='CASCADE')
    )

    # Связи
    index: Mapped["FaissIndexModel"] = relationship(back_populates="vector_mappings")

    # Индексы
    __table_args__ = (
        Index('idx_vector_mapping_text', 'index_id', 'text_id'),
    )


# ===== КЭШ EMBEDDINGS =====

class EmbeddingCacheModel(Base):
    """Модель кэша embeddings (L2 cache)"""
    __tablename__ = "embedding_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text_id: Mapped[str] = mapped_column(
        String(64), ForeignKey('texts.id', ondelete='CASCADE')
    )
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)

    # Embedding данные
    embedding: Mapped[dict] = mapped_column(JSON, nullable=False)  # Array of floats
    dimensions: Mapped[Optional[int]] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)

    # Связи
    text: Mapped["TextModel"] = relationship(back_populates="embeddings")

    # Индексы
    __table_args__ = (
        UniqueConstraint('text_id', 'model_name', name='uq_text_model'),
        Index('idx_embedding_text', 'text_id'),
        Index('idx_embedding_model', 'model_name'),
    )


# ===== СИСТЕМНЫЕ НАСТРОЙКИ =====

class SystemSettingsModel(Base):
    """Модель системных настроек (singleton)"""
    __tablename__ = "system_settings"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default='system')

    # Модели по умолчанию
    default_llm_model: Mapped[Optional[str]] = mapped_column(String(200))
    default_embedding_model: Mapped[Optional[str]] = mapped_column(String(200))
    default_chunking_strategy_id: Mapped[Optional[str]] = mapped_column(String(64))

    # Лимиты
    max_concurrent_llm_tasks: Mapped[int] = mapped_column(Integer, default=2)
    max_concurrent_embedding_tasks: Mapped[int] = mapped_column(Integer, default=4)
    max_texts_per_session: Mapped[int] = mapped_column(Integer, default=5)

    # Кэширование
    redis_ttl_embeddings: Mapped[int] = mapped_column(Integer, default=86400)
    redis_ttl_analysis: Mapped[int] = mapped_column(Integer, default=3600)

    # Очередь
    task_timeout_seconds: Mapped[int] = mapped_column(Integer, default=3600)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)

    # UI настройки
    ui_settings: Mapped[dict] = mapped_column(JSON, default={})

    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc, onupdate=now_utc)

    # Constraint для singleton
    __table_args__ = (
        CheckConstraint("id = 'system'", name='check_singleton'),
    )


# ===== ИСТОРИЯ ЗАДАЧ =====

class TaskHistoryModel(Base):
    """Модель истории задач"""
    __tablename__ = "task_history"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)
    session_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey('analysis_sessions.id', ondelete='SET NULL')
    )

    status: Mapped[str] = mapped_column(String(20), nullable=False)
    payload: Mapped[Optional[dict]] = mapped_column(JSON)
    result: Mapped[Optional[dict]] = mapped_column(JSON)
    error: Mapped[Optional[str]] = mapped_column(Text)

    queued_at: Mapped[datetime] = mapped_column(DateTime, default=now_utc)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    execution_time_ms: Mapped[Optional[float]] = mapped_column(Float)

    # Индексы
    __table_args__ = (
        Index('idx_task_status_queued', 'status', 'queued_at'),
        Index('idx_task_session', 'session_id'),
        Index('idx_task_type', 'task_type'),
    )
