"""
Схемы сообщений для очереди задач
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ===== БАЗОВЫЕ СХЕМЫ =====

class BaseTaskMessage(BaseModel):
    """Базовая схема сообщения задачи"""
    task_id: str = Field(..., description="Уникальный ID задачи")
    created_at: datetime = Field(default_factory=datetime.now, description="Время создания")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Дополнительные метаданные")


class TaskStatusUpdate(BaseModel):
    """Обновление статуса задачи"""
    task_id: str
    status: str  # pending, running, completed, failed, cancelled
    progress: float = Field(default=0, ge=0, le=100, description="Прогресс 0-100")
    elapsed_time: float = Field(default=0, description="Прошедшее время в секундах")
    estimated_time: Optional[float] = Field(None, description="Оценочное время до завершения")
    current_step: str = Field(default="", description="Текущий шаг выполнения")
    error: Optional[str] = Field(None, description="Сообщение об ошибке")


# ===== СООБЩЕНИЯ ДЛЯ АНАЛИЗА ТЕКСТА =====

class TextAnalysisMessage(BaseTaskMessage):
    """Сообщение для анализа одного текста"""
    text_id: str
    text_title: str
    text_content: str
    analyzer_name: str
    mode: str = "full_text"  # full_text или chunked
    chunk_size: int = 2000
    chunking_strategy_id: Optional[str] = None


class SessionExecutionMessage(BaseTaskMessage):
    """Сообщение для выполнения сессии анализа"""
    session_id: str
    session_name: str

    # Тексты
    text_ids: List[str]

    # Анализаторы
    analyzer_names: List[str]

    # Компаратор (если нужен)
    comparator_name: Optional[str] = None

    # Режим
    mode: str = "full_text"
    chunking_strategy_id: Optional[str] = None
    chunked_comparison_strategy: str = "aggregate_first"

    # FAISS
    use_faiss_search: bool = False
    faiss_index_id: Optional[str] = None
    similarity_top_k: int = 10
    similarity_threshold: float = 0.7


# ===== СООБЩЕНИЯ ДЛЯ ИНДЕКСОВ =====

class IndexBuildMessage(BaseTaskMessage):
    """Сообщение для построения FAISS индекса"""
    index_id: str
    index_name: str
    model_name: str
    index_type: str  # flat, ivf_flat, hnsw, ivf_pq
    text_ids: List[str]

    # Параметры индекса
    nlist: Optional[int] = None
    nprobe: Optional[int] = None
    hnsw_m: Optional[int] = None
    pq_m: Optional[int] = None
    pq_nbits: Optional[int] = None

    use_gpu: bool = True
    gpu_id: int = 0


class IndexSearchMessage(BaseModel):
    """Сообщение для поиска в FAISS индексе"""
    index_id: str
    query_text_id: str
    k: int = 10
    threshold: float = 0.7


# ===== СООБЩЕНИЯ ДЛЯ МОДЕЛЕЙ =====

class ModelDownloadMessage(BaseTaskMessage):
    """Сообщение для скачивания модели"""
    model_id: str
    model_name: str
    model_type: str  # llm или embedding


class ModelLoadMessage(BaseModel):
    """Сообщение для загрузки модели в память"""
    model_id: str
    model_name: str
    model_type: str
    device: str = "cuda"
    quantization: Optional[str] = None


# ===== СООБЩЕНИЯ ДЛЯ ЭКСПОРТА =====

class ExportMessage(BaseTaskMessage):
    """Сообщение для экспорта результатов"""
    session_id: str
    export_format: str  # json, csv, pdf
    include_graphs: bool = True  # Только для PDF


# ===== РЕЗУЛЬТАТЫ =====

class TaskResult(BaseModel):
    """Результат выполнения задачи"""
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: float
    completed_at: datetime = Field(default_factory=datetime.now)
