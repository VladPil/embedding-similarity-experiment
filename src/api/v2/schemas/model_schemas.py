"""
Pydantic схемы для управления моделями
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class ModelConfigRequest(BaseModel):
    """Запрос на создание конфигурации модели"""
    model_name: str = Field(..., description="Название модели (HuggingFace)")
    model_type: str = Field(..., description="Тип: llm/embedding")
    quantization: str = Field("none", description="Квантизация: none/int8/int4")
    max_memory_gb: float = Field(12.0, ge=1.0, le=24.0)
    device: str = Field("cuda", description="Устройство: cuda/cpu")
    device_id: int = Field(0, ge=0, description="ID GPU")
    priority: int = Field(0, description="Приоритет использования")

    # Для embedding моделей
    dimensions: Optional[int] = Field(None, ge=128, description="Размерность векторов")
    batch_size: Optional[int] = Field(32, ge=1, description="Размер батча")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "Qwen/Qwen2.5-3B-Instruct",
                "model_type": "llm",
                "quantization": "int4",
                "max_memory_gb": 6.0,
                "device": "cuda",
                "device_id": 0,
                "priority": 10
            }
        }


class ModelConfigResponse(BaseModel):
    """Ответ с информацией о конфигурации модели"""
    id: str
    model_name: str
    model_type: str
    quantization: str
    max_memory_gb: float
    device: str
    device_id: int
    priority: int
    status: str = Field(description="Статус: not_downloaded/downloaded/loading/loaded/error")
    is_enabled: bool
    memory_estimate_mb: float
    created_at: datetime


class ModelLoadRequest(BaseModel):
    """Запрос на загрузку модели в память"""
    config_id: str

    class Config:
        json_schema_extra = {
            "example": {
                "config_id": "model_config_123"
            }
        }


class ModelInstanceResponse(BaseModel):
    """Информация о загруженном экземпляре модели"""
    config_id: str
    model_name: str
    model_type: str
    is_busy: bool
    current_task_id: Optional[str]
    total_requests: int
    failed_requests: int
    success_rate: float
    allocated_memory_mb: float
    peak_memory_mb: float
    uptime_seconds: float
    loaded_at: datetime


class LLMGenerateRequest(BaseModel):
    """Запрос на генерацию LLM"""
    prompt: str = Field(..., min_length=1)
    model_name: Optional[str] = Field(None, description="Название модели (если None - дефолтная)")
    max_tokens: int = Field(512, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    stop_sequences: Optional[List[str]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Напиши короткий рассказ о космосе",
                "model_name": "Qwen/Qwen2.5-3B-Instruct",
                "max_tokens": 256,
                "temperature": 0.7
            }
        }


class LLMGenerateResponse(BaseModel):
    """Ответ на генерацию LLM"""
    generated_text: str
    model_used: str
    tokens_generated: int
    generation_time_seconds: float


class EmbeddingRequest(BaseModel):
    """Запрос на получение embeddings"""
    texts: List[str] = Field(..., min_items=1)
    model_name: Optional[str] = None
    normalize: bool = Field(True, description="Нормализовать векторы")

    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["Первый текст", "Второй текст"],
                "model_name": None,
                "normalize": True
            }
        }


class EmbeddingResponse(BaseModel):
    """Ответ с embeddings"""
    embeddings: List[List[float]]
    model_used: str
    dimension: int
    texts_count: int


class ModelPoolStatsResponse(BaseModel):
    """Статистика пула моделей"""
    total_models: int
    llm_models: int
    embedding_models: int
    busy_models: int
    available_models: int
    total_memory_gb: float
    max_memory_gb: float
    memory_usage_percent: float


class GPUStatsResponse(BaseModel):
    """Статистика GPU"""
    device_id: int
    memory_used_mb: float
    memory_total_mb: float
    memory_free_mb: float
    memory_usage_percent: float
    utilization_percent: float
    temperature_celsius: float
    is_available: bool
