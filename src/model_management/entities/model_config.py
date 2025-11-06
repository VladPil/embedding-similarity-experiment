"""
Конфигурация модели
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

from src.common.types import ModelType, ModelStatus, QuantizationType, DeviceType, Metadata
from src.common.utils import now_utc


@dataclass
class ModelConfig:
    """
    Конфигурация модели (LLM или Embedding)

    Содержит все настройки для загрузки и использования модели
    """

    # Основные поля
    id: str
    model_type: ModelType  # LLM или EMBEDDING
    model_name: str  # Название модели (например, "Qwen/Qwen2.5-3B-Instruct")

    # Пути и параметры загрузки
    model_path: Optional[str] = None  # Путь к скачанной модели
    quantization: QuantizationType = QuantizationType.NONE
    max_memory_gb: float = 12.0  # Максимальная память для модели

    # Для embedding моделей
    dimensions: Optional[int] = None  # Размерность векторов
    batch_size: int = 32  # Размер батча для обработки

    # Устройство
    device: DeviceType = DeviceType.CUDA
    device_id: int = 0  # ID GPU

    # Приоритет и доступность
    priority: int = 0  # Приоритет использования (больше = выше)
    is_enabled: bool = True

    # Статистика использования
    usage_count: int = 0
    last_used_at: Optional[datetime] = None
    avg_inference_time_ms: Optional[float] = None
    total_tokens_processed: int = 0  # Для LLM

    # Текущий статус
    status: ModelStatus = ModelStatus.NOT_DOWNLOADED

    # Метаданные
    metadata: Metadata = field(default_factory=dict)

    # Временные метки
    created_at: datetime = field(default_factory=now_utc)
    updated_at: datetime = field(default_factory=now_utc)

    def is_llm(self) -> bool:
        """Проверить является ли модель LLM"""
        return self.model_type == ModelType.LLM

    def is_embedding(self) -> bool:
        """Проверить является ли модель Embedding"""
        return self.model_type == ModelType.EMBEDDING

    def is_downloaded(self) -> bool:
        """Проверить скачана ли модель"""
        return self.status in [
            ModelStatus.DOWNLOADED,
            ModelStatus.LOADING,
            ModelStatus.LOADED,
        ]

    def is_loaded(self) -> bool:
        """Проверить загружена ли модель в память"""
        return self.status == ModelStatus.LOADED

    def is_available(self) -> bool:
        """Проверить доступна ли модель для использования"""
        return self.is_enabled and self.is_loaded()

    def get_memory_estimate_mb(self) -> float:
        """
        Оценить потребление памяти в МБ

        Returns:
            float: Оценка памяти в МБ
        """
        if self.is_llm():
            # Оценка для LLM на основе размера модели
            # Примерная формула: параметры * 2 байта (fp16) / 1024^2
            if "3B" in self.model_name or "3b" in self.model_name:
                base_memory = 6000  # ~6 GB для 3B модели
            elif "7B" in self.model_name or "7b" in self.model_name:
                base_memory = 14000  # ~14 GB для 7B модели
            elif "1.5B" in self.model_name or "1.5b" in self.model_name:
                base_memory = 3000  # ~3 GB для 1.5B модели
            else:
                base_memory = 2000  # Дефолт

            # Квантизация уменьшает размер
            if self.quantization == QuantizationType.INT8:
                base_memory *= 0.5
            elif self.quantization == QuantizationType.INT4:
                base_memory *= 0.25

            return base_memory
        else:
            # Embedding модели обычно меньше
            return 500  # ~500 MB

    def update_usage_stats(
        self,
        inference_time_ms: float,
        tokens_processed: int = 0
    ) -> None:
        """
        Обновить статистику использования

        Args:
            inference_time_ms: Время инференса в миллисекундах
            tokens_processed: Количество обработанных токенов
        """
        self.usage_count += 1
        self.last_used_at = now_utc()
        self.total_tokens_processed += tokens_processed

        # Обновляем среднее время
        if self.avg_inference_time_ms is None:
            self.avg_inference_time_ms = inference_time_ms
        else:
            # Экспоненциальное сглаживание
            alpha = 0.1
            self.avg_inference_time_ms = (
                alpha * inference_time_ms +
                (1 - alpha) * self.avg_inference_time_ms
            )

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь"""
        return {
            "id": self.id,
            "model_type": self.model_type.value,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "quantization": self.quantization.value,
            "max_memory_gb": self.max_memory_gb,
            "dimensions": self.dimensions,
            "batch_size": self.batch_size,
            "device": self.device.value,
            "device_id": self.device_id,
            "priority": self.priority,
            "is_enabled": self.is_enabled,
            "status": self.status.value,
            "usage_count": self.usage_count,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "avg_inference_time_ms": self.avg_inference_time_ms,
            "total_tokens_processed": self.total_tokens_processed,
            "memory_estimate_mb": self.get_memory_estimate_mb(),
            "is_available": self.is_available(),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def create_llm_config(
        cls,
        model_id: str,
        model_name: str,
        quantization: QuantizationType = QuantizationType.NONE,
        max_memory_gb: float = 12.0,
        priority: int = 0
    ) -> "ModelConfig":
        """
        Создать конфигурацию для LLM модели

        Args:
            model_id: ID модели
            model_name: Название модели
            quantization: Тип квантизации
            max_memory_gb: Максимальная память
            priority: Приоритет

        Returns:
            ModelConfig: Конфигурация
        """
        return cls(
            id=model_id,
            model_type=ModelType.LLM,
            model_name=model_name,
            quantization=quantization,
            max_memory_gb=max_memory_gb,
            priority=priority,
        )

    @classmethod
    def create_embedding_config(
        cls,
        model_id: str,
        model_name: str,
        dimensions: int,
        batch_size: int = 32,
        priority: int = 0
    ) -> "ModelConfig":
        """
        Создать конфигурацию для Embedding модели

        Args:
            model_id: ID модели
            model_name: Название модели
            dimensions: Размерность векторов
            batch_size: Размер батча
            priority: Приоритет

        Returns:
            ModelConfig: Конфигурация
        """
        return cls(
            id=model_id,
            model_type=ModelType.EMBEDDING,
            model_name=model_name,
            dimensions=dimensions,
            batch_size=batch_size,
            priority=priority,
        )

    def __str__(self) -> str:
        return f"ModelConfig(name={self.model_name}, type={self.model_type.value}, status={self.status.value})"

    def __repr__(self) -> str:
        return self.__str__()
