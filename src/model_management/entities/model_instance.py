"""
Экземпляр загруженной модели в памяти
"""
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime

from .model_config import ModelConfig
from src.common.utils import now_utc


@dataclass
class ModelInstance:
    """
    Экземпляр модели, загруженной в память

    Представляет реальную модель, готовую к использованию
    """

    # Конфигурация
    config: ModelConfig

    # Сама модель (pytorch model, sentence transformer, etc.)
    model: Any

    # Токенизатор (для LLM)
    tokenizer: Optional[Any] = None

    # Статус
    is_busy: bool = False  # Занята ли модель выполнением задачи
    current_task_id: Optional[str] = None

    # Статистика использования
    total_requests: int = 0
    failed_requests: int = 0
    last_request_at: Optional[datetime] = None

    # Метрики памяти
    allocated_memory_mb: float = 0.0  # Выделенная память
    peak_memory_mb: float = 0.0  # Пиковое потребление

    # Временные метки
    loaded_at: datetime = field(default_factory=now_utc)
    last_used_at: datetime = field(default_factory=now_utc)

    def acquire(self, task_id: str) -> bool:
        """
        Захватить модель для выполнения задачи

        Args:
            task_id: ID задачи

        Returns:
            bool: True если успешно захвачена
        """
        if self.is_busy:
            return False

        self.is_busy = True
        self.current_task_id = task_id
        self.last_request_at = now_utc()
        return True

    def release(self) -> None:
        """Освободить модель после выполнения задачи"""
        self.is_busy = False
        self.current_task_id = None
        self.last_used_at = now_utc()

    def record_request(self, success: bool = True) -> None:
        """
        Записать выполнение запроса

        Args:
            success: Успешно ли выполнен запрос
        """
        self.total_requests += 1
        if not success:
            self.failed_requests += 1

    def get_success_rate(self) -> float:
        """
        Получить процент успешных запросов

        Returns:
            float: Процент успеха (0-1)
        """
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests

    def get_uptime_seconds(self) -> float:
        """
        Получить время работы в секундах

        Returns:
            float: Время работы
        """
        return (now_utc() - self.loaded_at).total_seconds()

    def update_memory_stats(self, allocated_mb: float, peak_mb: float) -> None:
        """
        Обновить статистику памяти

        Args:
            allocated_mb: Выделенная память в МБ
            peak_mb: Пиковая память в МБ
        """
        self.allocated_memory_mb = allocated_mb
        self.peak_memory_mb = max(self.peak_memory_mb, peak_mb)

    def to_dict(self) -> dict:
        """Сериализация в словарь"""
        return {
            "config_id": self.config.id,
            "model_name": self.config.model_name,
            "model_type": self.config.model_type.value,
            "is_busy": self.is_busy,
            "current_task_id": self.current_task_id,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.get_success_rate(),
            "allocated_memory_mb": self.allocated_memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "uptime_seconds": self.get_uptime_seconds(),
            "loaded_at": self.loaded_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat(),
        }

    def __str__(self) -> str:
        status = "busy" if self.is_busy else "idle"
        return f"ModelInstance(name={self.config.model_name}, status={status})"

    def __repr__(self) -> str:
        return self.__str__()
