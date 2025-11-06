"""
Базовый класс для collectors метрик
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from datetime import datetime

from src.common.utils import now_utc


class BaseCollector(ABC):
    """
    Базовый класс для сборщиков метрик

    Определяет общий интерфейс для всех collectors
    """

    def __init__(self, collection_interval_seconds: int = 60):
        """
        Инициализация collector

        Args:
            collection_interval_seconds: Интервал сбора метрик (секунды)
        """
        self.collection_interval = collection_interval_seconds
        self.last_collection_at: datetime = now_utc()

    @abstractmethod
    async def collect(self) -> Dict[str, Any]:
        """
        Собрать метрики

        Returns:
            Dict: Словарь с метриками
        """
        pass

    @abstractmethod
    def get_collector_name(self) -> str:
        """
        Получить название collector

        Returns:
            str: Название
        """
        pass

    def should_collect(self) -> bool:
        """
        Проверить нужно ли собирать метрики

        Returns:
            bool: True если прошёл интервал
        """
        elapsed = (now_utc() - self.last_collection_at).total_seconds()
        return elapsed >= self.collection_interval

    def mark_collected(self) -> None:
        """Отметить что метрики собраны"""
        self.last_collection_at = now_utc()

    async def collect_if_needed(self) -> Dict[str, Any]:
        """
        Собрать метрики если прошёл интервал

        Returns:
            Dict: Метрики или пустой dict
        """
        if self.should_collect():
            metrics = await self.collect()
            self.mark_collected()
            return metrics
        return {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Сериализация в словарь

        Returns:
            Dict: Информация о collector
        """
        return {
            "collector_name": self.get_collector_name(),
            "collection_interval": self.collection_interval,
            "last_collection_at": self.last_collection_at.isoformat()
        }
