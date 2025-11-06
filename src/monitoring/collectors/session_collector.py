"""
Collector для метрик сессий анализа
"""
from typing import Dict, Any, List
from loguru import logger
from datetime import datetime, timedelta

from .base_collector import BaseCollector
from src.common.utils import now_utc


class SessionCollector(BaseCollector):
    """
    Сборщик метрик сессий анализа

    Собирает:
    - Количество сессий по статусам
    - Среднее время выполнения
    - Статистику ошибок
    - Популярные анализаторы
    """

    def __init__(self, collection_interval_seconds: int = 60):
        """
        Инициализация session collector

        Args:
            collection_interval_seconds: Интервал сбора
        """
        super().__init__(collection_interval_seconds)

        # Счётчики
        self.total_sessions_created = 0
        self.total_sessions_completed = 0
        self.total_sessions_failed = 0

        # Время выполнения (для расчёта среднего)
        self.execution_times: List[float] = []

        # Использование анализаторов
        self.analyzer_usage: Dict[str, int] = {}

    async def collect(self) -> Dict[str, Any]:
        """
        Собрать метрики сессий

        Returns:
            Dict: Метрики сессий
        """
        try:
            metrics = {
                "collector": self.get_collector_name(),
                "sessions": {
                    "created": self.total_sessions_created,
                    "completed": self.total_sessions_completed,
                    "failed": self.total_sessions_failed,
                    "success_rate": self._calculate_success_rate(),
                    "avg_execution_time_seconds": self._calculate_avg_execution_time()
                },
                "analyzers": {
                    "usage": self.analyzer_usage.copy(),
                    "top_5": self._get_top_analyzers(5)
                },
                "performance": {
                    "min_time": min(self.execution_times) if self.execution_times else 0,
                    "max_time": max(self.execution_times) if self.execution_times else 0,
                    "median_time": self._calculate_median(self.execution_times)
                }
            }

            logger.debug(f"Session метрики собраны: {self.total_sessions_completed} завершено")

            return metrics

        except Exception as e:
            logger.error(f"Ошибка сбора session метрик: {e}")
            return {
                "collector": self.get_collector_name(),
                "error": str(e)
            }

    def _calculate_success_rate(self) -> float:
        """
        Вычислить процент успешных сессий

        Returns:
            float: Процент успеха (0-100)
        """
        total = self.total_sessions_completed + self.total_sessions_failed
        if total == 0:
            return 100.0

        return (self.total_sessions_completed / total) * 100

    def _calculate_avg_execution_time(self) -> float:
        """
        Вычислить среднее время выполнения

        Returns:
            float: Среднее время в секундах
        """
        if not self.execution_times:
            return 0.0

        return sum(self.execution_times) / len(self.execution_times)

    def _calculate_median(self, values: List[float]) -> float:
        """
        Вычислить медиану

        Args:
            values: Список значений

        Returns:
            float: Медиана
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)

        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            return sorted_values[n//2]

    def _get_top_analyzers(self, top_k: int) -> List[Dict[str, Any]]:
        """
        Получить топ анализаторов по использованию

        Args:
            top_k: Количество топовых

        Returns:
            List[Dict]: Топ анализаторов
        """
        sorted_analyzers = sorted(
            self.analyzer_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {"analyzer": name, "usage_count": count}
            for name, count in sorted_analyzers[:top_k]
        ]

    def record_session_created(self) -> None:
        """Зарегистрировать создание сессии"""
        self.total_sessions_created += 1

    def record_session_completed(self, execution_time_seconds: float) -> None:
        """
        Зарегистрировать завершение сессии

        Args:
            execution_time_seconds: Время выполнения в секундах
        """
        self.total_sessions_completed += 1
        self.execution_times.append(execution_time_seconds)

        # Ограничиваем размер списка (храним последние 1000)
        if len(self.execution_times) > 1000:
            self.execution_times = self.execution_times[-1000:]

    def record_session_failed(self) -> None:
        """Зарегистрировать провал сессии"""
        self.total_sessions_failed += 1

    def record_analyzer_usage(self, analyzer_name: str) -> None:
        """
        Зарегистрировать использование анализатора

        Args:
            analyzer_name: Название анализатора
        """
        self.analyzer_usage[analyzer_name] = self.analyzer_usage.get(analyzer_name, 0) + 1

    def get_collector_name(self) -> str:
        """Название collector"""
        return "session_collector"
