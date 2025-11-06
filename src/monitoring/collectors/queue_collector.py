"""
Collector для метрик очереди задач
"""
from typing import Dict, Any
from loguru import logger

from .base_collector import BaseCollector


class QueueCollector(BaseCollector):
    """
    Сборщик метрик очереди задач (FastStream + Redis)

    Собирает:
    - Количество задач в очереди
    - Количество активных обработчиков
    - Статистику выполнения
    """

    def __init__(
        self,
        redis_client: Any = None,
        collection_interval_seconds: int = 30
    ):
        """
        Инициализация queue collector

        Args:
            redis_client: Redis клиент
            collection_interval_seconds: Интервал сбора
        """
        super().__init__(collection_interval_seconds)
        self.redis_client = redis_client

        # Счётчики (в памяти)
        self.total_tasks_created = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0

    async def collect(self) -> Dict[str, Any]:
        """
        Собрать метрики очереди

        Returns:
            Dict: Метрики очереди
        """
        try:
            metrics = {
                "collector": self.get_collector_name(),
                "tasks": {
                    "created": self.total_tasks_created,
                    "completed": self.total_tasks_completed,
                    "failed": self.total_tasks_failed,
                    "success_rate": self._calculate_success_rate()
                },
                "queue": {
                    "pending": await self._get_pending_tasks_count(),
                    "active": 0  # TODO: получить из FastStream
                }
            }

            logger.debug(f"Queue метрики собраны: {metrics['tasks']['completed']} выполнено")

            return metrics

        except Exception as e:
            logger.error(f"Ошибка сбора queue метрик: {e}")
            return {
                "collector": self.get_collector_name(),
                "error": str(e)
            }

    async def _get_pending_tasks_count(self) -> int:
        """
        Получить количество задач в очереди

        Returns:
            int: Количество задач
        """
        if not self.redis_client:
            return 0

        try:
            # Получаем длину списка задач в Redis
            # TODO: реальная реализация с FastStream
            return 0

        except Exception as e:
            logger.warning(f"Не удалось получить количество задач: {e}")
            return 0

    def _calculate_success_rate(self) -> float:
        """
        Вычислить процент успешных задач

        Returns:
            float: Процент успеха (0-100)
        """
        total = self.total_tasks_completed + self.total_tasks_failed
        if total == 0:
            return 100.0

        return (self.total_tasks_completed / total) * 100

    def record_task_created(self) -> None:
        """Зарегистрировать создание задачи"""
        self.total_tasks_created += 1

    def record_task_completed(self) -> None:
        """Зарегистрировать выполнение задачи"""
        self.total_tasks_completed += 1

    def record_task_failed(self) -> None:
        """Зарегистрировать провал задачи"""
        self.total_tasks_failed += 1

    def get_collector_name(self) -> str:
        """Название collector"""
        return "queue_collector"
