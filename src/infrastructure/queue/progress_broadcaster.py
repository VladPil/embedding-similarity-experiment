"""
Broadcaster для отправки обновлений прогресса через Redis pub/sub
"""
from typing import Optional
from loguru import logger

from .broker import broker
from .schemas import TaskStatusUpdate
from src.infrastructure.cache.cache_manager import cache_manager


class ProgressBroadcaster:
    """Broadcaster для отправки обновлений прогресса задач"""

    def __init__(self):
        """Инициализация broadcaster"""
        self.broker = broker
        self.cache = cache_manager

    async def broadcast_progress(
        self,
        task_id: str,
        status: str,
        progress: float = 0,
        current_step: str = "",
        elapsed_time: float = 0,
        estimated_time: Optional[float] = None,
        error: Optional[str] = None,
    ) -> bool:
        """
        Отправить обновление прогресса задачи

        Args:
            task_id: ID задачи
            status: Статус (pending, running, completed, failed, cancelled)
            progress: Прогресс 0-100
            current_step: Текущий шаг выполнения
            elapsed_time: Прошедшее время в секундах
            estimated_time: Оценочное время до завершения
            error: Сообщение об ошибке

        Returns:
            bool: True если успешно
        """
        try:
            update = TaskStatusUpdate(
                task_id=task_id,
                status=status,
                progress=progress,
                elapsed_time=elapsed_time,
                estimated_time=estimated_time,
                current_step=current_step,
                error=error,
            )

            # Сохраняем в кэш для быстрого доступа
            await self.cache.set_task_progress(
                task_id=task_id,
                progress=int(progress),
                message=current_step,
            )

            # Отправляем через pub/sub для WebSocket клиентов
            channel = f"progress:{task_id}"
            await self.broker.publish(
                message=update.model_dump(),
                channel=channel,
            )

            logger.debug(
                f"📊 Прогресс задачи {task_id}: {progress:.1f}% - {current_step}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка отправки прогресса: {e}")
            return False

    async def broadcast_started(
        self,
        task_id: str,
        current_step: str = "Задача запущена"
    ) -> bool:
        """
        Уведомить о запуске задачи

        Args:
            task_id: ID задачи
            current_step: Описание текущего шага

        Returns:
            bool: True если успешно
        """
        return await self.broadcast_progress(
            task_id=task_id,
            status="running",
            progress=0,
            current_step=current_step,
        )

    async def broadcast_completed(
        self,
        task_id: str,
        elapsed_time: float,
        final_message: str = "Задача завершена"
    ) -> bool:
        """
        Уведомить о завершении задачи

        Args:
            task_id: ID задачи
            elapsed_time: Время выполнения в секундах
            final_message: Финальное сообщение

        Returns:
            bool: True если успешно
        """
        return await self.broadcast_progress(
            task_id=task_id,
            status="completed",
            progress=100,
            current_step=final_message,
            elapsed_time=elapsed_time,
        )

    async def broadcast_failed(
        self,
        task_id: str,
        error: str,
        elapsed_time: float
    ) -> bool:
        """
        Уведомить об ошибке задачи

        Args:
            task_id: ID задачи
            error: Сообщение об ошибке
            elapsed_time: Время выполнения до ошибки

        Returns:
            bool: True если успешно
        """
        return await self.broadcast_progress(
            task_id=task_id,
            status="failed",
            progress=0,
            current_step="Ошибка выполнения",
            elapsed_time=elapsed_time,
            error=error,
        )

    async def broadcast_cancelled(
        self,
        task_id: str,
        elapsed_time: float
    ) -> bool:
        """
        Уведомить об отмене задачи

        Args:
            task_id: ID задачи
            elapsed_time: Время до отмены

        Returns:
            bool: True если успешно
        """
        return await self.broadcast_progress(
            task_id=task_id,
            status="cancelled",
            progress=0,
            current_step="Задача отменена",
            elapsed_time=elapsed_time,
        )


# Singleton instance
progress_broadcaster = ProgressBroadcaster()
