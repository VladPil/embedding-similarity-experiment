"""
Сервис для сбора и агрегации метрик
"""
from typing import Dict, Any, List, Optional
from loguru import logger
import asyncio

from ..collectors.base_collector import BaseCollector
from ..collectors.gpu_collector import GPUCollector
from ..collectors.queue_collector import QueueCollector
from ..collectors.cache_collector import CacheCollector
from ..collectors.session_collector import SessionCollector


class MetricsService:
    """
    Сервис для управления сбором метрик

    Координирует работу всех collectors и предоставляет
    единый интерфейс для получения метрик
    """

    def __init__(self):
        """Инициализация сервиса метрик"""
        self.collectors: Dict[str, BaseCollector] = {}
        self._running = False
        self._collection_task: Optional[asyncio.Task] = None

    def register_collector(self, collector: BaseCollector) -> None:
        """
        Зарегистрировать collector

        Args:
            collector: Collector для регистрации
        """
        collector_name = collector.get_collector_name()
        self.collectors[collector_name] = collector
        logger.info(f"✅ Collector зарегистрирован: {collector_name}")

    def unregister_collector(self, collector_name: str) -> bool:
        """
        Удалить collector

        Args:
            collector_name: Название collector

        Returns:
            bool: True если удалён
        """
        if collector_name in self.collectors:
            del self.collectors[collector_name]
            logger.info(f"🗑️ Collector удалён: {collector_name}")
            return True
        return False

    async def collect_all(self) -> Dict[str, Any]:
        """
        Собрать метрики со всех collectors

        Returns:
            Dict: Агрегированные метрики
        """
        metrics = {
            "timestamp": None,
            "collectors": {}
        }

        from src.common.utils import now_utc
        metrics["timestamp"] = now_utc().isoformat()

        # Собираем с каждого collector
        for collector_name, collector in self.collectors.items():
            try:
                collector_metrics = await collector.collect()
                metrics["collectors"][collector_name] = collector_metrics

            except Exception as e:
                logger.error(f"Ошибка сбора метрик с {collector_name}: {e}")
                metrics["collectors"][collector_name] = {
                    "error": str(e),
                    "collector": collector_name
                }

        return metrics

    async def collect_if_needed(self) -> Dict[str, Any]:
        """
        Собрать метрики только с тех collectors, у которых прошёл интервал

        Returns:
            Dict: Метрики от collectors которые собрали
        """
        metrics = {
            "timestamp": None,
            "collectors": {}
        }

        from src.common.utils import now_utc
        metrics["timestamp"] = now_utc().isoformat()

        # Собираем только если прошёл интервал
        for collector_name, collector in self.collectors.items():
            try:
                collector_metrics = await collector.collect_if_needed()

                if collector_metrics:  # Если собраны
                    metrics["collectors"][collector_name] = collector_metrics

            except Exception as e:
                logger.error(f"Ошибка сбора метрик с {collector_name}: {e}")

        return metrics

    async def get_collector_metrics(self, collector_name: str) -> Optional[Dict[str, Any]]:
        """
        Получить метрики конкретного collector

        Args:
            collector_name: Название collector

        Returns:
            Optional[Dict]: Метрики или None
        """
        collector = self.collectors.get(collector_name)
        if not collector:
            return None

        try:
            return await collector.collect()
        except Exception as e:
            logger.error(f"Ошибка сбора метрик с {collector_name}: {e}")
            return {"error": str(e), "collector": collector_name}

    def get_health_status(self) -> Dict[str, Any]:
        """
        Получить статус здоровья системы

        Returns:
            Dict: Статус здоровья
        """
        return {
            "collectors_count": len(self.collectors),
            "collectors_registered": list(self.collectors.keys()),
            "collection_running": self._running
        }

    async def start_background_collection(self, interval_seconds: int = 60) -> None:
        """
        Запустить фоновый сбор метрик

        Args:
            interval_seconds: Интервал сбора
        """
        if self._running:
            logger.warning("Фоновый сбор метрик уже запущен")
            return

        self._running = True
        logger.info(f"🚀 Запуск фонового сбора метрик (интервал: {interval_seconds}s)")

        self._collection_task = asyncio.create_task(
            self._background_collection_loop(interval_seconds)
        )

    async def stop_background_collection(self) -> None:
        """Остановить фоновый сбор метрик"""
        if not self._running:
            return

        self._running = False

        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass

        logger.info("⏹️ Фоновый сбор метрик остановлен")

    async def _background_collection_loop(self, interval_seconds: int) -> None:
        """
        Цикл фонового сбора метрик

        Args:
            interval_seconds: Интервал между сборами
        """
        while self._running:
            try:
                # Собираем метрики
                metrics = await self.collect_if_needed()

                # Логируем если что-то собрали
                if metrics.get("collectors"):
                    logger.debug(
                        f"Метрики собраны: {len(metrics['collectors'])} collectors"
                    )

                # Ждём до следующего сбора
                await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка в цикле сбора метрик: {e}")
                await asyncio.sleep(interval_seconds)

    def get_gpu_collector(self) -> Optional[GPUCollector]:
        """Получить GPU collector"""
        return self.collectors.get("gpu_collector")

    def get_queue_collector(self) -> Optional[QueueCollector]:
        """Получить Queue collector"""
        return self.collectors.get("queue_collector")

    def get_cache_collector(self) -> Optional[CacheCollector]:
        """Получить Cache collector"""
        return self.collectors.get("cache_collector")

    def get_session_collector(self) -> Optional[SessionCollector]:
        """Получить Session collector"""
        return self.collectors.get("session_collector")

    def __str__(self) -> str:
        return f"MetricsService(collectors={len(self.collectors)}, running={self._running})"
