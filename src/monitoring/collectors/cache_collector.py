"""
Collector для метрик кэша (Redis)
"""
from typing import Dict, Any
from loguru import logger

from .base_collector import BaseCollector


class CacheCollector(BaseCollector):
    """
    Сборщик метрик кэша (Redis L1 + PostgreSQL L2)

    Собирает:
    - Hit rate кэша
    - Размер кэша
    - Количество ключей
    - Статистику использования памяти
    """

    def __init__(
        self,
        redis_client: Any = None,
        collection_interval_seconds: int = 60
    ):
        """
        Инициализация cache collector

        Args:
            redis_client: Redis клиент
            collection_interval_seconds: Интервал сбора
        """
        super().__init__(collection_interval_seconds)
        self.redis_client = redis_client

        # Счётчики
        self.total_hits = 0
        self.total_misses = 0

    async def collect(self) -> Dict[str, Any]:
        """
        Собрать метрики кэша

        Returns:
            Dict: Метрики кэша
        """
        try:
            # Статистика Redis
            redis_info = await self._get_redis_info()

            metrics = {
                "collector": self.get_collector_name(),
                "hit_rate": self._calculate_hit_rate(),
                "stats": {
                    "total_hits": self.total_hits,
                    "total_misses": self.total_misses,
                    "total_requests": self.total_hits + self.total_misses
                },
                "redis": redis_info
            }

            logger.debug(f"Cache метрики собраны: hit rate {metrics['hit_rate']:.1f}%")

            return metrics

        except Exception as e:
            logger.error(f"Ошибка сбора cache метрик: {e}")
            return {
                "collector": self.get_collector_name(),
                "error": str(e)
            }

    async def _get_redis_info(self) -> Dict[str, Any]:
        """
        Получить информацию о Redis

        Returns:
            Dict: Информация о Redis
        """
        if not self.redis_client:
            return {"available": False}

        try:
            # Получаем информацию из Redis
            info = await self.redis_client.info()

            return {
                "available": True,
                "used_memory_mb": info.get('used_memory', 0) / (1024 * 1024),
                "keys_count": await self._get_keys_count(),
                "connected_clients": info.get('connected_clients', 0),
                "uptime_seconds": info.get('uptime_in_seconds', 0)
            }

        except Exception as e:
            logger.warning(f"Не удалось получить Redis info: {e}")
            return {"available": False, "error": str(e)}

    async def _get_keys_count(self) -> int:
        """
        Получить количество ключей в Redis

        Returns:
            int: Количество ключей
        """
        if not self.redis_client:
            return 0

        try:
            return await self.redis_client.dbsize()
        except Exception:
            return 0

    def _calculate_hit_rate(self) -> float:
        """
        Вычислить hit rate кэша

        Returns:
            float: Hit rate в процентах (0-100)
        """
        total = self.total_hits + self.total_misses
        if total == 0:
            return 0.0

        return (self.total_hits / total) * 100

    def record_hit(self) -> None:
        """Зарегистрировать попадание в кэш"""
        self.total_hits += 1

    def record_miss(self) -> None:
        """Зарегистрировать промах кэша"""
        self.total_misses += 1

    def get_collector_name(self) -> str:
        """Название collector"""
        return "cache_collector"
