"""
Collector для метрик GPU
"""
from typing import Dict, Any
from loguru import logger

from .base_collector import BaseCollector
from src.model_management.resources.gpu_monitor import GPUMonitor


class GPUCollector(BaseCollector):
    """
    Сборщик метрик GPU

    Собирает:
    - Использование памяти
    - Утилизацию GPU
    - Температуру
    """

    def __init__(self, device_id: int = 0, collection_interval_seconds: int = 30):
        """
        Инициализация GPU collector

        Args:
            device_id: ID GPU устройства
            collection_interval_seconds: Интервал сбора (секунды)
        """
        super().__init__(collection_interval_seconds)
        self.gpu_monitor = GPUMonitor(device_id=device_id)

    async def collect(self) -> Dict[str, Any]:
        """
        Собрать метрики GPU

        Returns:
            Dict: Метрики GPU
        """
        try:
            stats = self.gpu_monitor.get_full_stats()

            # Проверяем критические значения
            memory = stats.get('memory', {})
            temperature = stats.get('temperature_celsius', 0)
            utilization = stats.get('utilization_percent', 0)

            # Флаги состояния
            is_memory_critical = memory.get('usage_percent', 0) > 90
            is_temperature_high = temperature > 80
            is_overloaded = utilization > 95

            metrics = {
                "collector": self.get_collector_name(),
                "device_id": stats.get('device_id', 0),
                "memory": {
                    "used_mb": memory.get('used_mb', 0),
                    "total_mb": memory.get('total_mb', 0),
                    "free_mb": memory.get('free_mb', 0),
                    "usage_percent": memory.get('usage_percent', 0),
                    "is_critical": is_memory_critical
                },
                "utilization_percent": utilization,
                "temperature_celsius": temperature,
                "is_available": stats.get('available', False),
                "health": {
                    "memory_critical": is_memory_critical,
                    "temperature_high": is_temperature_high,
                    "overloaded": is_overloaded,
                    "overall_healthy": not (is_memory_critical or is_temperature_high or is_overloaded)
                }
            }

            logger.debug(f"GPU метрики собраны: память {memory.get('usage_percent', 0):.1f}%, утилизация {utilization:.1f}%")

            return metrics

        except Exception as e:
            logger.error(f"Ошибка сбора GPU метрик: {e}")
            return {
                "collector": self.get_collector_name(),
                "error": str(e),
                "available": False
            }

    def get_collector_name(self) -> str:
        """Название collector"""
        return "gpu_collector"
