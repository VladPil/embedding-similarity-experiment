"""
Сервис для аналитики и отчётности
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from loguru import logger

from .metrics_service import MetricsService
from src.common.utils import now_utc


class AnalyticsService:
    """
    Сервис аналитики

    Предоставляет аналитические отчёты на основе собранных метрик
    """

    def __init__(self, metrics_service: MetricsService):
        """
        Инициализация сервиса аналитики

        Args:
            metrics_service: Сервис метрик
        """
        self.metrics_service = metrics_service

        # История метрик (в памяти, последние 24 часа)
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history_size = 1440  # 24 часа при сборе каждую минуту

    async def collect_and_store(self) -> None:
        """Собрать метрики и сохранить в историю"""
        metrics = await self.metrics_service.collect_all()

        # Добавляем в историю
        self.metrics_history.append(metrics)

        # Ограничиваем размер истории
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]

        logger.debug(f"Метрики сохранены в историю (размер: {len(self.metrics_history)})")

    async def get_system_summary(self) -> Dict[str, Any]:
        """
        Получить сводку по системе

        Returns:
            Dict: Сводка
        """
        # Собираем текущие метрики
        current_metrics = await self.metrics_service.collect_all()

        collectors = current_metrics.get("collectors", {})

        # GPU
        gpu_metrics = collectors.get("gpu_collector", {})
        gpu_summary = {
            "available": gpu_metrics.get("is_available", False),
            "memory_usage": gpu_metrics.get("memory", {}).get("usage_percent", 0),
            "utilization": gpu_metrics.get("utilization_percent", 0),
            "temperature": gpu_metrics.get("temperature_celsius", 0),
            "healthy": gpu_metrics.get("health", {}).get("overall_healthy", True)
        }

        # Cache
        cache_metrics = collectors.get("cache_collector", {})
        cache_summary = {
            "hit_rate": cache_metrics.get("hit_rate", 0),
            "total_requests": cache_metrics.get("stats", {}).get("total_requests", 0),
            "redis_available": cache_metrics.get("redis", {}).get("available", False)
        }

        # Queue
        queue_metrics = collectors.get("queue_collector", {})
        queue_summary = {
            "tasks_completed": queue_metrics.get("tasks", {}).get("completed", 0),
            "tasks_failed": queue_metrics.get("tasks", {}).get("failed", 0),
            "success_rate": queue_metrics.get("tasks", {}).get("success_rate", 0),
            "pending": queue_metrics.get("queue", {}).get("pending", 0)
        }

        # Sessions
        session_metrics = collectors.get("session_collector", {})
        session_summary = {
            "total_completed": session_metrics.get("sessions", {}).get("completed", 0),
            "success_rate": session_metrics.get("sessions", {}).get("success_rate", 0),
            "avg_execution_time": session_metrics.get("sessions", {}).get("avg_execution_time_seconds", 0),
            "top_analyzers": session_metrics.get("analyzers", {}).get("top_5", [])
        }

        return {
            "timestamp": current_metrics.get("timestamp"),
            "gpu": gpu_summary,
            "cache": cache_summary,
            "queue": queue_summary,
            "sessions": session_summary,
            "overall_healthy": self._assess_overall_health(gpu_summary, cache_summary, queue_summary)
        }

    def _assess_overall_health(
        self,
        gpu: Dict,
        cache: Dict,
        queue: Dict
    ) -> bool:
        """
        Оценить общее здоровье системы

        Args:
            gpu: GPU метрики
            cache: Cache метрики
            queue: Queue метрики

        Returns:
            bool: True если система здорова
        """
        # Проверяем критические параметры
        gpu_healthy = gpu.get("healthy", True)
        cache_hit_rate_ok = cache.get("hit_rate", 0) > 50  # >50% hit rate
        queue_success_rate_ok = queue.get("success_rate", 0) > 80  # >80% success

        return gpu_healthy and cache_hit_rate_ok and queue_success_rate_ok

    async def get_performance_stats(self, last_hours: int = 24) -> Dict[str, Any]:
        """
        Получить статистику производительности за период

        Args:
            last_hours: Количество часов назад

        Returns:
            Dict: Статистика производительности
        """
        # Фильтруем историю по времени
        cutoff_time = now_utc() - timedelta(hours=last_hours)
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.get("timestamp", "2000-01-01")) >= cutoff_time
        ]

        if not recent_metrics:
            return {"error": "Недостаточно данных для анализа"}

        # Агрегируем метрики
        gpu_utilizations = []
        session_times = []

        for metrics in recent_metrics:
            collectors = metrics.get("collectors", {})

            # GPU утилизация
            gpu = collectors.get("gpu_collector", {})
            if "utilization_percent" in gpu:
                gpu_utilizations.append(gpu["utilization_percent"])

            # Время выполнения сессий
            sessions = collectors.get("session_collector", {})
            avg_time = sessions.get("sessions", {}).get("avg_execution_time_seconds", 0)
            if avg_time > 0:
                session_times.append(avg_time)

        return {
            "period_hours": last_hours,
            "data_points": len(recent_metrics),
            "gpu": {
                "avg_utilization": sum(gpu_utilizations) / len(gpu_utilizations) if gpu_utilizations else 0,
                "max_utilization": max(gpu_utilizations) if gpu_utilizations else 0,
                "min_utilization": min(gpu_utilizations) if gpu_utilizations else 0
            },
            "sessions": {
                "avg_execution_time": sum(session_times) / len(session_times) if session_times else 0,
                "max_execution_time": max(session_times) if session_times else 0,
                "min_execution_time": min(session_times) if session_times else 0
            }
        }

    async def get_trends(self) -> Dict[str, Any]:
        """
        Получить тренды метрик

        Returns:
            Dict: Тренды (растёт/падает/стабильно)
        """
        if len(self.metrics_history) < 10:
            return {"error": "Недостаточно данных для анализа трендов"}

        # Берём последние 10 точек
        recent = self.metrics_history[-10:]

        # Извлекаем ключевые метрики
        gpu_utils = []
        cache_hit_rates = []
        session_success_rates = []

        for metrics in recent:
            collectors = metrics.get("collectors", {})

            gpu = collectors.get("gpu_collector", {})
            if "utilization_percent" in gpu:
                gpu_utils.append(gpu["utilization_percent"])

            cache = collectors.get("cache_collector", {})
            if "hit_rate" in cache:
                cache_hit_rates.append(cache["hit_rate"])

            sessions = collectors.get("session_collector", {})
            sr = sessions.get("sessions", {}).get("success_rate", 0)
            if sr > 0:
                session_success_rates.append(sr)

        return {
            "gpu_utilization": self._calculate_trend(gpu_utils),
            "cache_hit_rate": self._calculate_trend(cache_hit_rates),
            "session_success_rate": self._calculate_trend(session_success_rates)
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """
        Вычислить тренд

        Args:
            values: Список значений

        Returns:
            str: "increasing", "decreasing", "stable"
        """
        if len(values) < 3:
            return "stable"

        # Простой линейный тренд
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)

        diff = second_half - first_half

        if diff > 5:
            return "increasing"
        elif diff < -5:
            return "decreasing"
        else:
            return "stable"

    async def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Получить список алертов (проблем)

        Returns:
            List[Dict]: Список алертов
        """
        alerts = []

        # Собираем текущие метрики
        current = await self.metrics_service.collect_all()
        collectors = current.get("collectors", {})

        # Проверяем GPU
        gpu = collectors.get("gpu_collector", {})
        health = gpu.get("health", {})

        if health.get("memory_critical"):
            alerts.append({
                "severity": "critical",
                "component": "gpu",
                "message": "Критическое использование GPU памяти (>90%)",
                "value": gpu.get("memory", {}).get("usage_percent", 0)
            })

        if health.get("temperature_high"):
            alerts.append({
                "severity": "warning",
                "component": "gpu",
                "message": "Высокая температура GPU (>80°C)",
                "value": gpu.get("temperature_celsius", 0)
            })

        # Проверяем Cache
        cache = collectors.get("cache_collector", {})
        hit_rate = cache.get("hit_rate", 0)

        if hit_rate < 30:
            alerts.append({
                "severity": "warning",
                "component": "cache",
                "message": f"Низкий hit rate кэша ({hit_rate:.1f}%)",
                "value": hit_rate
            })

        # Проверяем Queue
        queue = collectors.get("queue_collector", {})
        success_rate = queue.get("tasks", {}).get("success_rate", 100)

        if success_rate < 70:
            alerts.append({
                "severity": "critical",
                "component": "queue",
                "message": f"Низкий success rate задач ({success_rate:.1f}%)",
                "value": success_rate
            })

        return alerts

    def clear_history(self) -> None:
        """Очистить историю метрик"""
        self.metrics_history.clear()
        logger.info("История метрик очищена")
