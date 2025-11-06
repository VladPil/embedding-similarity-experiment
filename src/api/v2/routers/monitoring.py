"""
Роутер для мониторинга системы
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from loguru import logger

from ..schemas.monitoring_schemas import (
    MetricsResponse,
    SystemSummaryResponse,
    PerformanceStatsResponse,
    TrendsResponse,
    AlertsListResponse,
    HealthCheckResponse
)
from ..dependencies import get_metrics_service, get_analytics_service
from src.monitoring.services.metrics_service import MetricsService
from src.monitoring.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    metrics_service: MetricsService = Depends(get_metrics_service)
):
    """
    Получить текущие метрики системы

    Возвращает метрики от всех коллекторов (GPU, Cache, Queue, Session)
    """
    try:
        metrics = await metrics_service.collect_all()

        return MetricsResponse(
            timestamp=metrics["timestamp"],
            collectors=metrics["collectors"]
        )

    except Exception as e:
        logger.error(f"Ошибка получения метрик: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary", response_model=SystemSummaryResponse)
async def get_system_summary(
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Получить сводку по системе

    Возвращает агрегированную информацию по всем компонентам
    """
    try:
        summary = await analytics_service.get_system_summary()

        return SystemSummaryResponse(
            timestamp=summary["timestamp"],
            gpu=summary.get("gpu", {}),
            cache=summary.get("cache", {}),
            queue=summary.get("queue", {}),
            sessions=summary.get("sessions", {}),
            overall_healthy=summary.get("overall_healthy", True)
        )

    except Exception as e:
        logger.error(f"Ошибка получения сводки: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance", response_model=PerformanceStatsResponse)
async def get_performance_stats(
    period_hours: int = Query(24, ge=1, le=168),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Получить статистику производительности

    - **period_hours**: Период для анализа (1-168 часов, по умолчанию 24)

    Возвращает средние значения метрик за указанный период
    """
    try:
        logger.info(f"Получение статистики за {period_hours} часов")

        stats = await analytics_service.get_performance_stats(last_hours=period_hours)

        return PerformanceStatsResponse(
            period_hours=stats.get("period_hours", period_hours),
            data_points=stats.get("data_points", 0),
            gpu=stats.get("gpu", {
                "avg_memory_usage": 0.0,
                "avg_temperature": 0.0,
                "peak_memory_usage": 0.0
            }),
            sessions=stats.get("sessions", {
                "total_completed": 0,
                "avg_execution_time": 0.0,
                "success_rate": 0.0
            })
        )

    except Exception as e:
        logger.error(f"Ошибка получения статистики производительности: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends", response_model=TrendsResponse)
async def get_trends(
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Получить тренды метрик

    Анализирует изменения метрик за последние 24 часа
    Возвращает: increasing/decreasing/stable для каждой метрики
    """
    try:
        trends = await analytics_service.get_trends()

        return TrendsResponse(
            gpu_utilization=trends.get("gpu_utilization", "stable"),
            cache_hit_rate=trends.get("cache_hit_rate", "stable"),
            session_success_rate=trends.get("session_success_rate", "stable")
        )

    except Exception as e:
        logger.error(f"Ошибка получения трендов: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts", response_model=AlertsListResponse)
async def get_alerts(
    severity: str = Query(None, description="Фильтр по severity"),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Получить список алертов

    - **severity**: Фильтр по severity (critical/warning/info)

    Возвращает активные алерты о проблемах в системе
    """
    try:
        logger.info(f"Получение алертов, severity={severity}")

        alerts = await analytics_service.get_alerts()

        # Фильтруем по severity если указан
        if severity:
            alerts = [a for a in alerts if a.get("severity") == severity]

        # Подсчитываем количество по типам
        critical_count = sum(1 for a in alerts if a.get("severity") == "critical")
        warning_count = sum(1 for a in alerts if a.get("severity") == "warning")

        return AlertsListResponse(
            alerts=alerts,
            total=len(alerts),
            critical_count=critical_count,
            warning_count=warning_count
        )

    except Exception as e:
        logger.error(f"Ошибка получения алертов: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """
    Проверка здоровья системы

    Возвращает статус: healthy/degraded/unhealthy
    Проверяет все критичные компоненты
    """
    try:
        from src.common.utils import now_utc
        from src.infrastructure.database.connection import db_connection
        from src.infrastructure.cache.redis_client import redis_client
        from src.model_management.gpu_monitor import gpu_monitor

        checks = {}

        # Проверка GPU
        try:
            gpu_available = gpu_monitor.is_available()
            checks["gpu"] = {
                "status": "ok" if gpu_available else "warning",
                "message": "GPU доступна" if gpu_available else "GPU не доступна, используется CPU"
            }
        except Exception as e:
            checks["gpu"] = {"status": "error", "message": f"Ошибка проверки GPU: {e}"}

        # Проверка Cache (Redis)
        try:
            redis_healthy = await redis_client.health_check()
            checks["cache"] = {
                "status": "ok" if redis_healthy else "error",
                "message": "Redis подключен" if redis_healthy else "Redis недоступен"
            }
        except Exception as e:
            checks["cache"] = {"status": "error", "message": f"Ошибка подключения к Redis: {e}"}

        # Проверка Database
        try:
            db_healthy = await db_connection.health_check()
            checks["database"] = {
                "status": "ok" if db_healthy else "error",
                "message": "PostgreSQL подключена" if db_healthy else "PostgreSQL недоступна"
            }
        except Exception as e:
            checks["database"] = {"status": "error", "message": f"Ошибка подключения к БД: {e}"}

        # Проверка Queue
        checks["queue"] = {
            "status": "ok",
            "message": "Очередь работает (не реализовано)"
        }

        # Определяем общий статус
        statuses = [c["status"] for c in checks.values()]
        if "error" in statuses:
            overall_status = "unhealthy"
        elif "warning" in statuses:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        return HealthCheckResponse(
            status=overall_status,
            timestamp=now_utc(),
            checks=checks
        )

    except Exception as e:
        logger.error(f"Ошибка health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collectors/start")
async def start_background_collection(
    metrics_service: MetricsService = Depends(get_metrics_service)
):
    """
    Запустить фоновый сбор метрик

    Запускает MetricsService в фоновом режиме
    Метрики будут собираться автоматически по расписанию
    """
    try:
        logger.info("Запуск фонового сбора метрик")

        # Примечание: фоновый сбор требует asyncio задачи
        # Для простоты оставляем как placeholder
        # В production нужно использовать BackgroundTasks или Celery
        return {
            "status": "started",
            "message": "Фоновый сбор метрик запущен (placeholder - реализуйте через BackgroundTasks)"
        }

    except Exception as e:
        logger.error(f"Ошибка запуска фонового сбора: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collectors/stop")
async def stop_background_collection(
    metrics_service: MetricsService = Depends(get_metrics_service)
):
    """
    Остановить фоновый сбор метрик

    Останавливает автоматический сбор метрик
    """
    try:
        logger.info("Остановка фонового сбора метрик")

        # Примечание: фоновый сбор требует asyncio задачи
        # Для простоты оставляем как placeholder
        return {
            "status": "stopped",
            "message": "Фоновый сбор метрик остановлен (placeholder)"
        }

    except Exception as e:
        logger.error(f"Ошибка остановки фонового сбора: {e}")
        raise HTTPException(status_code=500, detail=str(e))
