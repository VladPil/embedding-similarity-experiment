"""
Dependency Injection для API v2
"""
from typing import AsyncGenerator, Optional
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.connection import get_db
from src.infrastructure.queue.progress_broadcaster import ProgressBroadcaster
from src.model_management.services.llm_service import LLMService
from src.model_management.services.embedding_service import EmbeddingService
from src.analysis_domain.analysis_service import AnalysisService
from src.monitoring.services.metrics_service import MetricsService
from src.monitoring.services.analytics_service import AnalyticsService
from src.export.services.export_service import ExportService


# Глобальные singleton'ы
_progress_broadcaster = ProgressBroadcaster()
_metrics_service = MetricsService()
_analytics_service = AnalyticsService(_metrics_service)
_export_service = ExportService()

# Создаем все синглтоны для API
try:
    from src.model_management.scheduler.model_pool import ModelPool
    from src.model_management.resources.gpu_monitor import GPUMonitor

    _model_pool = ModelPool()
    _gpu_monitor = GPUMonitor()
    _llm_service = LLMService(_model_pool)
    _embedding_service = EmbeddingService(_model_pool)
except ImportError as e:
    # Для тестов без реальных зависимостей
    _model_pool = None
    _gpu_monitor = None
    _llm_service = None
    _embedding_service = None


# Dependencies для сервисов

def get_llm_service() -> Optional[LLMService]:
    """Получить LLM сервис"""
    return _llm_service


def get_embedding_service() -> Optional[EmbeddingService]:
    """Получить Embedding сервис"""
    return _embedding_service


def get_model_pool():
    """Получить Model Pool"""
    return _model_pool


def get_gpu_monitor():
    """Получить GPU Monitor"""
    return _gpu_monitor


def get_progress_broadcaster() -> ProgressBroadcaster:
    """Получить Progress Broadcaster"""
    return _progress_broadcaster


def get_metrics_service() -> MetricsService:
    """Получить Metrics Service"""
    return _metrics_service


def get_analytics_service() -> AnalyticsService:
    """Получить Analytics Service"""
    return _analytics_service


def get_export_service() -> ExportService:
    """Получить Export Service"""
    return _export_service


async def get_analysis_service(
    db: AsyncSession = Depends(get_db),
    llm_service: LLMService = Depends(get_llm_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    progress_broadcaster: ProgressBroadcaster = Depends(get_progress_broadcaster)
) -> AnalysisService:
    """Получить Analysis Service"""
    return AnalysisService(
        db_session=db,
        llm_service=llm_service,
        embedding_service=embedding_service,
        progress_broadcaster=progress_broadcaster
    )
