"""
Dependency Injection для API v2
"""
from typing import AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.connection import get_db
from src.infrastructure.queue.progress_broadcaster import ProgressBroadcaster
from src.model_management.llm_service import LLMService, llm_service as _llm_service
from src.model_management.embedding_service import EmbeddingService, embedding_service as _embedding_service
from src.analysis_domain.analysis_service import AnalysisService
from src.monitoring.services.metrics_service import MetricsService
from src.monitoring.services.analytics_service import AnalyticsService
from src.export.services.export_service import ExportService


# Глобальные singleton'ы
_progress_broadcaster = ProgressBroadcaster()
_metrics_service = MetricsService()
_analytics_service = AnalyticsService(_metrics_service)
_export_service = ExportService()


# Dependencies для сервисов

def get_llm_service() -> LLMService:
    """Получить LLM сервис"""
    return _llm_service


def get_embedding_service() -> EmbeddingService:
    """Получить Embedding сервис"""
    return _embedding_service


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
