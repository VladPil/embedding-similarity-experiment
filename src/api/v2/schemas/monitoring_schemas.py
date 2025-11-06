"""
Pydantic схемы для мониторинга
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class MetricsResponse(BaseModel):
    """Ответ с метриками"""
    timestamp: datetime
    collectors: Dict[str, Dict[str, Any]]


class SystemSummaryResponse(BaseModel):
    """Сводка по системе"""
    timestamp: datetime
    gpu: Dict[str, Any] = Field(description="GPU метрики")
    cache: Dict[str, Any] = Field(description="Cache метрики")
    queue: Dict[str, Any] = Field(description="Queue метрики")
    sessions: Dict[str, Any] = Field(description="Session метрики")
    overall_healthy: bool


class PerformanceStatsResponse(BaseModel):
    """Статистика производительности"""
    period_hours: int
    data_points: int
    gpu: Dict[str, float]
    sessions: Dict[str, float]


class TrendsResponse(BaseModel):
    """Тренды метрик"""
    gpu_utilization: str = Field(description="increasing/decreasing/stable")
    cache_hit_rate: str
    session_success_rate: str


class AlertResponse(BaseModel):
    """Алерт (проблема)"""
    severity: str = Field(description="critical/warning/info")
    component: str
    message: str
    value: float


class AlertsListResponse(BaseModel):
    """Список алертов"""
    alerts: List[AlertResponse]
    total: int
    critical_count: int
    warning_count: int


class HealthCheckResponse(BaseModel):
    """Результат health check"""
    status: str = Field(description="healthy/degraded/unhealthy")
    timestamp: datetime
    checks: Dict[str, Dict[str, Any]] = Field(
        description="Результаты проверок по компонентам"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-01-01T12:00:00",
                "checks": {
                    "gpu": {"status": "ok", "memory_usage": 45.5},
                    "cache": {"status": "ok", "hit_rate": 85.3},
                    "queue": {"status": "ok", "pending_tasks": 3}
                }
            }
        }
