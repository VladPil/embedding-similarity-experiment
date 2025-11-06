"""
Pydantic схемы для анализа
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class AnalysisSessionCreateRequest(BaseModel):
    """Запрос на создание сессии анализа"""
    name: str = Field(..., min_length=1, max_length=200, description="Название сессии")
    text_ids: List[str] = Field(..., min_items=1, max_items=5, description="ID текстов для анализа (1-5)")
    analyzer_types: List[str] = Field(..., min_items=1, description="Типы анализаторов")
    mode: str = Field("full_text", description="Режим анализа: full_text/chunked")
    comparator_type: Optional[str] = Field(None, description="Тип компаратора для сравнения")
    chunking_config: Optional[Dict[str, Any]] = Field(None, description="Конфигурация чанкинга")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Сравнение двух романов",
                "text_ids": ["text_123", "text_456"],
                "analyzer_types": ["GenreAnalyzer", "StyleAnalyzer", "CharacterAnalyzer"],
                "mode": "chunked",
                "comparator_type": "CosineComparator",
                "chunking_config": {
                    "chunk_size": 1000,
                    "overlap": 100
                }
            }
        }


class AnalysisSessionResponse(BaseModel):
    """Ответ с информацией о сессии"""
    id: str
    name: str
    status: str = Field(description="Статус: draft/queued/running/completed/failed")
    mode: str
    text_ids: List[str]
    analyzer_types: List[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Прогресс 0-1")
    error_message: Optional[str]

    class Config:
        json_schema_extra = {
            "example": {
                "id": "session_789",
                "name": "Сравнение двух романов",
                "status": "running",
                "mode": "chunked",
                "text_ids": ["text_123", "text_456"],
                "analyzer_types": ["GenreAnalyzer", "StyleAnalyzer"],
                "created_at": "2025-01-01T12:00:00",
                "started_at": "2025-01-01T12:01:00",
                "completed_at": None,
                "progress": 0.45,
                "error_message": None
            }
        }


class AnalysisResultResponse(BaseModel):
    """Результат анализа"""
    text_id: str
    analyzer_type: str
    mode: str
    data: Dict[str, Any]
    interpretation: Optional[str]
    created_at: datetime


class ComparisonMatrixResponse(BaseModel):
    """Матрица сравнения"""
    text_ids: List[str]
    similarity_matrix: List[List[float]]
    method: str
    average_similarity: float
    most_similar_pairs: List[Dict[str, Any]]


class AnalysisSessionDetailResponse(AnalysisSessionResponse):
    """Детальная информация о сессии с результатами"""
    results: Dict[str, List[AnalysisResultResponse]] = Field(
        description="Результаты по text_id"
    )
    comparison_matrix: Optional[ComparisonMatrixResponse]
    execution_time_seconds: Optional[float]


class AnalysisSessionListResponse(BaseModel):
    """Список сессий"""
    sessions: List[AnalysisSessionResponse]
    total: int
    offset: int
    limit: int


class SessionRunRequest(BaseModel):
    """Запрос на запуск сессии"""
    session_id: str
    async_mode: bool = Field(True, description="Асинхронный режим (через очередь)")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_789",
                "async_mode": True
            }
        }
