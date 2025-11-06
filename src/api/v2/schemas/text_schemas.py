"""
Pydantic схемы для работы с текстами
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class TextCreateRequest(BaseModel):
    """Запрос на создание текста"""
    title: str = Field(..., min_length=1, max_length=500, description="Название текста")
    content: Optional[str] = Field(None, description="Содержимое текста (для коротких)")
    file_path: Optional[str] = Field(None, description="Путь к файлу (для длинных)")
    storage_type: str = Field("database", description="Тип хранения: database/file")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Метаданные")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Мой роман",
                "content": "Это короткий текст...",
                "storage_type": "database",
                "metadata": {"author": "Автор", "genre": "фантастика"}
            }
        }


class FB2CreateRequest(BaseModel):
    """Запрос на создание FB2 книги"""
    title: str = Field(..., min_length=1, max_length=500)
    file_path: str = Field(..., description="Путь к FB2 файлу")
    parse_metadata: bool = Field(True, description="Парсить метаданные из FB2")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Книга.fb2",
                "file_path": "/path/to/book.fb2",
                "parse_metadata": True
            }
        }


class TextResponse(BaseModel):
    """Ответ с информацией о тексте"""
    id: str
    title: str
    storage_type: str
    content_length: int = Field(description="Длина текста в символах")
    metadata: Dict[str, Any]
    created_at: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "id": "text_123",
                "title": "Мой роман",
                "storage_type": "database",
                "content_length": 5000,
                "metadata": {"author": "Автор"},
                "created_at": "2025-01-01T12:00:00"
            }
        }


class TextListResponse(BaseModel):
    """Список текстов"""
    texts: list[TextResponse]
    total: int
    offset: int
    limit: int


class ChunkingRequest(BaseModel):
    """Запрос на чанкинг текста"""
    text_id: str
    chunk_size: int = Field(1000, ge=100, le=5000, description="Размер чанка в символах")
    overlap: int = Field(100, ge=0, le=500, description="Перекрытие между чанками")
    boundary_type: str = Field("sentence", description="Тип границы: sentence/paragraph/none")

    class Config:
        json_schema_extra = {
            "example": {
                "text_id": "text_123",
                "chunk_size": 1000,
                "overlap": 100,
                "boundary_type": "sentence"
            }
        }


class ChunkResponse(BaseModel):
    """Ответ с информацией о чанке"""
    index: int
    content: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]


class ChunkingResponse(BaseModel):
    """Ответ на чанкинг"""
    text_id: str
    chunks: list[ChunkResponse]
    total_chunks: int
    strategy_used: str
