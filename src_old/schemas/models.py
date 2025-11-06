"""
Pydantic models for all API requests and responses.
Single file for simplicity - can be split later.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


# ============================================================================
# Text Models
# ============================================================================

class TextUploadRequest(BaseModel):
    """Upload text via JSON."""
    title: str = Field(..., min_length=1, max_length=500)
    text: str = Field(..., min_length=1)


class TextInfo(BaseModel):
    """Text information response."""
    id: str
    title: str
    lines: int
    length: int
    created_at: datetime = Field(default_factory=datetime.now)


class TextListResponse(BaseModel):
    """List of texts response."""
    success: bool = True
    texts: List[TextInfo]


# ============================================================================
# Embedding Models
# ============================================================================

class EmbeddingModelInfo(BaseModel):
    """Embedding model information."""
    key: str
    name: str
    dimensions: int
    description: str


class ModelsListResponse(BaseModel):
    """Available models response."""
    success: bool = True
    models: Dict[str, EmbeddingModelInfo]


# ============================================================================
# Analysis Models
# ============================================================================

class SemanticAnalysisRequest(BaseModel):
    """Semantic analysis request."""
    text_id1: str
    text_id2: str
    model: str = "multilingual-e5-small"


class StyleAnalysisRequest(BaseModel):
    """Style analysis request."""
    text_id1: str
    text_id2: str


class TFIDFAnalysisRequest(BaseModel):
    """TF-IDF analysis request."""
    text_id1: str
    text_id2: str


class EmotionAnalysisRequest(BaseModel):
    """Emotion analysis request."""
    text_id1: str
    text_id2: str
    num_segments: int = 10


class LLMAnalysisRequest(BaseModel):
    """LLM analysis request (background task)."""
    text_id1: str
    text_id2: str
    model: str = "qwen2.5-0.5b"


class CombinedAnalysisRequest(BaseModel):
    """Combined analysis request (background task)."""
    text_id1: str
    text_id2: str
    model: str = "multilingual-e5-small"
    weights: Optional[Dict[str, float]] = {
        "semantic": 0.3,
        "style": 0.25,
        "tfidf": 0.25,
        "emotion": 0.2,
    }


class MatrixAnalysisRequest(BaseModel):
    """Matrix analysis request (all vs all)."""
    text_ids: List[str]
    model: str = "multilingual-e5-small"
    analysis_type: str = "semantic"  # semantic, style, tfidf, emotion, combined


class LLMQuickSummaryRequest(BaseModel):
    """LLM quick summary request."""
    text_id: str
    model: str = "qwen2.5-0.5b"
    max_words: int = 100


class LLMExtractThemesRequest(BaseModel):
    """LLM extract themes request."""
    text_id: str
    model: str = "qwen2.5-0.5b"


class LLMExtractDifferencesRequest(BaseModel):
    """LLM extract differences request."""
    text_id1: str
    text_id2: str
    model: str = "qwen2.5-0.5b"


class LLMQuickSentimentRequest(BaseModel):
    """LLM quick sentiment request."""
    text_id: str
    model: str = "qwen2.5-0.5b"


class LLMCompareQuickRequest(BaseModel):
    """LLM quick comparison request."""
    text_id1: str
    text_id2: str
    model: str = "qwen2.5-0.5b"


class LLMGenerateReportRequest(BaseModel):
    """LLM report generation request (comparison report)."""
    text_id1: str
    text_id2: str
    model: str = "qwen2.5-1.5b"


class LLMGenerateSingleReportRequest(BaseModel):
    """LLM single text report generation request."""
    text_id: str
    model: str = "qwen2.5-1.5b"


class ChunkedAnalysisRequest(BaseModel):
    """Chunked analysis request."""
    text_id1: str
    text_id2: str
    model: str = "multilingual-e5-small"
    chunk_size: int = 10  # sentences
    overlap: int = 2  # sentences
    split_by: str = "sentences"  # sentences, paragraphs, characters
    top_n: int = 5  # top N most similar chunks


class AnalysisResponse(BaseModel):
    """Generic analysis response."""
    success: bool = True
    text1: TextInfo
    text2: TextInfo
    similarity: float
    interpretation: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# ============================================================================
# Task Models
# ============================================================================

class TaskInfo(BaseModel):
    """Background task information."""
    id: str
    name: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None


class TaskStartResponse(BaseModel):
    """Task started response."""
    success: bool = True
    task_id: str
    message: str = "Task started in background"


class TaskStatusResponse(BaseModel):
    """Task status response."""
    success: bool = True
    task: TaskInfo


class TaskListResponse(BaseModel):
    """Task list response."""
    success: bool = True
    tasks: List[TaskInfo]


# ============================================================================
# History Models
# ============================================================================

class HistoryEntry(BaseModel):
    """Analysis history entry."""
    id: str
    timestamp: datetime
    analysis_type: str
    config: Dict[str, Any]
    text_ids: List[str]
    text_titles: List[str]
    result: Optional[Dict[str, Any]] = None


class HistoryListResponse(BaseModel):
    """History list response."""
    success: bool = True
    history: List[HistoryEntry]


# ============================================================================
# Error Models
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = False
    error: str
    details: Optional[str] = None
