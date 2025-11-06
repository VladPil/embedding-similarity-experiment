"""
Message schemas for FastStream queue.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AnalysisTaskMessage(BaseModel):
    """Message schema for book analysis task."""

    task_id: str
    text_id: str
    text_title: str
    text_content: str
    analyses: List[str] = Field(
        description="List of analysis types: genre, character, pace, tension, water, theme"
    )
    chunk_size: int = Field(default=2000)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextItemMessage(BaseModel):
    """Single text in a collection."""

    text_id: str
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CollectionComparisonMessage(BaseModel):
    """Message schema for collection comparison task."""

    task_id: str
    collection_id: str
    collection_name: str
    texts: List[TextItemMessage] = Field(
        description="List of texts in collection"
    )

    # Strategy settings
    analyses: List[str] = Field(
        default_factory=lambda: ["genre", "character", "pace", "tension", "water", "theme"],
        description="List of analysis types to compare"
    )
    embedding_method: str = Field(default="hybrid")
    embedding_model: str = Field(default="multilingual-e5-small")
    chunk_size: int = Field(default=2000)
    compare_all_pairs: bool = Field(default=True)
    include_self_comparison: bool = Field(default=False)

    metadata: Dict[str, Any] = Field(default_factory=dict)


# Legacy comparison message for backward compatibility
class ComparisonTaskMessage(BaseModel):
    """Message schema for book comparison task (DEPRECATED - use CollectionComparisonMessage)."""

    task_id: str
    text1_id: str
    text2_id: str
    text1_title: str
    text2_title: str
    text1_content: str
    text2_content: str
    analyses: List[str] = Field(
        description="List of analysis types to compare"
    )
    chunk_size: int = Field(default=2000)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingComparisonMessage(BaseModel):
    """Message schema for embedding comparison task."""

    task_id: str
    text1_id: str
    text2_id: str
    text1_content: str
    text2_content: str
    method: str = Field(
        description="Comparison method: cosine, semantic, or hybrid"
    )
    model_name: str = Field(default="multilingual-e5-small")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskStatusUpdate(BaseModel):
    """Message schema for task status updates (WebSocket/pub-sub)."""

    task_id: str
    status: str
    progress: float = Field(ge=0.0, le=100.0)
    elapsed_time: float
    estimated_time: Optional[float] = None
    current_step: str = ""
    error: Optional[str] = None
