"""
Task models for queue system.
"""

from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task status enumeration."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Task type enumeration."""

    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    EMBEDDING = "embedding"


class TaskProgress(BaseModel):
    """Task progress information."""

    task_id: str
    status: TaskStatus
    progress: float = Field(ge=0.0, le=100.0, description="Progress percentage")
    elapsed_time: float = Field(ge=0.0, description="Elapsed time in seconds")
    estimated_time: Optional[float] = Field(None, description="Estimated total time")
    current_step: str = Field(default="", description="Current processing step")
    error: Optional[str] = Field(None, description="Error message if failed")
    started_at: datetime
    completed_at: Optional[datetime] = None


class TaskResult(BaseModel):
    """Task result information."""

    task_id: str
    task_type: TaskType
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None  # in seconds


class TaskMetadata(BaseModel):
    """Task metadata for tracking."""

    task_id: str
    task_type: TaskType
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    priority: int = Field(default=5, ge=1, le=10)
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
