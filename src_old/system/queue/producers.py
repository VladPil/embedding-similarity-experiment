"""
Task producers for publishing messages to queues.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any

from .app import get_broker
from .config import get_queue_settings
from .schemas import (
    AnalysisTaskMessage,
    ComparisonTaskMessage,
    EmbeddingComparisonMessage,
    TaskStatusUpdate,
)
from .models import TaskStatus, TaskMetadata, TaskType


async def submit_analysis_task(
    text_id: str,
    text_title: str,
    text_content: str,
    analyses: List[str],
    chunk_size: int = 2000,
    metadata: Dict[str, Any] = None,
) -> str:
    """
    Submit book analysis task to queue.

    Args:
        text_id: Text ID
        text_title: Text title
        text_content: Text content
        analyses: List of analysis types
        chunk_size: Chunk size for text splitting
        metadata: Additional metadata

    Returns:
        str: Task ID
    """
    task_id = str(uuid.uuid4())
    settings = get_queue_settings()
    broker = get_broker()

    # Create task message
    message = AnalysisTaskMessage(
        task_id=task_id,
        text_id=text_id,
        text_title=text_title,
        text_content=text_content,
        analyses=analyses,
        chunk_size=chunk_size,
        metadata=metadata or {},
    )

    # Publish to analysis queue
    await broker.publish(
        message=message.model_dump(),
        channel=settings.analysis_queue,
    )

    # Publish initial status update
    status_update = TaskStatusUpdate(
        task_id=task_id,
        status=TaskStatus.PENDING.value,
        progress=0.0,
        elapsed_time=0.0,
        current_step="Task queued",
    )

    await broker.publish(
        message=status_update.model_dump(),
        channel=settings.progress_channel,
    )

    return task_id


async def submit_comparison_task(
    text1_id: str,
    text2_id: str,
    text1_title: str,
    text2_title: str,
    text1_content: str,
    text2_content: str,
    analyses: List[str],
    chunk_size: int = 2000,
    metadata: Dict[str, Any] = None,
) -> str:
    """
    Submit book comparison task to queue.

    Args:
        text1_id: First text ID
        text2_id: Second text ID
        text1_title: First text title
        text2_title: Second text title
        text1_content: First text content
        text2_content: Second text content
        analyses: List of analysis types to compare
        chunk_size: Chunk size for text splitting
        metadata: Additional metadata

    Returns:
        str: Task ID
    """
    task_id = str(uuid.uuid4())
    settings = get_queue_settings()
    broker = get_broker()

    # Create task message
    message = ComparisonTaskMessage(
        task_id=task_id,
        text1_id=text1_id,
        text2_id=text2_id,
        text1_title=text1_title,
        text2_title=text2_title,
        text1_content=text1_content,
        text2_content=text2_content,
        analyses=analyses,
        chunk_size=chunk_size,
        metadata=metadata or {},
    )

    # Publish to comparison queue
    await broker.publish(
        message=message.model_dump(),
        channel=settings.comparison_queue,
    )

    # Publish initial status update
    status_update = TaskStatusUpdate(
        task_id=task_id,
        status=TaskStatus.PENDING.value,
        progress=0.0,
        elapsed_time=0.0,
        current_step="Task queued",
    )

    await broker.publish(
        message=status_update.model_dump(),
        channel=settings.progress_channel,
    )

    return task_id


async def publish_progress_update(
    task_id: str,
    status: TaskStatus,
    progress: float,
    elapsed_time: float,
    current_step: str = "",
    estimated_time: float = None,
    error: str = None,
):
    """
    Publish task progress update to progress channel.

    Args:
        task_id: Task ID
        status: Task status
        progress: Progress percentage (0-100)
        elapsed_time: Elapsed time in seconds
        current_step: Current processing step
        estimated_time: Estimated total time
        error: Error message if failed
    """
    settings = get_queue_settings()
    broker = get_broker()

    status_update = TaskStatusUpdate(
        task_id=task_id,
        status=status.value,
        progress=progress,
        elapsed_time=elapsed_time,
        estimated_time=estimated_time,
        current_step=current_step,
        error=error,
    )

    await broker.publish(
        message=status_update.model_dump(),
        channel=settings.progress_channel,
    )
