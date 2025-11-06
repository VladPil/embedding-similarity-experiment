"""
Task consumers for processing messages from queues.
"""

import time
from datetime import datetime
from typing import Dict, Any

from faststream.redis import RedisRouter

from .config import get_queue_settings
from .schemas import AnalysisTaskMessage, ComparisonTaskMessage
from .models import TaskStatus
from .producers import publish_progress_update
from server.services.book_analysis.service import BookAnalysisService
from server.core.analysis.base import AnalysisType
from server.core.chunks.manager import ChunkManager
from server.core.analysis.chunk_indexer import Chunk


# Create router for consumers
settings = get_queue_settings()
router = RedisRouter()


def convert_chunks_to_objects(chunk_dicts):
    """Convert dictionary chunks to Chunk objects."""
    chunks = []
    for i, chunk_dict in enumerate(chunk_dicts):
        chunk = Chunk(
            index=i,
            text=chunk_dict['text'],
            start_pos=chunk_dict.get('start', 0),
            end_pos=chunk_dict.get('end', len(chunk_dict['text'])),
            position_ratio=i / len(chunk_dicts) if chunk_dicts else 0.0
        )
        chunks.append(chunk)
    return chunks


@router.subscriber(settings.analysis_queue)
async def process_analysis_task(message: Dict[str, Any]):
    """
    Process book analysis task.

    Args:
        message: Analysis task message
    """
    # Parse message
    task_msg = AnalysisTaskMessage(**message)
    task_id = task_msg.task_id
    start_time = time.time()

    try:
        # Update status: in progress
        await publish_progress_update(
            task_id=task_id,
            status=TaskStatus.IN_PROGRESS,
            progress=5.0,
            elapsed_time=time.time() - start_time,
            current_step="Starting analysis...",
        )

        # Validate analysis types
        valid_types = {
            "genre": AnalysisType.GENRE,
            "character": AnalysisType.CHARACTER,
            "pace": AnalysisType.PACE,
            "tension": AnalysisType.TENSION,
            "water": AnalysisType.WATER,
            "theme": AnalysisType.THEME,
        }

        selected_analyses = []
        for analysis_name in task_msg.analyses:
            if analysis_name in valid_types:
                selected_analyses.append(valid_types[analysis_name])

        # Create chunks
        await publish_progress_update(
            task_id=task_id,
            status=TaskStatus.IN_PROGRESS,
            progress=10.0,
            elapsed_time=time.time() - start_time,
            current_step="Creating text chunks...",
        )

        chunk_manager = ChunkManager(chunk_size=task_msg.chunk_size)
        chunk_dicts = chunk_manager.chunk_by_characters(task_msg.text_content)
        chunks = convert_chunks_to_objects(chunk_dicts)

        # Run analysis
        await publish_progress_update(
            task_id=task_id,
            status=TaskStatus.IN_PROGRESS,
            progress=20.0,
            elapsed_time=time.time() - start_time,
            current_step="Running analyses...",
        )

        service = BookAnalysisService()
        result = await service.analyze_book(
            text=task_msg.text_content,
            chunks=chunks,
            selected_analyses=selected_analyses,
            metadata={
                "task_id": task_id,
                "text_id": task_msg.text_id,
                "title": task_msg.text_title,
            },
        )

        # Complete
        await publish_progress_update(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            progress=100.0,
            elapsed_time=time.time() - start_time,
            current_step="Analysis completed",
        )

        return result

    except Exception as e:
        # Failed
        await publish_progress_update(
            task_id=task_id,
            status=TaskStatus.FAILED,
            progress=0.0,
            elapsed_time=time.time() - start_time,
            current_step="Analysis failed",
            error=str(e),
        )
        raise


@router.subscriber(settings.comparison_queue)
async def process_comparison_task(message: Dict[str, Any]):
    """
    Process collection comparison task.

    Args:
        message: Collection comparison task message
    """
    from .schemas import CollectionComparisonMessage, TextItemMessage
    from server.services.book_analysis.collections import (
        TextCollection,
        TextItem,
        ComparisonStrategy,
    )
    from server.services.book_analysis.comparison_service import CollectionComparisonService

    # Parse message
    task_msg = CollectionComparisonMessage(**message)
    task_id = task_msg.task_id
    start_time = time.time()

    try:
        # Update status: in progress
        await publish_progress_update(
            task_id=task_id,
            status=TaskStatus.IN_PROGRESS,
            progress=5.0,
            elapsed_time=time.time() - start_time,
            current_step="Starting comparison...",
        )

        # Create text items
        texts = []
        for text_msg in task_msg.texts:
            text_item = TextItem(
                text_id=text_msg.text_id,
                title=text_msg.title,
                content=text_msg.content,
                metadata=text_msg.metadata,
            )
            texts.append(text_item)

        # Create strategy
        strategy = ComparisonStrategy(
            analyses=task_msg.analyses,
            embedding_method=task_msg.embedding_method,
            embedding_model=task_msg.embedding_model,
            chunk_size=task_msg.chunk_size,
            compare_all_pairs=task_msg.compare_all_pairs,
            include_self_comparison=task_msg.include_self_comparison,
        )

        # Create collection
        collection = TextCollection(
            collection_id=task_msg.collection_id,
            name=task_msg.collection_name,
            texts=texts,
            strategy=strategy,
            metadata=task_msg.metadata,
        )

        # Progress callback
        async def progress_callback(step: str, current: int, total: int, message: str):
            if step == "analysis":
                progress = 10 + (current / total * 40)  # 10-50%
            else:  # comparison
                progress = 50 + (current / total * 45)  # 50-95%

            await publish_progress_update(
                task_id=task_id,
                status=TaskStatus.IN_PROGRESS,
                progress=progress,
                elapsed_time=time.time() - start_time,
                current_step=message,
            )

        # Run comparison
        service = CollectionComparisonService()

        await publish_progress_update(
            task_id=task_id,
            status=TaskStatus.IN_PROGRESS,
            progress=10.0,
            elapsed_time=time.time() - start_time,
            current_step=f"Analyzing {len(texts)} texts...",
        )

        # Analyze all texts
        await service.analyze_collection(collection, progress_callback)

        await publish_progress_update(
            task_id=task_id,
            status=TaskStatus.IN_PROGRESS,
            progress=50.0,
            elapsed_time=time.time() - start_time,
            current_step="Comparing texts...",
        )

        # Compare collection
        matrix = await service.compare_collection(collection, progress_callback)

        # Format result
        result = {
            "task_id": task_id,
            "collection": collection.to_dict(),
            "comparison_matrix": matrix.to_dict(),
        }

        # Complete
        await publish_progress_update(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            progress=100.0,
            elapsed_time=time.time() - start_time,
            current_step="Comparison completed",
        )

        return result

    except Exception as e:
        # Failed
        await publish_progress_update(
            task_id=task_id,
            status=TaskStatus.FAILED,
            progress=0.0,
            elapsed_time=time.time() - start_time,
            current_step="Comparison failed",
            error=str(e),
        )
        raise
