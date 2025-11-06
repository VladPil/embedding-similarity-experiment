"""
API endpoints for book content analysis.
Analyzes single text with multiple analysis strategies.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from server.db.database import get_db
from server.services.text_service import TextService
from server.services.book_analysis.service import BookAnalysisService
from server.core.analysis.base import AnalysisType
from server.core.chunks.manager import ChunkManager
from server.core.analysis.chunk_indexer import Chunk


router = APIRouter()


def get_text_service(db: Session = Depends(get_db)) -> TextService:
    """Get text service instance."""
    return TextService(db)


class BookAnalysisRequest(BaseModel):
    """Request model for book analysis."""

    text_id: str = Field(..., description="ID of text to analyze")
    analyses: List[str] = Field(
        default=["genre", "character", "pace", "tension", "water", "theme"],
        description="List of analysis types to run"
    )
    chunk_size: int = Field(default=2000, description="Chunk size for text splitting")


class BookAnalysisResponse(BaseModel):
    """Response model for book analysis."""

    text_id: str
    text_title: str
    success: bool
    results: dict
    summary: Optional[str] = None


@router.post(
    "/analyze-book",
    response_model=BookAnalysisResponse,
    summary="Analyze book content",
    description="Analyze single text with multiple content analysis strategies"
)
async def analyze_book_content(
    request: BookAnalysisRequest,
    text_service: TextService = Depends(get_text_service)
):
    """
    Analyze book content with selected strategies.

    Available analyses:
    - genre: Определение жанра
    - character: Анализ персонажей
    - pace: Анализ темпа повествования
    - tension: Анализ напряжения
    - water: Анализ качества текста
    - theme: Анализ тем

    Args:
        request: Book analysis request
        text_service: Text service instance

    Returns:
        Book analysis response with results
    """
    try:
        # Get text
        text = text_service.get_text(request.text_id)
        if not text:
            raise HTTPException(
                status_code=404,
                detail=f"Text with id {request.text_id} not found"
            )

        # Get text content
        text_content = text_service.get_text_content(text)

        # Validate analysis types
        valid_types = {
            "genre": AnalysisType.GENRE,
            "character": AnalysisType.CHARACTER,
            "pace": AnalysisType.PACE,
            "tension": AnalysisType.TENSION,
            "water": AnalysisType.WATER,
            "theme": AnalysisType.THEME
        }

        selected_analyses = []
        for analysis_name in request.analyses:
            if analysis_name not in valid_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid analysis type: {analysis_name}. "
                           f"Valid types: {list(valid_types.keys())}"
                )
            selected_analyses.append(valid_types[analysis_name])

        # Create chunks
        chunk_manager = ChunkManager(chunk_size=request.chunk_size)
        chunk_dicts = chunk_manager.chunk_by_characters(text_content)

        # Convert dict chunks to Chunk objects
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

        # Run analysis
        service = BookAnalysisService()
        result = await service.analyze_book(
            text=text_content,
            chunks=chunks,
            selected_analyses=selected_analyses,
            metadata={
                "text_id": request.text_id,
                "title": text.title
            }
        )

        # Format response
        return BookAnalysisResponse(
            text_id=request.text_id,
            text_title=text.title,
            success=result.get("success", True),
            results=result.get("results", {}),
            summary=result.get("summary")
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.get(
    "/available-analyses",
    summary="Get available analysis types",
    description="Get list of available book analysis types"
)
async def get_available_analyses():
    """Get list of available analysis types."""
    return {
        "analyses": [
            {
                "name": "genre",
                "label": "Жанр",
                "description": "Определение жанра произведения",
                "icon": "🎭"
            },
            {
                "name": "character",
                "label": "Персонажи",
                "description": "Анализ персонажей и их развития",
                "icon": "👥"
            },
            {
                "name": "pace",
                "label": "Темп",
                "description": "Анализ темпа повествования",
                "icon": "⚡"
            },
            {
                "name": "tension",
                "label": "Напряжение",
                "description": "Анализ драматического напряжения",
                "icon": "🔥"
            },
            {
                "name": "water",
                "label": "Качество",
                "description": "Анализ качества текста и 'воды'",
                "icon": "💧"
            },
            {
                "name": "theme",
                "label": "Темы",
                "description": "Анализ основных тем произведения",
                "icon": "📚"
            }
        ]
    }


class ComparisonRequest(BaseModel):
    """Request model for collection comparison."""

    collection_name: str = Field(..., description="Collection name")
    text_ids: List[str] = Field(..., description="List of text IDs to compare (2 or more)")

    # Strategy settings
    analyses: List[str] = Field(
        default=["genre", "character", "pace", "tension", "water", "theme"],
        description="List of analysis types"
    )
    embedding_method: str = Field(default="hybrid", description="Embedding comparison method")
    embedding_model: str = Field(default="multilingual-e5-small", description="Embedding model")
    chunk_size: int = Field(default=2000, description="Chunk size")
    compare_all_pairs: bool = Field(default=True, description="Compare all pairs (matrix)")


class ComparisonResponse(BaseModel):
    """Response model for collection comparison."""

    task_id: str
    collection_id: str
    collection_name: str
    text_count: int
    message: str


@router.post(
    "/compare-collection",
    response_model=ComparisonResponse,
    summary="Compare collection of texts",
    description="Compare multiple texts using matrix comparison"
)
async def compare_text_collection(
    request: ComparisonRequest,
    text_service: TextService = Depends(get_text_service)
):
    """
    Compare collection of texts.

    Creates a comparison matrix for N texts using specified analyses.
    Works with 2, 3, 4, or more texts - generates all pairwise comparisons.

    Args:
        request: Comparison request
        text_service: Text service instance

    Returns:
        Task ID for tracking comparison progress
    """
    import uuid
    from server.system.queue.task_manager import FastStreamTaskManager
    from server.system.queue.schemas import TextItemMessage, CollectionComparisonMessage

    try:
        # Validate minimum 2 texts
        if len(request.text_ids) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 texts are required for comparison"
            )

        # Load all texts
        texts = []
        for text_id in request.text_ids:
            text = text_service.get_text(text_id)
            if not text:
                raise HTTPException(
                    status_code=404,
                    detail=f"Text with id {text_id} not found"
                )

            # Get content
            content = text_service.get_text_content(text)

            # Create text item message
            text_msg = TextItemMessage(
                text_id=text_id,
                title=text.title,
                content=content,
                metadata={}
            )
            texts.append(text_msg)

        # Generate collection ID and task ID
        collection_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())

        # Create comparison message
        comparison_msg = CollectionComparisonMessage(
            task_id=task_id,
            collection_id=collection_id,
            collection_name=request.collection_name,
            texts=texts,
            analyses=request.analyses,
            embedding_method=request.embedding_method,
            embedding_model=request.embedding_model,
            chunk_size=request.chunk_size,
            compare_all_pairs=request.compare_all_pairs,
            include_self_comparison=False,
        )

        # Submit to task queue
        task_manager = FastStreamTaskManager()

        # Publish to comparison queue
        from server.system.queue.app import get_broker
        from server.system.queue.config import get_queue_settings

        broker = get_broker()
        settings = get_queue_settings()

        await broker.publish(
            message=comparison_msg.model_dump(),
            channel=settings.comparison_queue,
        )

        # Calculate pairs count
        n = len(texts)
        pairs_count = n * (n - 1) // 2  # Combinations without repetition

        return ComparisonResponse(
            task_id=task_id,
            collection_id=collection_id,
            collection_name=request.collection_name,
            text_count=len(texts),
            message=f"Comparison task created. Will compare {pairs_count} text pairs."
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )
