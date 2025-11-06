"""
Analysis API endpoints - refactored version using service layer.
Handles all types of text similarity analysis.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from loguru import logger

from server.db.database import database
from server.services.analysis_service import AnalysisService
from server.core.tasks import get_task_manager
from server.schemas.models import (
    SemanticAnalysisRequest,
    StyleAnalysisRequest,
    TFIDFAnalysisRequest,
    EmotionAnalysisRequest,
    LLMAnalysisRequest,
    ChunkedAnalysisRequest,
    CombinedAnalysisRequest,
    AnalysisResponse,
    TaskStartResponse,
    TextInfo
)

router = APIRouter(prefix="/analysis", tags=["analysis"])


# Dependency injection
def get_db() -> Session:
    """Get database session."""
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_analysis_service(db: Session = Depends(get_db)) -> AnalysisService:
    """Get analysis service instance."""
    return AnalysisService(db)


async def _prepare_text_info(service: AnalysisService, text_id: str) -> TextInfo:
    """Helper to prepare text info for response."""
    text_service = service.text_service
    text = text_service.get_text(text_id)

    if not text:
        raise HTTPException(status_code=404, detail=f"Text {text_id} not found")

    return TextInfo(
        id=text.id,
        title=text.title,
        lines=text.lines,
        length=text.length
    )


@router.post("/semantic", response_model=AnalysisResponse)
async def analyze_semantic(
    request: SemanticAnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service)
):
    """
    Perform semantic similarity analysis using embeddings.

    Args:
        request: Semantic analysis request

    Returns:
        Analysis results with similarity score
    """
    try:
        # Initialize service
        await service.initialize()

        # Get text info
        text1_info = await _prepare_text_info(service, request.text_id1)
        text2_info = await _prepare_text_info(service, request.text_id2)

        # Perform analysis
        result = await service.analyze(
            analysis_type="semantic",
            text1_id=request.text_id1,
            text2_id=request.text_id2,
            params={"model": request.model}
        )

        logger.info(f"Semantic similarity: {result['similarity']:.4f}")

        return AnalysisResponse(
            text1=text1_info,
            text2=text2_info,
            similarity=result["similarity"],
            interpretation=result["interpretation"],
            details=result
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Semantic analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await service.cleanup()


@router.post("/style", response_model=AnalysisResponse)
async def analyze_style(
    request: StyleAnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service)
):
    """
    Perform style analysis.

    Args:
        request: Style analysis request

    Returns:
        Analysis results with similarity score
    """
    try:
        # Initialize service
        await service.initialize()

        # Get text info
        text1_info = await _prepare_text_info(service, request.text_id1)
        text2_info = await _prepare_text_info(service, request.text_id2)

        # Perform analysis
        result = await service.analyze(
            analysis_type="style",
            text1_id=request.text_id1,
            text2_id=request.text_id2
        )

        return AnalysisResponse(
            text1=text1_info,
            text2=text2_info,
            similarity=result["similarity"],
            interpretation=result["interpretation"],
            details=result
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Style analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await service.cleanup()


@router.post("/tfidf", response_model=AnalysisResponse)
async def analyze_tfidf(
    request: TFIDFAnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service)
):
    """
    Perform TF-IDF analysis.

    Args:
        request: TF-IDF analysis request

    Returns:
        Analysis results with similarity score
    """
    try:
        # Initialize service
        await service.initialize()

        # Get text info
        text1_info = await _prepare_text_info(service, request.text_id1)
        text2_info = await _prepare_text_info(service, request.text_id2)

        # Perform analysis
        result = await service.analyze(
            analysis_type="tfidf",
            text1_id=request.text_id1,
            text2_id=request.text_id2
        )

        return AnalysisResponse(
            text1=text1_info,
            text2=text2_info,
            similarity=result["similarity"],
            interpretation=result["interpretation"],
            details=result
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TF-IDF analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await service.cleanup()


@router.post("/emotion", response_model=AnalysisResponse)
async def analyze_emotion(
    request: EmotionAnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service)
):
    """
    Perform emotional trajectory analysis.

    Args:
        request: Emotion analysis request

    Returns:
        Analysis results with similarity score
    """
    try:
        # Initialize service
        await service.initialize()

        # Get text info
        text1_info = await _prepare_text_info(service, request.text_id1)
        text2_info = await _prepare_text_info(service, request.text_id2)

        # Perform analysis
        result = await service.analyze(
            analysis_type="emotion",
            text1_id=request.text_id1,
            text2_id=request.text_id2,
            params={"num_segments": request.num_segments}
        )

        return AnalysisResponse(
            text1=text1_info,
            text2=text2_info,
            similarity=result["similarity"],
            interpretation=result["interpretation"],
            details=result
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Emotion analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await service.cleanup()


@router.post("/llm", response_model=TaskStartResponse)
async def analyze_llm(
    request: LLMAnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service)
):
    """
    Perform LLM-based analysis (as a background task).

    Args:
        request: LLM analysis request

    Returns:
        Task ID for tracking progress
    """
    try:
        # Initialize service
        await service.initialize()

        # Validate texts exist
        text1 = service.text_service.get_text(request.text_id1)
        text2 = service.text_service.get_text(request.text_id2)

        if not text1:
            raise HTTPException(status_code=404, detail=f"Text {request.text_id1} not found")
        if not text2:
            raise HTTPException(status_code=404, detail=f"Text {request.text_id2} not found")

        # Get task manager
        task_mgr = get_task_manager()

        # Define the analysis function
        async def run_llm_analysis(**kwargs):
            """Execute LLM analysis."""
            result = await service.analyze(
                analysis_type="llm",
                text1_id=request.text_id1,
                text2_id=request.text_id2,
                params={
                    "model": request.model,
                    "task_type": "compare",
                    "text1_title": text1.title,
                    "text2_title": text2.title
                }
            )
            return result

        # Submit task to task manager
        task_id = await task_mgr.submit_task(
            run_llm_analysis,
            name=f"LLM Analysis ({request.model})",
            metadata={
                "type": "llm",
                "text1_id": request.text_id1,
                "text2_id": request.text_id2,
                "text1_title": text1.title,
                "text2_title": text2.title,
                "model": request.model
            }
        )

        logger.info(f"LLM analysis task created: {task_id}")

        return TaskStartResponse(
            task_id=task_id,
            message="LLM analysis started in background"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start LLM analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await service.cleanup()


@router.post("/chunked", response_model=TaskStartResponse)
async def analyze_chunked(
    request: ChunkedAnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service)
):
    """
    Perform chunked analysis (as a background task).

    Args:
        request: Chunked analysis request

    Returns:
        Task ID for tracking progress
    """
    try:
        # Initialize service
        await service.initialize()

        # Validate texts exist
        text1 = service.text_service.get_text(request.text_id1)
        text2 = service.text_service.get_text(request.text_id2)

        if not text1:
            raise HTTPException(status_code=404, detail=f"Text {request.text_id1} not found")
        if not text2:
            raise HTTPException(status_code=404, detail=f"Text {request.text_id2} not found")

        # Get task manager
        task_mgr = get_task_manager()

        # Define the analysis function
        async def run_chunked_analysis(**kwargs):
            """Execute chunked analysis."""
            result = await service.analyze(
                analysis_type="chunked",
                text1_id=request.text_id1,
                text2_id=request.text_id2,
                params={
                    "model": request.model,
                    "chunk_size": request.chunk_size,
                    "overlap": request.overlap,
                    "split_by": request.split_by,
                    "top_n": request.top_n,
                    "adaptive": getattr(request, "adaptive", False)
                }
            )
            return result

        # Submit task to task manager
        task_id = await task_mgr.submit_task(
            run_chunked_analysis,
            name="Chunked Analysis",
            metadata={
                "text1_id": request.text_id1,
                "text2_id": request.text_id2,
                "text1_title": text1.title,
                "text2_title": text2.title,
                "model": request.model,
                "chunk_size": request.chunk_size
            }
        )

        logger.info(f"Chunked analysis task created: {task_id}")

        return TaskStartResponse(
            task_id=task_id,
            message="Chunked analysis started in background"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start chunked analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await service.cleanup()


@router.post("/combined", response_model=TaskStartResponse)
async def analyze_combined(
    request: CombinedAnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service)
):
    """
    Perform combined analysis using multiple methods (as a background task).

    Args:
        request: Combined analysis request
        service: Analysis service

    Returns:
        Task ID for tracking progress
    """
    try:
        # Initialize service
        await service.initialize()

        # Validate texts exist
        text1 = service.text_service.get_text(request.text_id1)
        text2 = service.text_service.get_text(request.text_id2)

        if not text1:
            raise HTTPException(status_code=404, detail=f"Text {request.text_id1} not found")
        if not text2:
            raise HTTPException(status_code=404, detail=f"Text {request.text_id2} not found")

        # Get task manager
        task_mgr = get_task_manager()

        # Define the analysis function
        async def run_combined_analysis(**kwargs):
            """Execute combined analysis."""
            result = await service.analyze(
                analysis_type="combined",
                text1_id=request.text_id1,
                text2_id=request.text_id2,
                params={
                    "model": request.model,
                    "weights": request.weights or {
                        "semantic": 0.3,
                        "style": 0.25,
                        "tfidf": 0.25,
                        "emotion": 0.2
                    }
                }
            )
            return result

        # Submit task to task manager
        task_id = await task_mgr.submit_task(
            run_combined_analysis,
            name="Combined Analysis",
            metadata={
                "text1_id": request.text_id1,
                "text2_id": request.text_id2,
                "text1_title": text1.title,
                "text2_title": text2.title,
                "model": request.model
            }
        )

        logger.info(f"Combined analysis task created: {task_id}")

        return TaskStartResponse(
            task_id=task_id,
            message="Combined analysis started in background"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start combined analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await service.cleanup()


@router.get("/history")
async def get_analysis_history(
    limit: int = 10,
    analysis_type: Optional[str] = None,
    service: AnalysisService = Depends(get_analysis_service)
):
    """
    Get analysis history.

    Args:
        limit: Maximum number of results
        analysis_type: Filter by analysis type

    Returns:
        List of recent analyses
    """
    try:
        await service.initialize()

        history = await service.get_analysis_history(
            limit=limit,
            analysis_type=analysis_type
        )

        return {"success": True, "history": history}

    except Exception as e:
        logger.error(f"Failed to get analysis history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await service.cleanup()