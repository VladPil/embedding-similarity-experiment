"""
API endpoints for cache management.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any

from server.db.database import get_db
from server.repositories.embedding_repository import EmbeddingRepository
from loguru import logger

router = APIRouter()


@router.get("/stats")
async def get_cache_stats(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get cache statistics.

    Returns:
        Dictionary with cache statistics
    """
    try:
        repo = EmbeddingRepository(db)
        stats = await repo.get_cache_stats()

        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_cache(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Clear all cache (Redis + PostgreSQL).

    Returns:
        Number of deleted entries
    """
    try:
        repo = EmbeddingRepository(db)
        count = await repo.clear_all_cache()

        return {
            "success": True,
            "message": f"Cleared {count} cache entries",
            "count": count
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/text/{text_id}")
async def delete_text_cache(
    text_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Delete cache for specific text.

    Args:
        text_id: Text ID

    Returns:
        Number of deleted entries
    """
    try:
        repo = EmbeddingRepository(db)
        count = await repo.delete_text_embeddings(text_id)

        return {
            "success": True,
            "message": f"Deleted {count} cache entries for text {text_id}",
            "count": count
        }
    except Exception as e:
        logger.error(f"Failed to delete text cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/model/{model_name}")
async def delete_model_cache(
    model_name: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Delete cache for specific model.

    Args:
        model_name: Model name

    Returns:
        Number of deleted entries
    """
    try:
        repo = EmbeddingRepository(db)
        count = await repo.delete_model_embeddings(model_name)

        return {
            "success": True,
            "message": f"Deleted {count} cache entries for model {model_name}",
            "count": count
        }
    except Exception as e:
        logger.error(f"Failed to delete model cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))
