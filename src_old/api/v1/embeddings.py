"""
Embeddings API endpoints.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from server.schemas.models import ModelsListResponse, EmbeddingModelInfo
from server.core.embeddings import EmbeddingManager

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    """
    List all available embedding models.

    Returns:
        List of available models with metadata
    """
    try:
        models = EmbeddingManager.list_models()

        # Convert to EmbeddingModelInfo objects
        model_infos = {}
        for key, info in models.items():
            model_infos[key] = EmbeddingModelInfo(
                key=key,
                name=info['name'],
                dimensions=info['dimensions'],
                description=info['description']
            )

        return ModelsListResponse(models=model_infos)

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compute")
async def compute_embedding(text: str, model: str = "multilingual-e5-small"):
    """
    Compute embedding for text (mainly for testing).

    Args:
        text: Text to encode
        model: Model key

    Returns:
        Embedding vector and metadata
    """
    try:
        # Initialize manager
        emb_mgr = EmbeddingManager(model_key=model)

        # Get embedding
        embedding = await emb_mgr.get_embedding(text, use_cache=True)

        return {
            "success": True,
            "model": model,
            "dimensions": len(embedding),
            "embedding": embedding.tolist(),
            "text_length": len(text)
        }

    except Exception as e:
        logger.error(f"Failed to compute embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))
