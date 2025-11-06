"""
API endpoints for analyzing 3 or more texts.
"""

from typing import List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from server.db.database import get_db
from server.services.embedding_service import EmbeddingService
from server.services.text_service import TextService
from server.services.strategies.multi_text import MultiTextAnalysisStrategy
from server.schemas.models import AnalysisResponse


router = APIRouter()


def get_text_service(db: Session = Depends(get_db)) -> TextService:
    """Get text service instance."""
    return TextService(db)


def get_embedding_service(db: Session = Depends(get_db)) -> EmbeddingService:
    """Get embedding service instance."""
    return EmbeddingService(db)


class MultiTextAnalysisRequest(BaseModel):
    """Request model for multi-text analysis."""

    text_ids: List[str] = Field(
        ...,
        min_length=3,
        description="List of text IDs to analyze (minimum 3)"
    )
    model: str = Field(
        default="multilingual-e5-base",
        description="Embedding model to use"
    )


class MultiTextAnalysisResponse(BaseModel):
    """Response model for multi-text analysis."""

    n_texts: int
    model_used: str
    similarity_matrix: List[List[float]]
    text_info: List[dict]
    pairwise_similarities: List[dict]
    most_similar_pair: dict
    most_different_pair: dict
    statistics: dict
    clusters: List[dict]
    interpretation: str


@router.post(
    "/analyze-multiple",
    response_model=MultiTextAnalysisResponse,
    summary="Analyze 3 or more texts",
    description="Perform pairwise similarity analysis on 3 or more texts and find clusters"
)
async def analyze_multiple_texts(
    request: MultiTextAnalysisRequest,
    text_service: TextService = Depends(get_text_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    Analyze multiple texts (3 or more).

    Args:
        request: Multi-text analysis request
        text_service: Text service instance
        embedding_service: Embedding service instance

    Returns:
        Multi-text analysis response with similarity matrix and clusters
    """
    # Validate number of texts
    if len(request.text_ids) < 3:
        raise HTTPException(
            status_code=400,
            detail="Need at least 3 texts for multi-text analysis"
        )

    # Get all texts
    texts = []
    text_contents = []
    text_titles = []

    for text_id in request.text_ids:
        text = text_service.get_text(text_id)
        if not text:
            raise HTTPException(
                status_code=404,
                detail=f"Text with id {text_id} not found"
            )

        texts.append(text)
        text_contents.append(text_service.get_text_content(text_id))
        text_titles.append(text.title)

    # Create strategy and analyze
    strategy = MultiTextAnalysisStrategy(embedding_service)

    try:
        result = await strategy.analyze_multiple(
            text_contents=text_contents,
            text_ids=request.text_ids,
            text_titles=text_titles,
            params={"model": request.model}
        )

        return MultiTextAnalysisResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


class TextComparisonMatrix(BaseModel):
    """Quick comparison matrix for multiple texts."""

    text_id: int
    title: str


@router.post(
    "/comparison-matrix",
    summary="Get quick comparison matrix",
    description="Get a simple similarity matrix for quick comparison"
)
async def get_comparison_matrix(
    text_ids: List[str],
    model: str = "multilingual-e5-base",
    text_service: TextService = Depends(get_text_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    Get a quick comparison matrix for multiple texts.

    Args:
        text_ids: List of text IDs
        model: Embedding model to use
        text_service: Text service instance
        embedding_service: Embedding service instance

    Returns:
        Similarity matrix and text info
    """
    if len(text_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 texts for comparison"
        )

    # Get embeddings
    embeddings = []
    text_info = []

    for text_id in text_ids:
        text = text_service.get_text(text_id)
        if not text:
            raise HTTPException(
                status_code=404,
                detail=f"Text with id {text_id} not found"
            )

        emb = await embedding_service.get_embedding(text_id, model)
        if emb is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embedding for text {text_id}"
            )

        embeddings.append(emb)
        text_info.append({
            "id": text_id,
            "title": text.title
        })

    # Calculate similarity matrix
    from server.core.similarity_calc import SimilarityCalculator
    import numpy as np

    calc = SimilarityCalculator()
    n = len(embeddings)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                sim = calc.cosine_similarity(embeddings[i], embeddings[j])
                matrix[i, j] = sim
                matrix[j, i] = sim

    return {
        "text_info": text_info,
        "similarity_matrix": matrix.tolist(),
        "model_used": model
    }


@router.post(
    "/find-similar-to-all",
    summary="Find texts similar to all given texts",
    description="Find which texts are similar to all provided texts"
)
async def find_similar_to_all(
    text_ids: List[str],
    threshold: float = 0.7,
    model: str = "multilingual-e5-base",
    text_service: TextService = Depends(get_text_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    Find texts that are similar to all provided texts.

    Useful for finding texts that share common themes/style with multiple examples.

    Args:
        text_ids: List of text IDs to compare against
        threshold: Minimum similarity threshold
        model: Embedding model to use
        text_service: Text service instance
        embedding_service: Embedding service instance

    Returns:
        List of texts similar to all provided texts
    """
    if len(text_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 texts"
        )

    # Get all texts from database with proper pagination
    offset = 0
    batch_size = 100
    all_texts = []

    while True:
        batch = text_service.list_texts(limit=batch_size, offset=offset)
        if not batch:
            break
        all_texts.extend(batch)
        offset += batch_size
        # Safety limit to prevent infinite loops
        if offset >= 10000:
            break

    # Get embeddings for query texts
    query_embeddings = []
    for text_id in text_ids:
        emb = await embedding_service.get_embedding(text_id, model)
        if emb is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embedding for text {text_id}"
            )
        query_embeddings.append(emb)

    # Find texts similar to all query texts
    from server.core.similarity_calc import SimilarityCalculator
    calc = SimilarityCalculator()

    similar_texts = []

    for text in all_texts:
        # Skip if text is in query list
        if text.id in text_ids:
            continue

        # Get embedding for this text
        emb = await embedding_service.get_embedding(text.id, model)
        if emb is None:
            continue

        # Check similarity to all query texts
        similarities = [
            calc.cosine_similarity(emb, query_emb)
            for query_emb in query_embeddings
        ]

        # Text must be similar to ALL query texts
        min_similarity = min(similarities)
        avg_similarity = sum(similarities) / len(similarities)

        if min_similarity >= threshold:
            similar_texts.append({
                "id": text.id,
                "title": text.title,
                "min_similarity": float(min_similarity),
                "avg_similarity": float(avg_similarity),
                "similarities": [float(s) for s in similarities]
            })

    # Sort by average similarity
    similar_texts.sort(key=lambda x: x['avg_similarity'], reverse=True)

    return {
        "query_text_ids": text_ids,
        "threshold": threshold,
        "found_texts": similar_texts,
        "count": len(similar_texts)
    }
