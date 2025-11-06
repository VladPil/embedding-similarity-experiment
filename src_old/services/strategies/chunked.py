"""
Chunked analysis strategy.
"""

from typing import Dict, Any
import numpy as np

from server.services.strategies.base import AnalysisStrategy
from server.services.embedding_service import EmbeddingService
from server.core.similarity_calc import SimilarityCalculator
from server.core.chunks import ChunkManager


class ChunkedAnalysisStrategy(AnalysisStrategy):
    """Strategy for chunked analysis."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.similarity_calc = SimilarityCalculator()

    async def analyze(
        self,
        text1_content: str,
        text2_content: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform chunked analysis.

        Args:
            text1_content: First text content
            text2_content: Second text content
            params: Analysis parameters including chunk_size, overlap, split_by, top_n, model, adaptive

        Returns:
            Dictionary with chunks counts, similarity_matrix, overall_similarity, top_matches
        """
        # Get parameters
        chunk_size = params.get("chunk_size", 1000)
        overlap = params.get("overlap", 100)
        split_by = params.get("split_by", "sentences")
        top_n = params.get("top_n", 10)
        model_key = params.get("model", "multilingual-e5-base")

        # Create chunk manager
        chunk_mgr = ChunkManager(chunk_size, overlap, split_by)

        # Check for adaptive chunking
        if params.get("adaptive", False):
            chunks1 = chunk_mgr.adaptive_chunk_text(text1_content)
            chunks2 = chunk_mgr.adaptive_chunk_text(text2_content)
        else:
            chunks1 = chunk_mgr.chunk_text(text1_content)
            chunks2 = chunk_mgr.chunk_text(text2_content)

        # Generate embeddings for chunks
        model = self.embedding_service.get_model(model_key)

        emb1_list = [model._encode_text(chunk["text"]) for chunk in chunks1]
        emb2_list = [model._encode_text(chunk["text"]) for chunk in chunks2]

        # Calculate similarity matrix
        matrix = np.zeros((len(chunks1), len(chunks2)))
        for i, emb1 in enumerate(emb1_list):
            for j, emb2 in enumerate(emb2_list):
                matrix[i, j] = self.similarity_calc.cosine_similarity(emb1, emb2)

        # Find best matches
        comparisons = chunk_mgr.compare_chunks(
            chunks1, chunks2,
            lambda t1, t2: self.similarity_calc.cosine_similarity(
                model._encode_text(t1),
                model._encode_text(t2)
            )
        )
        top_matches = chunk_mgr.get_most_similar_chunks(comparisons, top_n)

        return {
            "chunks1_count": len(chunks1),
            "chunks2_count": len(chunks2),
            "similarity_matrix": matrix.tolist(),
            "overall_similarity": float(np.mean(matrix)),
            "top_matches": top_matches,
            "interpretation": f"Средняя схожесть чанков: {np.mean(matrix):.1%}"
        }

    def get_type(self) -> str:
        """Get analysis type identifier."""
        return "chunked"
