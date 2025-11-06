"""
Cosine Similarity Strategy.
Classic vector-based similarity using cosine distance.
"""

import numpy as np
from typing import Dict, Any
from loguru import logger

from server.core.similarity.base import (
    ISimilarityStrategy,
    SimilarityContext,
    SimilarityResult,
    SimilarityMethod,
    SimilarityScope
)


class CosineSimilarityStrategy(ISimilarityStrategy):
    """
    Cosine similarity strategy.

    - Fast and efficient
    - Works with embeddings
    - Good for semantic similarity
    - Range: -1 to 1 (normalized to 0-1)
    """

    def __init__(self, scope: SimilarityScope = SimilarityScope.FULL_TEXT):
        """
        Initialize cosine similarity strategy.

        Args:
            scope: Similarity scope (full_text or chunk)
        """
        self.scope = scope

    def get_method(self) -> SimilarityMethod:
        """Get similarity method identifier."""
        return SimilarityMethod.COSINE

    def get_scope(self) -> SimilarityScope:
        """Get similarity scope."""
        return self.scope

    def requires_embeddings(self) -> bool:
        """Cosine similarity requires embeddings."""
        return True

    def get_estimated_time(self) -> float:
        """Estimate 1 second for cosine similarity."""
        return 1.0

    async def calculate(self, context: SimilarityContext) -> SimilarityResult:
        """
        Calculate cosine similarity.

        Args:
            context: Similarity context with embeddings

        Returns:
            Similarity result
        """
        try:
            logger.info(f"Calculating cosine similarity (scope: {self.scope.value})")

            if not context.embeddings1 or not context.embeddings2:
                raise ValueError("Embeddings required for cosine similarity")

            if self.scope == SimilarityScope.FULL_TEXT:
                score, details = self._calculate_full_text(
                    context.embeddings1,
                    context.embeddings2
                )
            elif self.scope == SimilarityScope.CHUNK:
                score, details = self._calculate_chunk_level(
                    context.embeddings1,
                    context.embeddings2,
                    context.chunks1,
                    context.chunks2
                )
            else:
                raise ValueError(f"Unsupported scope: {self.scope}")

            interpretation = self.interpret_score(score)

            logger.info(f"Cosine similarity calculated: {score:.3f}")

            return SimilarityResult(
                method=self.get_method(),
                scope=self.scope,
                score=score,
                confidence=0.9,  # High confidence for cosine
                details=details,
                interpretation=interpretation
            )

        except Exception as e:
            logger.error(f"Cosine similarity calculation failed: {e}")
            return SimilarityResult(
                method=self.get_method(),
                scope=self.scope,
                score=0.0,
                confidence=0.0,
                details={"error": str(e)},
                interpretation="Ошибка вычисления схожести"
            )

    def _calculate_full_text(
        self,
        embeddings1: list,
        embeddings2: list
    ) -> tuple[float, Dict[str, Any]]:
        """Calculate full text similarity using average embeddings."""
        # Average embeddings for each text
        avg_emb1 = np.mean(embeddings1, axis=0)
        avg_emb2 = np.mean(embeddings2, axis=0)

        # Cosine similarity
        similarity = self._cosine_similarity(avg_emb1, avg_emb2)

        # Normalize to 0-1 range
        score = (similarity + 1) / 2

        details = {
            "method": "average_embeddings",
            "chunks_text1": len(embeddings1),
            "chunks_text2": len(embeddings2),
            "raw_similarity": float(similarity)
        }

        return score, details

    def _calculate_chunk_level(
        self,
        embeddings1: list,
        embeddings2: list,
        chunks1: list,
        chunks2: list
    ) -> tuple[float, Dict[str, Any]]:
        """
        Calculate chunk-level similarity using alignment.

        Uses dynamic programming to find optimal chunk alignment.
        """
        # Create similarity matrix
        n1, n2 = len(embeddings1), len(embeddings2)
        similarity_matrix = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                sim = self._cosine_similarity(embeddings1[i], embeddings2[j])
                similarity_matrix[i, j] = (sim + 1) / 2  # Normalize

        # Find best alignment using maximum similarity
        # Simple approach: average of maximum similarities
        max_sim_1 = np.max(similarity_matrix, axis=1).mean()
        max_sim_2 = np.max(similarity_matrix, axis=0).mean()

        # Overall score is average of both directions
        score = (max_sim_1 + max_sim_2) / 2

        # Find highly similar chunk pairs
        threshold = 0.7
        similar_pairs = []
        for i in range(n1):
            for j in range(n2):
                if similarity_matrix[i, j] > threshold:
                    similar_pairs.append({
                        "chunk1_index": i,
                        "chunk2_index": j,
                        "similarity": float(similarity_matrix[i, j])
                    })

        # Sort by similarity
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)

        details = {
            "method": "chunk_alignment",
            "chunks_text1": n1,
            "chunks_text2": n2,
            "similarity_matrix_shape": list(similarity_matrix.shape),
            "max_similarity_text1_to_text2": float(max_sim_1),
            "max_similarity_text2_to_text1": float(max_sim_2),
            "highly_similar_pairs": similar_pairs[:10]  # Top 10
        }

        return score, details

    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def interpret_score(self, score: float) -> str:
        """
        Interpret cosine similarity score.

        Args:
            score: Similarity score (0.0 to 1.0)

        Returns:
            Human-readable interpretation
        """
        if score > 0.9:
            return "Очень высокая схожесть - тексты почти идентичны"
        elif score > 0.7:
            return "Высокая схожесть - тексты значительно похожи"
        elif score > 0.5:
            return "Умеренная схожесть - тексты частично похожи"
        elif score > 0.3:
            return "Низкая схожесть - тексты имеют мало общего"
        else:
            return "Очень низкая схожесть - тексты практически различны"
