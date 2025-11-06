"""
Hybrid Similarity Strategy.
Combines multiple similarity methods for robust comparison.
Implements Composite pattern.
"""

import numpy as np
from typing import Dict, Any, List
from loguru import logger

from server.core.similarity.base import (
    ISimilarityStrategy,
    SimilarityContext,
    SimilarityResult,
    SimilarityMethod,
    SimilarityScope
)
from server.core.similarity.strategies.cosine_strategy import CosineSimilarityStrategy
from server.core.similarity.strategies.semantic_strategy import SemanticSimilarityStrategy


class HybridSimilarityStrategy(ISimilarityStrategy):
    """
    Hybrid similarity strategy combining multiple methods.

    Features:
    - Combines cosine and semantic similarity
    - Adaptive weighting based on text characteristics
    - Higher confidence through consensus
    - Detailed breakdown of each component
    """

    def __init__(
        self,
        scope: SimilarityScope = SimilarityScope.FULL_TEXT,
        weights: Dict[str, float] = None
    ):
        """
        Initialize hybrid similarity strategy.

        Args:
            scope: Similarity scope
            weights: Optional custom weights for each method
        """
        self.scope = scope

        # Default weights
        self.weights = weights or {
            "cosine": 0.4,
            "semantic": 0.4,
            "structural": 0.2
        }

        # Sub-strategies
        self.cosine_strategy = CosineSimilarityStrategy(scope=scope)
        self.semantic_strategy = SemanticSimilarityStrategy(scope=SimilarityScope.THEMATIC)

    def get_method(self) -> SimilarityMethod:
        """Get similarity method identifier."""
        return SimilarityMethod.HYBRID

    def get_scope(self) -> SimilarityScope:
        """Get similarity scope."""
        return self.scope

    def requires_embeddings(self) -> bool:
        """Hybrid similarity requires embeddings."""
        return True

    def get_estimated_time(self) -> float:
        """Estimate 7 seconds for hybrid similarity."""
        return 7.0

    async def calculate(self, context: SimilarityContext) -> SimilarityResult:
        """
        Calculate hybrid similarity using multiple methods.

        Args:
            context: Similarity context with embeddings

        Returns:
            Similarity result with combined score
        """
        try:
            logger.info("Calculating hybrid similarity...")

            if not context.embeddings1 or not context.embeddings2:
                raise ValueError("Embeddings required for hybrid similarity")

            # 1. Cosine similarity
            cosine_result = await self.cosine_strategy.calculate(context)
            cosine_score = cosine_result.score

            # 2. Semantic similarity
            semantic_result = await self.semantic_strategy.calculate(context)
            semantic_score = semantic_result.score

            # 3. Structural similarity (length, chunk count)
            structural_score, structural_details = self._calculate_structural(context)

            # Adaptive weighting based on agreement
            weights = self._calculate_adaptive_weights(
                cosine_score,
                semantic_score,
                structural_score
            )

            # Combined score
            final_score = (
                weights["cosine"] * cosine_score +
                weights["semantic"] * semantic_score +
                weights["structural"] * structural_score
            )

            # Calculate confidence based on agreement
            scores = [cosine_score, semantic_score, structural_score]
            confidence = self._calculate_confidence(scores)

            # Build detailed response
            details = {
                "method": "hybrid",
                "components": {
                    "cosine": {
                        "score": float(cosine_score),
                        "weight": weights["cosine"],
                        "details": cosine_result.details
                    },
                    "semantic": {
                        "score": float(semantic_score),
                        "weight": weights["semantic"],
                        "details": semantic_result.details
                    },
                    "structural": {
                        "score": float(structural_score),
                        "weight": weights["structural"],
                        "details": structural_details
                    }
                },
                "final_score": float(final_score),
                "confidence": float(confidence),
                "score_variance": float(np.var(scores)),
                "weights_used": weights
            }

            interpretation = self._build_interpretation(
                final_score,
                cosine_score,
                semantic_score,
                structural_score,
                confidence
            )

            logger.info(f"Hybrid similarity calculated: {final_score:.3f} (confidence: {confidence:.2f})")

            return SimilarityResult(
                method=self.get_method(),
                scope=self.scope,
                score=final_score,
                confidence=confidence,
                details=details,
                interpretation=interpretation
            )

        except Exception as e:
            logger.error(f"Hybrid similarity calculation failed: {e}")
            return SimilarityResult(
                method=self.get_method(),
                scope=self.scope,
                score=0.0,
                confidence=0.0,
                details={"error": str(e)},
                interpretation="Ошибка вычисления гибридной схожести"
            )

    def _calculate_structural(
        self,
        context: SimilarityContext
    ) -> tuple[float, Dict[str, Any]]:
        """
        Calculate structural similarity based on text characteristics.

        Compares:
        - Length similarity
        - Chunk count similarity
        - Embedding distribution similarity
        """
        # Length similarity
        len1 = len(context.text1)
        len2 = len(context.text2)
        length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0

        # Chunk count similarity
        chunk_count1 = len(context.embeddings1)
        chunk_count2 = len(context.embeddings2)
        chunk_ratio = min(chunk_count1, chunk_count2) / max(chunk_count1, chunk_count2)

        # Embedding variance similarity (writing style consistency)
        var1 = np.var([np.linalg.norm(e) for e in context.embeddings1])
        var2 = np.var([np.linalg.norm(e) for e in context.embeddings2])
        var_ratio = min(var1, var2) / max(var1, var2) if max(var1, var2) > 0 else 0.0

        # Combined structural score
        structural_score = (
            0.3 * length_ratio +
            0.4 * chunk_ratio +
            0.3 * var_ratio
        )

        details = {
            "length_text1": len1,
            "length_text2": len2,
            "length_ratio": float(length_ratio),
            "chunks_text1": chunk_count1,
            "chunks_text2": chunk_count2,
            "chunk_ratio": float(chunk_ratio),
            "variance_ratio": float(var_ratio)
        }

        return structural_score, details

    def _calculate_adaptive_weights(
        self,
        cosine_score: float,
        semantic_score: float,
        structural_score: float
    ) -> Dict[str, float]:
        """
        Calculate adaptive weights based on score agreement.

        If scores agree (low variance), trust all equally.
        If scores disagree, favor the more reliable method.
        """
        scores = [cosine_score, semantic_score, structural_score]
        variance = np.var(scores)

        if variance < 0.05:
            # High agreement - use default weights
            return self.weights.copy()
        else:
            # Low agreement - favor semantic (more robust)
            return {
                "cosine": 0.3,
                "semantic": 0.5,
                "structural": 0.2
            }

    def _calculate_confidence(self, scores: List[float]) -> float:
        """
        Calculate confidence based on score agreement.

        High agreement = high confidence
        Low agreement = low confidence
        """
        variance = np.var(scores)

        # Inverse relationship with variance
        # variance 0 -> confidence 1.0
        # variance 0.5 -> confidence 0.5
        confidence = 1.0 - min(variance * 2, 1.0)

        return max(confidence, 0.5)  # Minimum confidence 0.5

    def _build_interpretation(
        self,
        final_score: float,
        cosine_score: float,
        semantic_score: float,
        structural_score: float,
        confidence: float
    ) -> str:
        """Build detailed interpretation of hybrid similarity."""
        lines = []

        # Overall score
        if final_score > 0.85:
            lines.append("🎯 **Очень высокая общая схожесть**")
        elif final_score > 0.7:
            lines.append("✅ **Высокая общая схожесть**")
        elif final_score > 0.5:
            lines.append("📊 **Умеренная общая схожесть**")
        else:
            lines.append("📉 **Низкая общая схожесть**")

        lines.append("")

        # Component breakdown
        lines.append("**Детализация по компонентам:**")
        lines.append(f"  • Векторная схожесть (cosine): {cosine_score:.2%}")
        lines.append(f"  • Семантическая схожесть: {semantic_score:.2%}")
        lines.append(f"  • Структурная схожесть: {structural_score:.2%}")
        lines.append("")

        # Confidence interpretation
        conf_emoji = "🎯" if confidence > 0.8 else "✅" if confidence > 0.6 else "⚠️"
        lines.append(f"{conf_emoji} **Уверенность в результате:** {confidence:.0%}")

        if confidence < 0.7:
            lines.append("  ℹ️ Различные методы дают разные оценки - результат может быть неоднозначным")

        return "\n".join(lines)

    def interpret_score(self, score: float) -> str:
        """
        Interpret hybrid similarity score.

        Args:
            score: Similarity score (0.0 to 1.0)

        Returns:
            Human-readable interpretation
        """
        return self.interpret_score(score)  # Use _build_interpretation in calculate
