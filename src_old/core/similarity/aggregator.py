"""
Similarity Result Aggregator.
Combines multiple similarity scores using weighted average or consensus.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from loguru import logger

from server.core.similarity.base import (
    ISimilarityAggregator,
    SimilarityResult,
    SimilarityMethod
)


class WeightedAggregator(ISimilarityAggregator):
    """
    Weighted average aggregator.

    Combines similarity scores using weighted average with optional
    confidence-based adjustments.
    """

    def __init__(self, confidence_weighting: bool = True):
        """
        Initialize weighted aggregator.

        Args:
            confidence_weighting: Whether to adjust weights by confidence
        """
        self.confidence_weighting = confidence_weighting

    def aggregate(
        self,
        results: List[SimilarityResult],
        weights: Optional[Dict[SimilarityMethod, float]] = None
    ) -> Tuple[float, str]:
        """
        Aggregate multiple similarity results.

        Args:
            results: List of similarity results
            weights: Optional custom weights for each method

        Returns:
            Tuple of (final_score, interpretation)
        """
        if not results:
            return 0.0, "Нет результатов для агрегации"

        try:
            # Default equal weights
            if weights is None:
                weights = {result.method: 1.0 for result in results}

            # Normalize weights
            total_weight = sum(weights.get(r.method, 1.0) for r in results)

            if total_weight == 0:
                return 0.0, "Некорректные веса"

            # Calculate weighted average
            weighted_sum = 0.0
            weight_sum = 0.0

            for result in results:
                method_weight = weights.get(result.method, 1.0) / total_weight

                # Adjust by confidence if enabled
                if self.confidence_weighting:
                    method_weight *= result.confidence

                weighted_sum += result.score * method_weight
                weight_sum += method_weight

            final_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0

            # Build interpretation
            interpretation = self._build_interpretation(results, final_score)

            logger.info(f"Aggregated {len(results)} similarity results: {final_score:.3f}")

            return final_score, interpretation

        except Exception as e:
            logger.error(f"Similarity aggregation failed: {e}")
            return 0.0, f"Ошибка агрегации: {str(e)}"

    def _build_interpretation(
        self,
        results: List[SimilarityResult],
        final_score: float
    ) -> str:
        """Build interpretation of aggregated results."""
        lines = []

        # Overall score interpretation
        if final_score > 0.85:
            lines.append("🎯 **Очень высокая схожесть** (агрегированная оценка)")
        elif final_score > 0.7:
            lines.append("✅ **Высокая схожесть** (агрегированная оценка)")
        elif final_score > 0.5:
            lines.append("📊 **Умеренная схожесть** (агрегированная оценка)")
        else:
            lines.append("📉 **Низкая схожесть** (агрегированная оценка)")

        lines.append("")
        lines.append(f"**Итоговый балл:** {final_score:.2%}")
        lines.append("")

        # Individual method scores
        lines.append("**Оценки по методам:**")
        for result in results:
            method_name = self._method_name(result.method)
            lines.append(
                f"  • {method_name}: {result.score:.2%} "
                f"(уверенность: {result.confidence:.0%})"
            )

        lines.append("")

        # Consensus analysis
        scores = [r.score for r in results]
        variance = np.var(scores)

        if variance < 0.02:
            lines.append("✅ **Консенсус:** Все методы согласны в оценке")
        elif variance < 0.05:
            lines.append("📊 **Консенсус:** Методы в целом согласны")
        else:
            lines.append("⚠️ **Расхождение:** Методы дают разные оценки")

        return "\n".join(lines)

    def _method_name(self, method: SimilarityMethod) -> str:
        """Get human-readable method name."""
        names = {
            SimilarityMethod.COSINE: "Косинусная схожесть",
            SimilarityMethod.SEMANTIC: "Семантическая схожесть",
            SimilarityMethod.HYBRID: "Гибридная схожесть",
            SimilarityMethod.EUCLIDEAN: "Евклидова дистанция",
            SimilarityMethod.DOT_PRODUCT: "Скалярное произведение",
            SimilarityMethod.JACCARD: "Коэффициент Жаккара"
        }
        return names.get(method, method.value)


class ConsensusAggregator(ISimilarityAggregator):
    """
    Consensus-based aggregator.

    Uses median or voting-based approach instead of weighted average.
    More robust to outliers.
    """

    def __init__(self, method: str = "median"):
        """
        Initialize consensus aggregator.

        Args:
            method: Aggregation method ('median' or 'vote')
        """
        self.method = method

    def aggregate(
        self,
        results: List[SimilarityResult],
        weights: Optional[Dict[SimilarityMethod, float]] = None
    ) -> Tuple[float, str]:
        """
        Aggregate using consensus method.

        Args:
            results: List of similarity results
            weights: Ignored for consensus methods

        Returns:
            Tuple of (final_score, interpretation)
        """
        if not results:
            return 0.0, "Нет результатов для агрегации"

        try:
            scores = [r.score for r in results]

            if self.method == "median":
                final_score = float(np.median(scores))
            elif self.method == "vote":
                # Vote-based: classify into bins and use most common
                final_score = self._vote_based(scores)
            else:
                final_score = float(np.mean(scores))

            interpretation = self._build_interpretation(results, final_score)

            logger.info(
                f"Consensus aggregation ({self.method}) of {len(results)} results: {final_score:.3f}"
            )

            return final_score, interpretation

        except Exception as e:
            logger.error(f"Consensus aggregation failed: {e}")
            return 0.0, f"Ошибка консенсусной агрегации: {str(e)}"

    def _vote_based(self, scores: List[float]) -> float:
        """
        Vote-based aggregation.

        Classify scores into bins and return center of most common bin.
        """
        # Define bins: very_low, low, medium, high, very_high
        bins = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]
        bin_centers = [0.15, 0.4, 0.6, 0.775, 0.925]

        # Classify scores
        bin_counts = [0] * (len(bins) - 1)
        for score in scores:
            for i in range(len(bins) - 1):
                if bins[i] <= score < bins[i+1]:
                    bin_counts[i] += 1
                    break

        # Find most common bin
        max_bin_idx = bin_counts.index(max(bin_counts))

        return bin_centers[max_bin_idx]

    def _build_interpretation(
        self,
        results: List[SimilarityResult],
        final_score: float
    ) -> str:
        """Build interpretation for consensus aggregation."""
        lines = []

        # Overall interpretation
        if final_score > 0.85:
            lines.append("🎯 **Очень высокая схожесть** (консенсусная оценка)")
        elif final_score > 0.7:
            lines.append("✅ **Высокая схожесть** (консенсусная оценка)")
        elif final_score > 0.5:
            lines.append("📊 **Умеренная схожесть** (консенсусная оценка)")
        else:
            lines.append("📉 **Низкая схожесть** (консенсусная оценка)")

        lines.append("")
        lines.append(f"**Консенсусный балл ({self.method}):** {final_score:.2%}")
        lines.append("")

        # Show individual scores
        lines.append("**Оценки методов:**")
        for result in results:
            lines.append(f"  • {result.method.value}: {result.score:.2%}")

        return "\n".join(lines)
