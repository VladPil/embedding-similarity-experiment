"""
Water Level Analysis Strategy.
Analyzes "water" (filler content) using semantic density.
"""

from typing import Dict, Any, List
import numpy as np
from loguru import logger

from server.core.analysis.base import IAnalysisStrategy, AnalysisContext, AnalysisType


class WaterAnalysisStrategy(IAnalysisStrategy):
    """
    Strategy for water level analysis.

    - NO LLM required (very fast!)
    - Uses semantic embeddings for information density
    - Returns water percentage and verbose chunks
    """

    def get_type(self) -> AnalysisType:
        """Get analysis type identifier."""
        return AnalysisType.WATER

    def requires_llm(self) -> bool:
        """Water analysis does NOT require LLM."""
        return False

    def requires_embeddings(self) -> bool:
        """Water analysis requires embeddings for best results."""
        return True

    def get_estimated_time(self, chunk_count: int) -> float:
        """Estimate 5 seconds (very fast)."""
        return 5.0

    async def analyze(self, context: AnalysisContext) -> Dict[str, Any]:
        """
        Execute water level analysis.

        Args:
            context: Analysis context with chunks and embeddings

        Returns:
            {
                "water_percentage": float,
                "info_density": float,
                "rating": str,
                "verbose_chunks": List[int]
            }
        """
        try:
            logger.info("Starting water level analysis...")

            if context.embeddings and len(context.embeddings) > 0:
                # Use embeddings for accurate analysis
                water_data = self._analyze_with_embeddings(
                    context.chunks,
                    context.embeddings
                )
            else:
                # Fallback to heuristic analysis
                logger.warning("No embeddings available, using heuristic analysis")
                water_data = self._analyze_with_heuristics(context.chunks)

            logger.info(
                f"Water analysis complete: {water_data['water_percentage']:.1f}% water"
            )

            return water_data

        except Exception as e:
            logger.error(f"Water analysis failed: {e}")
            return {
                "water_percentage": 0.0,
                "info_density": 0.0,
                "rating": "unknown",
                "verbose_chunks": [],
                "error": str(e)
            }

    def _analyze_with_embeddings(
        self,
        chunks: List[Any],
        embeddings: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Analyze water level using semantic embeddings.

        Method:
        - Calculate semantic variance (low variance = repetitive = water)
        - Compare chunk similarity (high similarity = redundant = water)
        - Semantic density score
        """
        if len(embeddings) != len(chunks):
            logger.warning("Embeddings count mismatch, falling back to heuristics")
            return self._analyze_with_heuristics(chunks)

        # Convert to numpy array
        emb_matrix = np.array([emb for emb in embeddings])

        # 1. Calculate semantic variance per chunk
        # Lower variance within chunk = less information density
        chunk_variances = []
        for emb in emb_matrix:
            variance = np.var(emb)
            chunk_variances.append(variance)

        # Normalize variances
        max_var = max(chunk_variances) if chunk_variances else 1.0
        normalized_variances = [v / max_var for v in chunk_variances]

        # 2. Calculate redundancy (similarity between consecutive chunks)
        redundancy_scores = []
        for i in range(len(emb_matrix) - 1):
            similarity = self._cosine_similarity(emb_matrix[i], emb_matrix[i+1])
            redundancy_scores.append(similarity)

        avg_redundancy = sum(redundancy_scores) / len(redundancy_scores) if redundancy_scores else 0.0

        # 3. Calculate information density
        # High variance + low redundancy = high information density
        info_density_scores = []
        for i, variance in enumerate(normalized_variances):
            if i < len(redundancy_scores):
                # Low redundancy with next chunk = more unique info
                uniqueness = 1.0 - redundancy_scores[i]
                density = (variance + uniqueness) / 2.0
            else:
                density = variance

            info_density_scores.append(density)

        # Average information density
        avg_info_density = sum(info_density_scores) / len(info_density_scores)

        # Water percentage = inverse of information density
        water_percentage = (1.0 - avg_info_density) * 100

        # Find verbose (watery) chunks (bottom 20%)
        sorted_indices = sorted(
            range(len(info_density_scores)),
            key=lambda i: info_density_scores[i]
        )
        verbose_count = max(int(len(chunks) * 0.2), 1)
        verbose_chunks = sorted_indices[:verbose_count]

        # Rating
        if water_percentage < 20:
            rating = "excellent"
        elif water_percentage < 40:
            rating = "good"
        elif water_percentage < 60:
            rating = "moderate"
        else:
            rating = "high"

        return {
            "water_percentage": round(water_percentage, 1),
            "info_density": round(avg_info_density, 3),
            "rating": rating,
            "verbose_chunks": [chunks[i].index for i in verbose_chunks],
            "redundancy_score": round(avg_redundancy, 3)
        }

    def _analyze_with_heuristics(self, chunks: List[Any]) -> Dict[str, Any]:
        """
        Fallback analysis using text heuristics.

        Heuristics:
        - Repetition of words/phrases
        - Low lexical diversity
        - High description ratio
        """
        water_scores = []

        for chunk in chunks:
            text = chunk.text.lower()
            words = text.split()

            if not words:
                water_scores.append(1.0)
                continue

            # Lexical diversity (unique words / total words)
            unique_words = len(set(words))
            diversity = unique_words / len(words)

            # Low diversity = more water
            water_score = 1.0 - diversity

            water_scores.append(water_score)

        avg_water = sum(water_scores) / len(water_scores) if water_scores else 0.5
        water_percentage = avg_water * 100

        # Find most watery chunks
        sorted_indices = sorted(
            range(len(water_scores)),
            key=lambda i: water_scores[i],
            reverse=True
        )
        verbose_count = max(int(len(chunks) * 0.2), 1)
        verbose_chunks = sorted_indices[:verbose_count]

        # Rating
        if water_percentage < 30:
            rating = "good"
        elif water_percentage < 50:
            rating = "moderate"
        else:
            rating = "high"

        return {
            "water_percentage": round(water_percentage, 1),
            "info_density": round(1.0 - avg_water, 3),
            "rating": rating,
            "verbose_chunks": [chunks[i].index for i in verbose_chunks],
            "method": "heuristic"
        }

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def interpret_results(self, results: Dict[str, Any]) -> str:
        """
        Interpret water analysis results for UI display.

        Args:
            results: Water analysis results

        Returns:
            Human-readable interpretation
        """
        if 'error' in results:
            return f"❌ Ошибка анализа воды: {results['error']}"

        water_pct = results.get('water_percentage', 0)
        info_density = results.get('info_density', 0)
        rating = results.get('rating', 'unknown')
        verbose_chunks = results.get('verbose_chunks', [])
        method = results.get('method', 'embeddings')

        # Rating emoji
        rating_emoji = {
            'excellent': '🌟',
            'good': '✅',
            'moderate': '⚠️',
            'high': '❌',
            'unknown': '❓'
        }
        emoji = rating_emoji.get(rating, '❓')

        # Build interpretation
        lines = [
            f"💧 **Анализ уровня \"воды\" в тексте**\n",
            f"{emoji} **Оценка**: {rating.upper()}\n",
            f"📊 **Метрики**:",
            f"   • Уровень воды: {water_pct:.1f}%",
            f"   • Плотность информации: {info_density:.3f}",
            f"   • Метод анализа: {method}\n"
        ]

        # Interpretation based on water percentage
        if water_pct < 20:
            interpretation = (
                "Отличный результат! Текст очень плотный и информативный. "
                "Каждое предложение несет смысловую нагрузку."
            )
        elif water_pct < 40:
            interpretation = (
                "Хороший результат. Текст достаточно плотный, "
                "хотя местами встречаются описательные фрагменты."
            )
        elif water_pct < 60:
            interpretation = (
                "Умеренный уровень воды. Текст содержит заметное количество "
                "повторений и описательных элементов, но это может быть стилистическим выбором."
            )
        else:
            interpretation = (
                "Высокий уровень воды. Текст содержит много повторений, "
                "избыточных описаний или семантически бедных фрагментов. "
                "Рекомендуется редактирование для повышения плотности."
            )

        lines.append(f"💡 **Интерпретация**: {interpretation}\n")

        # Verbose chunks info
        if verbose_chunks:
            chunk_count = len(verbose_chunks)
            lines.append(
                f"🔍 **Обнаружено {chunk_count} наиболее \"водных\" фрагментов** "
                f"(индексы: {', '.join(map(str, verbose_chunks[:5]))}{'...' if chunk_count > 5 else ''})"
            )

        # Recommendations
        if water_pct > 50:
            lines.append(
                f"\n📝 **Рекомендации**:",
                "   • Сократите повторяющиеся описания",
                "   • Усильте динамику повествования",
                "   • Удалите избыточные пояснения"
            )

        return '\n'.join(lines)
