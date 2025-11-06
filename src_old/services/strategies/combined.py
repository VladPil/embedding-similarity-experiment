"""
Combined analysis strategy.
Combines multiple analysis strategies into one comprehensive result.
"""

from typing import Dict, Any, List

from server.services.strategies.base import AnalysisStrategy


class CombinedAnalysisStrategy(AnalysisStrategy):
    """Strategy for combined analysis using multiple strategies."""

    def __init__(self, strategies: Dict[str, AnalysisStrategy]):
        """
        Initialize combined strategy.

        Args:
            strategies: Dictionary of strategy_name -> strategy_instance
        """
        self.strategies = strategies

    async def analyze(
        self,
        text1_content: str,
        text2_content: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform combined analysis using multiple strategies.

        Args:
            text1_content: First text content
            text2_content: Second text content
            params: Analysis parameters including 'strategies' list and 'weights'

        Returns:
            Dictionary with results from all strategies and combined metrics
        """
        # Get list of strategies to run (include LLM by default for richer analysis)
        strategy_names = params.get("strategies", ["semantic", "style", "tfidf", "emotion"])

        # Default weights for each strategy (can be customized)
        default_weights = {
            "semantic": 0.35,   # Most important - captures meaning
            "style": 0.20,      # Important - captures writing style
            "tfidf": 0.15,      # Moderate - captures keywords
            "emotion": 0.15,    # Moderate - captures emotional tone
            "llm": 0.15         # Moderate - LLM-based deep analysis
        }

        weights = params.get("weights", default_weights)

        results = {}
        weighted_scores = []
        total_weight = 0

        # Run each strategy
        for strategy_name in strategy_names:
            if strategy_name in self.strategies:
                strategy = self.strategies[strategy_name]
                weight = weights.get(strategy_name, 1.0 / len(strategy_names))

                try:
                    result = await strategy.analyze(text1_content, text2_content, params)
                    results[strategy_name] = {
                        **result,
                        "weight": weight,
                        "weighted_score": result.get("similarity", 0) * weight
                    }

                    # Collect weighted scores
                    if "similarity" in result:
                        weighted_scores.append(result["similarity"] * weight)
                        total_weight += weight

                except Exception as e:
                    results[strategy_name] = {
                        "error": str(e),
                        "weight": weight,
                        "weighted_score": 0
                    }

        # Calculate combined metrics
        weighted_similarity = sum(weighted_scores) / total_weight if total_weight > 0 else 0

        # Calculate simple average for comparison
        simple_similarities = [r.get("similarity", 0) for r in results.values() if "similarity" in r]
        avg_similarity = sum(simple_similarities) / len(simple_similarities) if simple_similarities else 0

        # Build detailed interpretation
        strategy_names_ru = {
            "semantic": "Семантический анализ",
            "style": "Стилистический анализ",
            "tfidf": "TF-IDF анализ",
            "emotion": "Эмоциональный анализ",
            "llm": "Анализ ИИ"
        }

        interpretation_parts = [
            f"🔍 Комплексный анализ текстов",
            f"═══════════════════════════",
            f"",
            f"▪ Итоговая схожесть: {weighted_similarity * 100:.1f}%",
            f"",
            f"📊 Детализация по методам:"
        ]

        for name, result in results.items():
            if "error" not in result:
                score = result.get("similarity", 0)
                weight = result.get("weight", 0)
                contribution = score * weight
                ru_name = strategy_names_ru.get(name, name)
                interpretation_parts.append(f"")
                interpretation_parts.append(
                    f"▫ {ru_name}: {score * 100:.1f}%"
                )
                interpretation_parts.append(
                    f"    Вес: {weight * 100:.0f}% → Вклад: {contribution * 100:.1f}%"
                )

        interpretation_parts.extend([
            f"",
            f"➤ Интерпретация результата:",
            f"  {self._get_combined_interpretation(weighted_similarity)}",
            f"",
            f"ℹ️ Методология:",
            f"  • Взвешенное усреднение результатов",
            f"  • Простое среднее: {avg_similarity * 100:.1f}%",
            f"  • Учёт важности каждого метода",
            f"  • Комплексная оценка схожести"
        ])

        return {
            "results": results,
            "similarity": float(weighted_similarity),  # Main similarity score
            "combined_similarity": float(weighted_similarity),
            "average_similarity": float(avg_similarity),
            "interpretation": "\n".join(interpretation_parts),
            "strategies_used": strategy_names,
            "weights_used": {k: v for k, v in weights.items() if k in strategy_names},
            "total_weight": total_weight
        }

    def _get_combined_interpretation(self, similarity: float) -> str:
        """Get interpretation for combined similarity score."""
        if similarity >= 0.9:
            return "Тексты практически идентичны по всем параметрам"
        elif similarity >= 0.75:
            return "Очень высокая схожесть - тексты явно связаны"
        elif similarity >= 0.6:
            return "Высокая схожесть - заметное сходство по большинству параметров"
        elif similarity >= 0.45:
            return "Умеренная схожесть - есть общие черты"
        elif similarity >= 0.3:
            return "Низкая схожесть - мало общего"
        else:
            return "Тексты существенно различаются"

    def get_type(self) -> str:
        """Get analysis type identifier."""
        return "combined"
