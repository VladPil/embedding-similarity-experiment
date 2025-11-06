"""
LLM-based analysis strategy.
"""

from typing import Dict, Any, Optional

from server.services.strategies.base import AnalysisStrategy
from server.core.analysis.llm_manager import LLMManager


class LLMAnalysisStrategy(AnalysisStrategy):
    """Strategy for LLM-based text analysis."""

    def __init__(self, llm_manager: Optional[LLMManager] = None):
        """
        Initialize LLM analysis strategy.

        Args:
            llm_manager: LLMManager instance (optional, will create if not provided)
        """
        self.llm_manager = llm_manager

    async def analyze(
        self,
        text1_content: str,
        text2_content: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform LLM-based analysis.

        Args:
            text1_content: First text content
            text2_content: Second text content
            params: Analysis parameters including task_type, text1_title, text2_title

        Returns:
            Dictionary with LLM analysis results
        """
        if self.llm_manager is None:
            from server.core.analysis.llm_manager import get_llm_manager
            self.llm_manager = await get_llm_manager()

        # Get task type
        task_type = params.get("task_type", "compare")

        # Get model parameter - create new LLM manager if model specified
        model = params.get("model")
        if model and model != getattr(self.llm_manager, 'model_key', None):
            from server.core.analysis.llm_manager import LLMManager
            self.llm_manager = LLMManager(model_key=model)

        # Prepare kwargs
        kwargs = {}
        if "text1_title" in params:
            kwargs["text1_title"] = params["text1_title"]
        if "text2_title" in params:
            kwargs["text2_title"] = params["text2_title"]
        if "max_words" in params:
            kwargs["max_words"] = params["max_words"]

        # Execute LLM task
        result = await self.llm_manager.execute_task(
            task_type=task_type,
            text1=text1_content,
            text2=text2_content if task_type not in ["summary", "themes", "sentiment", "single_report"] else None,
            **kwargs
        )

        # Normalize result - ensure 'similarity' field exists
        if "overall_similarity" in result and "similarity" not in result:
            result["similarity"] = result["overall_similarity"]

        # Build detailed interpretation
        if "interpretation" not in result:
            if "explanation" in result:
                result["interpretation"] = result["explanation"]
            else:
                # Build detailed breakdown
                interpretation_parts = []

                if "overall_similarity" in result:
                    overall = result["overall_similarity"]
                    interpretation_parts.append(f"🎯 Общая схожесть: {overall * 100:.1f}%")

                interpretation_parts.append("\n📊 Детальная разбивка:")

                metrics = [
                    ("plot_similarity", "📖 Сюжет"),
                    ("style_similarity", "✍️ Стиль"),
                    ("genre_similarity", "🎭 Жанр"),
                    ("characters_similarity", "👥 Персонажи"),
                    ("language_similarity", "🗣️ Язык")
                ]

                for key, label in metrics:
                    if key in result:
                        value = result[key]
                        interpretation_parts.append(f"  {label}: {value * 100:.1f}%")

                if "model" in result:
                    interpretation_parts.append(f"\n🤖 Модель: {result['model']}")

                result["interpretation"] = "\n".join(interpretation_parts)

        return result

    def get_type(self) -> str:
        """Get analysis type identifier."""
        return "llm"
