"""
Анализатор эмоциональной окраски текста
"""
import json
from typing import Optional, Dict, Any
from loguru import logger

from src.text_domain.entities.base_text import BaseText
from src.text_domain.entities.chunking_strategy import ChunkingStrategy
from ..entities.base_analyzer import BaseAnalyzer
from ..entities.analysis_result import AnalysisResult
from src.common.types import AnalysisMode
from src.common.exceptions import AnalysisError


class EmotionAnalyzer(BaseAnalyzer):
    """
    Анализатор эмоциональной окраски и тональности текста
    """

    def __init__(self, llm_service=None):
        """Инициализация"""
        self.llm_service = llm_service

    @property
    def name(self) -> str:
        return "emotion"

    @property
    def display_name(self) -> str:
        return "Анализ эмоций"

    @property
    def description(self) -> str:
        return "Определяет эмоциональную окраску и тональность текста"

    @property
    def requires_llm(self) -> bool:
        return True

    async def analyze(
        self,
        text: BaseText,
        mode: AnalysisMode = AnalysisMode.FULL_TEXT,
        chunking_strategy: Optional[ChunkingStrategy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """Анализ эмоций"""
        try:
            import time
            start_time = time.time()

            content = await text.get_content()
            analysis_text = content[:3000]

            # Мок-данные (TODO: реальный LLM)
            result_data = {
                "dominant_emotion": "нейтральная",
                "emotions": {
                    "радость": 0.2,
                    "грусть": 0.1,
                    "гнев": 0.05,
                    "страх": 0.1,
                    "удивление": 0.15,
                },
                "sentiment": "нейтральный",
                "sentiment_score": 0.55,  # 0-1
            }

            execution_time = (time.time() - start_time) * 1000

            result = AnalysisResult(
                text_id=text.id,
                analyzer_name=self.name,
                data=result_data,
                execution_time_ms=execution_time,
                mode=mode.value,
            )

            result.interpretation = self.interpret_results(result)
            return result

        except Exception as e:
            logger.error(f"Ошибка анализа эмоций: {e}")
            raise AnalysisError(f"Emotion analysis failed: {e}")

    def interpret_results(self, result: AnalysisResult) -> str:
        """Интерпретация"""
        data = result.data
        dominant = data.get("dominant_emotion", "н/д")
        sentiment = data.get("sentiment", "н/д")
        emotions = data.get("emotions", {})

        lines = [
            f"😊 Эмоциональный анализ:",
            f"Доминирующая эмоция: {dominant}",
            f"Тональность: {sentiment}",
            "\nРаспределение эмоций:"
        ]

        for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {emotion}: {score:.0%}")

        return "\n".join(lines)
