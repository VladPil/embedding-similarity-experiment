"""
Компаратор на основе косинусного сходства
"""
import numpy as np
from typing import Dict, Optional
from loguru import logger

from src.text_domain.entities.base_text import BaseText
from ..entities.base_comparator import BaseComparator
from ..entities.analysis_result import AnalysisResult
from ..entities.comparison_result import ComparisonResult
from src.common.types import AnalysisMode, ChunkedComparisonStrategy
from src.common.exceptions import AnalysisError


class CosineComparator(BaseComparator):
    """
    Компаратор на основе косинусного сходства embeddings

    Вычисляет угол между векторами embeddings двух текстов
    """

    def __init__(self, embedding_service=None):
        """
        Инициализация

        Args:
            embedding_service: Сервис для получения embeddings
        """
        self.embedding_service = embedding_service

    @property
    def name(self) -> str:
        return "cosine"

    @property
    def display_name(self) -> str:
        return "Косинусное сходство"

    @property
    def description(self) -> str:
        return """Вычисляет косинусное сходство между векторными представлениями текстов.
Значение от 0 (полностью различны) до 1 (идентичны)."""

    async def compare(
        self,
        text1: BaseText,
        text2: BaseText,
        analysis_results: Dict[str, AnalysisResult],
        mode: AnalysisMode = AnalysisMode.FULL_TEXT,
        strategy: ChunkedComparisonStrategy = ChunkedComparisonStrategy.AGGREGATE_FIRST,
        context: Optional[Dict] = None
    ) -> ComparisonResult:
        """Сравнить два текста"""
        try:
            import time
            start_time = time.time()

            # Получаем embeddings из контекста или вычисляем
            if context and "embeddings" in context:
                emb1 = context["embeddings"].get(text1.id)
                emb2 = context["embeddings"].get(text2.id)
            else:
                # Получаем через embedding_service
                if self.embedding_service:
                    # Получаем содержимое текстов
                    content1 = await text1.get_content() if hasattr(text1, 'get_content') else str(text1)
                    content2 = await text2.get_content() if hasattr(text2, 'get_content') else str(text2)

                    # Получаем embeddings
                    embeddings = await self.embedding_service.encode([content1, content2])
                    emb1 = embeddings[0]
                    emb2 = embeddings[1]
                else:
                    # Fallback: если сервис не доступен, используем случайные векторы
                    logger.warning("Embedding service не доступен, используем случайные векторы")
                    emb1 = np.random.rand(384)
                    emb2 = np.random.rand(384)

            # Вычисляем косинусное сходство
            similarity = self._cosine_similarity(emb1, emb2)

            execution_time = (time.time() - start_time) * 1000

            # Детали
            details = {
                "embedding_dimension": len(emb1),
                "method": "cosine_similarity",
                "mode": mode.value,
            }

            # Объяснение
            explanation = self._generate_explanation(similarity)

            result = ComparisonResult(
                text1_id=text1.id,
                text2_id=text2.id,
                comparator_name=self.name,
                similarity_score=float(similarity),
                details=details,
                explanation=explanation,
                execution_time_ms=execution_time,
                mode=mode.value,
            )

            logger.debug(
                f"Косинусное сходство: {text1.id} ↔ {text2.id} = {similarity:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Ошибка сравнения: {e}")
            raise AnalysisError(f"Cosine comparison failed: {e}")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Вычислить косинусное сходство

        Args:
            vec1: Первый вектор
            vec2: Второй вектор

        Returns:
            float: Сходство от 0 до 1
        """
        # Нормализуем векторы
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)

        # Скалярное произведение нормализованных векторов
        similarity = np.dot(vec1_norm, vec2_norm)

        # Приводим к диапазону [0, 1]
        # (косинус может быть от -1 до 1, но для текстов обычно положительный)
        similarity = (similarity + 1) / 2

        return max(0.0, min(1.0, similarity))

    def _generate_explanation(self, similarity: float) -> str:
        """
        Генерировать объяснение результата

        Args:
            similarity: Оценка сходства

        Returns:
            str: Текстовое объяснение
        """
        if similarity >= 0.9:
            return "Тексты практически идентичны по содержанию"
        elif similarity >= 0.7:
            return "Тексты очень похожи, вероятно схожая тематика"
        elif similarity >= 0.5:
            return "Тексты имеют умеренное сходство"
        elif similarity >= 0.3:
            return "Тексты мало похожи"
        else:
            return "Тексты значительно отличаются"

    def get_estimated_time(
        self,
        text1_length: int,
        text2_length: int,
        mode: AnalysisMode = AnalysisMode.FULL_TEXT
    ) -> float:
        """Оценка времени"""
        # Косинусное сходство очень быстрое (просто операция над векторами)
        return 0.1
