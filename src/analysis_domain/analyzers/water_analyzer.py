"""
Анализатор "воды" в тексте (water level)
"""
import numpy as np
from typing import Dict, Any, List, Optional
from loguru import logger

from ..entities.base_analyzer import BaseAnalyzer
from ..entities.analysis_result import AnalysisResult
from src.text_domain.entities.base_text import BaseText
from src.common.types import AnalysisMode
from src.common.exceptions import AnalysisError


class WaterAnalyzer(BaseAnalyzer):
    """
    Анализатор "воды" (избыточного контента)

    НЕ ТРЕБУЕТ LLM - очень быстрый!
    ТРЕБУЕТ embeddings для точности

    Использует:
    - Семантическую вариативность (низкая = повтор = вода)
    - Избыточность (схожесть соседних чанков = вода)
    - Информационную плотность

    Возвращает:
    - Процент "воды" (0-100%)
    - Информационную плотность
    - Рейтинг (concise/balanced/verbose)
    - Индексы "водянистых" чанков
    """

    def __init__(self):
        """Инициализация анализатора воды"""
        pass

    @property
    def requires_llm(self) -> bool:
        """НЕ требуется LLM"""
        return False

    @property
    def requires_embeddings(self) -> bool:
        """ТРЕБУЮТСЯ embeddings"""
        return True

    async def analyze(
        self,
        text: BaseText,
        mode: AnalysisMode,
        **kwargs
    ) -> AnalysisResult:
        """
        Выполнить анализ воды

        Args:
            text: Текст для анализа
            mode: Режим анализа
            **kwargs: Дополнительные параметры (должны содержать embeddings)

        Returns:
            AnalysisResult: Результат анализа
        """
        try:
            logger.info(f"Начало анализа воды: {text.title}")

            # Получаем чанки и embeddings
            chunks = kwargs.get('chunks', [])
            embeddings = kwargs.get('embeddings', [])

            if not chunks:
                raise AnalysisError(
                    "Анализ воды требует чанки текста",
                    details={"text_id": text.id}
                )

            if embeddings and len(embeddings) > 0:
                # Используем embeddings для точного анализа
                water_data = self._analyze_with_embeddings(chunks, embeddings)
            else:
                # Fallback на эвристику
                logger.warning("Embeddings недоступны, используем эвристику")
                water_data = self._analyze_with_heuristics(chunks)

            logger.info(f"Анализ воды завершён: {water_data['water_percentage']:.1f}%")

            return AnalysisResult(
                analyzer_type=self.__class__.__name__,
                text_id=text.id,
                mode=mode,
                data=water_data
            )

        except Exception as e:
            logger.error(f"Ошибка анализа воды: {e}")
            raise AnalysisError(
                f"Не удалось проанализировать воду: {str(e)}",
                details={"text_id": text.id}
            )

    def _analyze_with_embeddings(
        self,
        chunks: List[Any],
        embeddings: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Анализировать воду используя semantic embeddings

        Args:
            chunks: Список чанков
            embeddings: Список векторов

        Returns:
            Dict: Результаты анализа
        """
        if len(embeddings) != len(chunks):
            logger.warning("Несоответствие количества embeddings, fallback")
            return self._analyze_with_heuristics(chunks)

        # Конвертируем в numpy array
        emb_matrix = np.array([emb for emb in embeddings])

        # 1. Вычисляем семантическую вариативность для каждого чанка
        # Низкая вариативность = низкая информационная плотность
        chunk_variances = []
        for emb in emb_matrix:
            variance = np.var(emb)
            chunk_variances.append(variance)

        # Нормализуем вариативность
        max_var = max(chunk_variances) if chunk_variances else 1.0
        normalized_variances = [v / max_var for v in chunk_variances]

        # 2. Вычисляем избыточность (схожесть соседних чанков)
        redundancy_scores = []
        for i in range(len(emb_matrix) - 1):
            similarity = self._cosine_similarity(emb_matrix[i], emb_matrix[i+1])
            redundancy_scores.append(similarity)

        avg_redundancy = sum(redundancy_scores) / len(redundancy_scores) if redundancy_scores else 0.0

        # 3. Вычисляем информационную плотность
        # Высокая вариативность + низкая избыточность = высокая плотность
        info_density_scores = []
        for i, variance in enumerate(normalized_variances):
            if i < len(redundancy_scores):
                # Низкая избыточность с соседним = более уникальная информация
                uniqueness = 1.0 - redundancy_scores[i]
                density = (variance + uniqueness) / 2.0
            else:
                density = variance

            info_density_scores.append(density)

        # Средняя информационная плотность
        avg_info_density = sum(info_density_scores) / len(info_density_scores)

        # Процент воды = обратная информационная плотность
        water_percentage = (1.0 - avg_info_density) * 100

        # Находим "водянистые" чанки (нижние 20%)
        sorted_indices = sorted(
            range(len(info_density_scores)),
            key=lambda i: info_density_scores[i]
        )
        verbose_count = max(int(len(chunks) * 0.2), 1)
        verbose_chunks = sorted_indices[:verbose_count]

        # Рейтинг
        if water_percentage < 20:
            rating = "concise"
            rating_ru = "лаконичный"
            rating_emoji = "✨"
        elif water_percentage < 50:
            rating = "balanced"
            rating_ru = "сбалансированный"
            rating_emoji = "✅"
        else:
            rating = "verbose"
            rating_ru = "многословный"
            rating_emoji = "💧"

        return {
            "water_percentage": round(water_percentage, 2),
            "info_density": round(avg_info_density, 2),
            "rating": rating,
            "rating_ru": rating_ru,
            "rating_emoji": rating_emoji,
            "verbose_chunks": verbose_chunks,
            "avg_redundancy": round(avg_redundancy, 2)
        }

    def _analyze_with_heuristics(self, chunks: List[Any]) -> Dict[str, Any]:
        """
        Анализировать воду используя эвристику

        Args:
            chunks: Список чанков

        Returns:
            Dict: Результаты анализа
        """
        # Простая эвристика: повторяющиеся слова
        word_frequencies = {}
        total_words = 0

        for chunk in chunks:
            words = chunk.content.lower().split()
            total_words += len(words)

            for word in words:
                if len(word) > 3:  # Игнорируем короткие слова
                    word_frequencies[word] = word_frequencies.get(word, 0) + 1

        # Вычисляем повторяемость
        if total_words == 0:
            water_percentage = 50.0  # Дефолт
        else:
            # Слова, повторяющиеся >5 раз = "вода"
            repetitive_words = sum(
                count for count in word_frequencies.values() if count > 5
            )
            water_percentage = min((repetitive_words / total_words) * 200, 100)

        # Рейтинг
        if water_percentage < 20:
            rating = "concise"
            rating_ru = "лаконичный"
            rating_emoji = "✨"
        elif water_percentage < 50:
            rating = "balanced"
            rating_ru = "сбалансированный"
            rating_emoji = "✅"
        else:
            rating = "verbose"
            rating_ru = "многословный"
            rating_emoji = "💧"

        return {
            "water_percentage": round(water_percentage, 2),
            "info_density": round(1.0 - (water_percentage / 100), 2),
            "rating": rating,
            "rating_ru": rating_ru,
            "rating_emoji": rating_emoji,
            "verbose_chunks": [],
            "avg_redundancy": 0.0,
            "method": "heuristic"
        }

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Вычислить косинусное сходство

        Args:
            vec1: Первый вектор
            vec2: Второй вектор

        Returns:
            float: Косинусное сходство (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def interpret_results(self, result: AnalysisResult) -> str:
        """
        Интерпретировать результаты анализа

        Args:
            result: Результат анализа

        Returns:
            str: Человекочитаемая интерпретация
        """
        data = result.data
        water_percent = data.get('water_percentage', 0)
        info_density = data.get('info_density', 0)
        rating_ru = data.get('rating_ru', 'неизвестный')
        rating_emoji = data.get('rating_emoji', '💧')

        lines = [
            f"💧 **Анализ "воды" в тексте**\n",
            f"{rating_emoji} **Уровень воды**: {water_percent:.1f}%",
            f"📊 **Информационная плотность**: {info_density:.2f}",
            f"⭐ **Рейтинг**: {rating_ru}\n"
        ]

        # Интерпретация
        if water_percent < 20:
            interpretation = (
                "Очень лаконичный текст. Высокая информационная плотность. "
                "Каждое предложение несёт смысл. Отличный результат!"
            )
        elif water_percent < 35:
            interpretation = (
                "Хороший баланс между информацией и читабельностью. "
                "Умеренное количество повторов и описаний."
            )
        elif water_percent < 50:
            interpretation = (
                "Средний уровень воды. Есть повторы и избыточные описания, "
                "но они не критичны."
            )
        elif water_percent < 70:
            interpretation = (
                "Много "воды". Значительное количество повторов, избыточных "
                "описаний и малоинформативных фрагментов. Рекомендуется редактура."
            )
        else:
            interpretation = (
                "Очень высокий уровень воды. Текст перегружен повторами "
                "и избыточными описаниями. Требуется серьёзная редактура."
            )

        lines.append(f"💡 **Интерпретация**: {interpretation}")

        return '\n'.join(lines)
