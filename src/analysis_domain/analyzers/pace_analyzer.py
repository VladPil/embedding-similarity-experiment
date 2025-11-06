"""
Анализатор темпа повествования (pace)
"""
from typing import Dict, Any, List
from loguru import logger

from ..entities.base_analyzer import BaseAnalyzer
from ..entities.analysis_result import AnalysisResult
from ..helpers.chunk_indexer import ChunkIndexer
from src.text_domain.entities.base_text import BaseText
from src.common.types import AnalysisMode
from src.common.exceptions import AnalysisError


class PaceAnalyzer(BaseAnalyzer):
    """
    Анализатор темпа повествования

    НЕ ТРЕБУЕТ LLM - очень быстрый анализ!

    Использует:
    - Плотность событий (event density)
    - Соотношение диалогов
    - Длину предложений
    - Ключевые слова действий

    Возвращает:
    - Общий темп (slow/medium/fast)
    - Скор темпа (0-10)
    - Timeline темпа
    - Статистику
    """

    def __init__(self):
        """Инициализация анализатора темпа"""
        self.indexer = ChunkIndexer()

    @property
    def requires_llm(self) -> bool:
        """НЕ требуется LLM"""
        return False

    @property
    def requires_embeddings(self) -> bool:
        """Не требуются embeddings"""
        return False

    async def analyze(
        self,
        text: BaseText,
        mode: AnalysisMode,
        **kwargs
    ) -> AnalysisResult:
        """
        Выполнить анализ темпа

        Args:
            text: Текст для анализа
            mode: Режим анализа
            **kwargs: Дополнительные параметры

        Returns:
            AnalysisResult: Результат анализа
        """
        try:
            logger.info(f"Начало анализа темпа: {text.title}")

            # Получаем чанки
            chunks = kwargs.get('chunks', [])
            if not chunks:
                raise AnalysisError(
                    "Анализ темпа требует чанки текста",
                    details={"text_id": text.id}
                )

            # Вычисляем темп для каждого чанка
            pace_scores = []

            for chunk in chunks:
                # Плотность событий
                density = self.indexer._calculate_event_density(chunk.content)

                # Соотношение диалогов
                dialogue_ratio = self.indexer._calculate_dialogue_ratio(chunk.content)

                # Комбинированный скор темпа
                pace_score = self._calculate_pace_score(density, dialogue_ratio)

                pace_scores.append({
                    "position": chunk.metadata.get('position_ratio', 0.0),
                    "score": pace_score,
                    "event_density": density,
                    "dialogue_ratio": dialogue_ratio
                })

            # Общий темп (медиана)
            scores_only = [p['score'] for p in pace_scores]
            overall_score = self._median(scores_only)

            # Определяем рейтинг
            if overall_score < 4:
                pace_rating = "slow"
                pace_emoji = "🐌"
                pace_ru = "медленный"
            elif overall_score < 7:
                pace_rating = "medium"
                pace_emoji = "🚶"
                pace_ru = "средний"
            else:
                pace_rating = "fast"
                pace_emoji = "🏃"
                pace_ru = "быстрый"

            # Timeline (семплируем каждые 10% текста)
            timeline = self._create_timeline(pace_scores, sample_rate=0.1)

            # Статистика
            stats = self._calculate_statistics(pace_scores)

            logger.info(f"Темп: {pace_ru} ({overall_score:.1f}/10)")

            return AnalysisResult(
                analyzer_type=self.__class__.__name__,
                text_id=text.id,
                mode=mode,
                data={
                    "overall_pace": pace_rating,
                    "pace_emoji": pace_emoji,
                    "pace_ru": pace_ru,
                    "pace_score": round(overall_score, 2),
                    "timeline": timeline,
                    "statistics": stats
                }
            )

        except Exception as e:
            logger.error(f"Ошибка анализа темпа: {e}")
            raise AnalysisError(
                f"Не удалось проанализировать темп: {str(e)}",
                details={"text_id": text.id}
            )

    def _calculate_pace_score(self, event_density: float, dialogue_ratio: float) -> float:
        """
        Вычислить скор темпа

        Args:
            event_density: Плотность событий (0-1)
            dialogue_ratio: Соотношение диалогов (0-1)

        Returns:
            float: Скор темпа (0-10)
        """
        # События вносят 70%, диалоги 30%
        score = (event_density * 7.0) + (dialogue_ratio * 3.0)
        return min(score * 10, 10.0)

    def _median(self, values: List[float]) -> float:
        """Вычислить медиану"""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)

        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            return sorted_values[n//2]

    def _create_timeline(
        self,
        pace_scores: List[Dict],
        sample_rate: float = 0.1
    ) -> List[Dict]:
        """
        Создать timeline темпа

        Args:
            pace_scores: Все скоры темпа
            sample_rate: Частота семплирования (0-1)

        Returns:
            List[Dict]: Timeline
        """
        if not pace_scores:
            return []

        # Семплируем через интервалы
        step = max(int(len(pace_scores) * sample_rate), 1)
        timeline = []

        for i in range(0, len(pace_scores), step):
            sample = pace_scores[i]
            timeline.append({
                "position": sample['position'],
                "pace": round(sample['score'], 2)
            })

        return timeline

    def _calculate_statistics(self, pace_scores: List[Dict]) -> Dict:
        """
        Вычислить статистику темпа

        Args:
            pace_scores: Все скоры темпа

        Returns:
            Dict: Статистика
        """
        if not pace_scores:
            return {}

        scores = [p['score'] for p in pace_scores]
        event_densities = [p['event_density'] for p in pace_scores]
        dialogue_ratios = [p['dialogue_ratio'] for p in pace_scores]

        return {
            "min_pace": round(min(scores), 2),
            "max_pace": round(max(scores), 2),
            "avg_pace": round(sum(scores) / len(scores), 2),
            "median_pace": round(self._median(scores), 2),
            "avg_event_density": round(sum(event_densities) / len(event_densities), 2),
            "avg_dialogue_ratio": round(sum(dialogue_ratios) / len(dialogue_ratios), 2),
            "pace_variance": round(self._variance(scores), 2)
        }

    def _variance(self, values: List[float]) -> float:
        """Вычислить дисперсию"""
        if not values:
            return 0.0

        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def interpret_results(self, result: AnalysisResult) -> str:
        """
        Интерпретировать результаты анализа

        Args:
            result: Результат анализа

        Returns:
            str: Человекочитаемая интерпретация
        """
        data = result.data
        pace_rating = data.get('overall_pace', 'unknown')
        pace_score = data.get('pace_score', 0)
        pace_emoji = data.get('pace_emoji', '📊')
        pace_ru = data.get('pace_ru', 'неизвестный')
        stats = data.get('statistics', {})

        lines = [
            f"🏃 **Анализ темпа повествования**\n",
            f"{pace_emoji} **Общий темп**: {pace_ru} ({pace_score:.1f}/10)\n"
        ]

        # Интерпретация
        if pace_rating == "fast":
            interpretation = (
                "Быстрый темп повествования. Много действий, динамичные сцены, "
                "короткие предложения. Держит читателя в напряжении."
            )
        elif pace_rating == "medium":
            interpretation = (
                "Средний темп повествования. Хороший баланс между действием "
                "и описаниями. Комфортное чтение."
            )
        else:
            interpretation = (
                "Медленный темп повествования. Много описаний, размышлений, "
                "атмосферных сцен. Фокус на деталях."
            )

        lines.append(f"💡 **Интерпретация**: {interpretation}\n")

        # Статистика
        if stats:
            lines.append(f"📊 **Статистика**:")
            lines.append(f"   • Мин. темп: {stats.get('min_pace', 0)}/10")
            lines.append(f"   • Макс. темп: {stats.get('max_pace', 0)}/10")
            lines.append(f"   • Средний: {stats.get('avg_pace', 0)}/10")
            lines.append(f"   • Плотность событий: {stats.get('avg_event_density', 0):.2f}")
            lines.append(f"   • Доля диалогов: {stats.get('avg_dialogue_ratio', 0):.2f}")

        return '\n'.join(lines)
