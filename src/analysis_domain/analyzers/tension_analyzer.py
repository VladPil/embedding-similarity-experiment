"""
Анализатор напряжения (tension)
"""
import json
import re
from typing import Dict, Any, List, Optional
from loguru import logger

from ..entities.base_analyzer import BaseAnalyzer
from ..entities.analysis_result import AnalysisResult
from ..entities.prompt_template import PromptTemplate
from ..helpers.chunk_indexer import ChunkIndexer
from src.text_domain.entities.base_text import BaseText
from src.common.types import AnalysisMode
from src.common.exceptions import AnalysisError


class TensionAnalyzer(BaseAnalyzer):
    """
    Анализатор напряжения в произведении

    Находит пики напряжения и строит timeline

    Использует:
    - ChunkIndexer для отбора напряжённых чанков
    - LLM для описания источников напряжения
    - Ключевые слова для оценки уровня напряжения

    Возвращает:
    - Средний уровень напряжения (0-10)
    - Timeline напряжения
    - Пики напряжения с описаниями
    """

    def __init__(
        self,
        llm_service: Optional[Any] = None,
        prompt_template: Optional[PromptTemplate] = None,
        tension_threshold: float = 6.0,
        max_peaks: int = 15
    ):
        """
        Инициализация анализатора напряжения

        Args:
            llm_service: Сервис для работы с LLM
            prompt_template: Шаблон промпта
            tension_threshold: Порог напряжения для анализа
            max_peaks: Максимальное количество пиков
        """
        self.llm_service = llm_service
        self.prompt_template = prompt_template or self._create_default_prompt()
        self.tension_threshold = tension_threshold
        self.max_peaks = max_peaks
        self.indexer = ChunkIndexer()

    @property
    def requires_llm(self) -> bool:
        """Требуется LLM для описания пиков"""
        return True

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
        Выполнить анализ напряжения

        Args:
            text: Текст для анализа
            mode: Режим анализа
            **kwargs: Дополнительные параметры

        Returns:
            AnalysisResult: Результат анализа
        """
        try:
            logger.info(f"Начало анализа напряжения: {text.title}")

            # Получаем чанки
            chunks = kwargs.get('chunks', [])
            if not chunks:
                raise AnalysisError(
                    "Анализ напряжения требует чанки текста",
                    details={"text_id": text.id}
                )

            # Вычисляем среднее напряжение по всем чанкам
            all_scores = [
                self.indexer._calculate_tension_from_keywords(chunk.content)
                for chunk in chunks
            ]
            average_tension = sum(all_scores) / len(all_scores) if all_scores else 0.0

            # Строим индекс напряжённых чанков
            tension_index = self.indexer.build_tension_index(
                chunks,
                threshold=self.tension_threshold / 10  # Конвертируем в 0-1
            )

            logger.info(
                f"Индекс напряжения: {len(tension_index.chunk_indices)} пиков "
                f"({tension_index.coverage*100:.1f}% охват)"
            )

            # Получаем напряжённые чанки
            tension_chunks = self.indexer.get_chunk_subset(chunks, 'tension')

            if not tension_chunks:
                logger.info("Не найдено пиков напряжения")
                return AnalysisResult(
                    analyzer_type=self.__class__.__name__,
                    text_id=text.id,
                    mode=mode,
                    data={
                        "average_tension": round(average_tension, 2),
                        "timeline": [],
                        "peaks": [],
                        "peak_count": 0,
                        "analysis_coverage": 0.0
                    }
                )

            # Анализируем пики с LLM
            tension_points = await self._analyze_peaks_with_llm(
                tension_chunks,
                tension_index
            )

            logger.info(f"Проанализировано {len(tension_points)} пиков напряжения")

            return AnalysisResult(
                analyzer_type=self.__class__.__name__,
                text_id=text.id,
                mode=mode,
                data={
                    "average_tension": round(average_tension, 2),
                    "timeline": tension_points,
                    "peaks": [p['position'] for p in tension_points],
                    "peak_count": len(tension_points),
                    "analysis_coverage": tension_index.coverage
                }
            )

        except Exception as e:
            logger.error(f"Ошибка анализа напряжения: {e}")
            raise AnalysisError(
                f"Не удалось проанализировать напряжение: {str(e)}",
                details={"text_id": text.id}
            )

    async def _analyze_peaks_with_llm(
        self,
        chunks: List[Any],
        tension_index: Any
    ) -> List[Dict[str, Any]]:
        """
        Анализировать пики напряжения с помощью LLM

        Args:
            chunks: Список напряжённых чанков
            tension_index: Индекс напряжения

        Returns:
            List[Dict]: Список точек напряжения
        """
        tension_points = []

        # Ограничиваем количество пиков
        chunks_to_process = chunks[:self.max_peaks]

        for chunk in chunks_to_process:
            try:
                # Формируем промпт
                prompt = self._format_tension_prompt(chunk.content)

                # LLM запрос
                result = await self.llm_service.generate(
                    prompt=prompt,
                    max_tokens=256,
                    temperature=0.3
                )

                # Парсим ответ
                tension_data = self._parse_tension_response(result)

                if tension_data:
                    # Получаем score из индекса
                    chunk_idx_in_index = tension_index.chunk_indices.index(chunk.index)
                    score = tension_index.scores[chunk_idx_in_index]

                    tension_point = {
                        "position": chunk.metadata.get('position_ratio', 0.0),
                        "score": round(score, 2),
                        "source": tension_data.get('source', 'unknown'),
                        "description": tension_data.get('description', ''),
                        "excerpt": tension_data.get('excerpt', chunk.content[:200])
                    }

                    tension_points.append(tension_point)

            except Exception as e:
                logger.warning(f"Ошибка анализа пика в чанке {chunk.index}: {e}")
                continue

        # Сортируем по позиции
        tension_points.sort(key=lambda x: x['position'])

        return tension_points

    def _format_tension_prompt(self, chunk_text: str) -> str:
        """
        Форматировать промпт для анализа напряжения

        Args:
            chunk_text: Текст чанка

        Returns:
            str: Отформатированный промпт
        """
        prompt = f"""Проанализируй фрагмент текста и определи источник напряжения.

Фрагмент:
{chunk_text[:1000]}

Задача:
1. Определи основной источник напряжения: conflict (конфликт), danger (опасность), mystery (загадка), emotion (эмоция), time_pressure (нехватка времени)
2. Кратко опиши (1-2 предложения), что создаёт напряжение
3. Приведи короткую цитату-доказательство (до 50 символов)

Ответь СТРОГО в формате JSON:
{{
    "source": "conflict|danger|mystery|emotion|time_pressure",
    "description": "краткое описание",
    "excerpt": "цитата из текста"
}}

JSON:"""

        return prompt

    def _parse_tension_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Распарсить ответ LLM

        Args:
            llm_response: Ответ LLM

        Returns:
            Dict: Данные напряжения
        """
        try:
            # Извлекаем JSON
            json_match = re.search(r'\{[^{}]*\}', llm_response, re.DOTALL)

            if json_match:
                parsed = json.loads(json_match.group(0))
                return parsed

            return {}

        except Exception as e:
            logger.warning(f"Ошибка парсинга ответа напряжения: {e}")
            return {}

    def interpret_results(self, result: AnalysisResult) -> str:
        """
        Интерпретировать результаты анализа

        Args:
            result: Результат анализа

        Returns:
            str: Человекочитаемая интерпретация
        """
        data = result.data
        avg_tension = data.get('average_tension', 0)
        timeline = data.get('timeline', [])
        peak_count = data.get('peak_count', 0)

        # Эмодзи в зависимости от уровня
        if avg_tension > 7:
            tension_emoji = '🔥'
            tension_level = 'очень высокое'
        elif avg_tension > 5:
            tension_emoji = '⚡'
            tension_level = 'высокое'
        elif avg_tension > 3:
            tension_emoji = '📊'
            tension_level = 'умеренное'
        else:
            tension_emoji = '🌊'
            tension_level = 'низкое'

        lines = [
            f"🎭 **Анализ напряжения в произведении**\n",
            f"{tension_emoji} **Среднее напряжение**: {tension_level} ({avg_tension:.1f}/10)",
            f"📈 **Обнаружено пиков напряжения**: {peak_count}\n"
        ]

        # Timeline
        if timeline:
            lines.append(f"📊 **Динамика напряжения**:\n")

            for i, point in enumerate(timeline[:5], 1):
                position = point.get('position', 0) * 100
                score = point.get('score', 0)
                source = point.get('source', 'unknown')
                description = point.get('description', '')[:80]

                lines.append(
                    f"\n   {i}. На {position:.0f}% текста (напряжение: {score:.1f}/10)\n"
                    f"      Источник: {source}\n"
                    f"      {description}..."
                )

        return '\n'.join(lines)

    def _create_default_prompt(self) -> PromptTemplate:
        """Создать дефолтный шаблон промпта"""
        return PromptTemplate.create_default_tension_prompt()
