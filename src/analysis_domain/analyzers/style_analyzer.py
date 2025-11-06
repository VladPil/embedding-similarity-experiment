"""
Анализатор стиля текста
"""
import json
import re
from typing import Optional, Dict, Any
from loguru import logger

from src.text_domain.entities.base_text import BaseText
from src.text_domain.entities.chunking_strategy import ChunkingStrategy
from src.text_domain.services.chunking_service import ChunkingService
from ..entities.base_analyzer import BaseAnalyzer
from ..entities.analysis_result import AnalysisResult
from ..entities.prompt_template import PromptTemplate
from src.common.types import AnalysisMode
from src.common.exceptions import AnalysisError


class StyleAnalyzer(BaseAnalyzer):
    """
    Анализатор стиля текста

    Комбинирует статистический анализ и LLM для определения
    стилистических особенностей текста
    """

    def __init__(
        self,
        llm_service=None,
        prompt_template: Optional[PromptTemplate] = None
    ):
        """Инициализация анализатора"""
        self.llm_service = llm_service
        self.prompt_template = prompt_template or PromptTemplate.create_default_style_prompt()
        self.chunking_service = ChunkingService()

    @property
    def name(self) -> str:
        return "style"

    @property
    def display_name(self) -> str:
        return "Анализ стиля"

    @property
    def description(self) -> str:
        return """Анализирует стилистические особенности текста: лексику,
синтаксис, образность, ритм, эмоциональность."""

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
        """Анализировать стиль текста"""
        try:
            import time
            start_time = time.time()

            self.validate_mode(mode)
            content = await text.get_content()

            # Статистический анализ
            stats = await self._statistical_analysis(content)

            # LLM анализ
            if mode == AnalysisMode.FULL_TEXT:
                llm_data = await self._llm_analysis_full(content)
            else:
                if not chunking_strategy:
                    raise AnalysisError("Chunking strategy required for CHUNKED mode")
                llm_data = await self._llm_analysis_chunked(content, chunking_strategy)

            # Объединяем результаты
            result_data = {
                "statistics": stats,
                "llm_assessment": llm_data,
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

            logger.debug(f"Анализ стиля завершён для текста {text.id}")
            return result

        except Exception as e:
            logger.error(f"Ошибка анализа стиля: {e}")
            raise AnalysisError(
                message=f"Style analysis failed: {e}",
                details={"text_id": text.id, "error": str(e)}
            )

    async def _statistical_analysis(self, content: str) -> Dict[str, Any]:
        """
        Статистический анализ стиля

        Args:
            content: Текст

        Returns:
            Dict: Статистические метрики
        """
        # Предложения
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Слова
        words = re.findall(r'\b\w+\b', content.lower())

        # Параграфы
        paragraphs = content.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Метрики
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        avg_paragraph_length = len(content) / len(paragraphs) if paragraphs else 0

        # Сложные слова (>7 символов)
        complex_words = [w for w in words if len(w) > 7]
        complex_word_ratio = len(complex_words) / len(words) if words else 0

        # Уникальные слова
        unique_words = set(words)
        vocabulary_richness = len(unique_words) / len(words) if words else 0

        return {
            "total_words": len(words),
            "unique_words": len(unique_words),
            "total_sentences": len(sentences),
            "total_paragraphs": len(paragraphs),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "avg_word_length": round(avg_word_length, 2),
            "avg_paragraph_length": round(avg_paragraph_length, 2),
            "complex_word_ratio": round(complex_word_ratio, 3),
            "vocabulary_richness": round(vocabulary_richness, 3),
        }

    async def _llm_analysis_full(self, content: str) -> Dict[str, Any]:
        """LLM анализ полного текста"""
        # Ограничиваем длину
        analysis_text = content[:3000]

        user_prompt = self.prompt_template.format_user_prompt(text=analysis_text)

        if self.llm_service:
            response = await self._call_llm(user_prompt)
        else:
            response = self._get_mock_style_response()

        return await self._parse_style_response(response)

    async def _llm_analysis_chunked(
        self,
        content: str,
        strategy: ChunkingStrategy
    ) -> Dict[str, Any]:
        """LLM анализ по чанкам"""
        chunks = await self.chunking_service.chunk_text(content, strategy)

        chunk_results = []
        for chunk in chunks[:3]:  # Анализируем первые 3 чанка
            response = await self._llm_analysis_full(chunk.content)
            chunk_results.append({
                "chunk_index": chunk.chunk_index,
                "analysis": response
            })

        # Агрегируем результаты
        # TODO: Более сложная агрегация
        return {
            "chunk_count": len(chunks),
            "analyzed_chunks": len(chunk_results),
            "chunk_results": chunk_results,
        }

    async def _call_llm(self, user_prompt: str) -> str:
        """Вызов LLM"""
        return self._get_mock_style_response()

    def _get_mock_style_response(self) -> str:
        """Обогащенный мок-ответ для стиля с детальным анализом"""

        # Обогащенные стилистические характеристики
        style_characteristics = {
            "lexical_complexity": "средняя",
            "sentence_length": "средняя",
            "imagery": "средняя",
            "emotionality": "низкая",
            "formality": "нейтральная",

            # Дополнительные стилистические аспекты
            "narrative_perspective": "третье лицо",  # первое/второе/третье/смешанное
            "temporal_structure": "линейная",  # линейная/нелинейная/циклическая
            "dialogue_density": "умеренная",  # низкая/умеренная/высокая
            "descriptive_detail": "средняя",  # минимальная/средняя/богатая
            "metaphorical_language": "умеренная",  # редкая/умеренная/обильная
            "syntax_variety": "разнообразная",  # простая/разнообразная/сложная
            "rhythm_pattern": "переменный",  # монотонный/переменный/динамичный
            "tone_consistency": "стабильная",  # стабильная/переменная/контрастная

            # Жанровые особенности
            "literary_devices": [
                "эпитеты", "сравнения", "метафоры", "олицетворение", "гипербола"
            ],
            "style_markers": [
                "авторские отступления", "внутренние монологи", "описания природы"
            ],

            # Характеристики языка
            "register_level": "нейтральный",  # разговорный/нейтральный/книжный
            "archaic_elements": "отсутствуют",  # отсутствуют/редкие/заметные
            "colloquial_expressions": "умеренные",  # редкие/умеренные/частые
            "technical_vocabulary": "минимальная",  # минимальная/умеренная/специализированная

            # Композиционные особенности
            "paragraph_structure": "традиционная",  # традиционная/экспериментальная
            "chapter_organization": "хронологическая",  # хронологическая/тематическая/ассоциативная
            "pacing_variation": "равномерная"  # равномерная/контрастная/градуальная
        }

        return json.dumps(style_characteristics, ensure_ascii=False)

    async def _parse_style_response(self, response: str) -> Dict[str, Any]:
        """Парсинг ответа LLM"""
        try:
            return json.loads(response)
        except:
            return {"raw_response": response}

    def interpret_results(self, result: AnalysisResult) -> str:
        """Интерпретация результатов"""
        data = result.data
        stats = data.get("statistics", {})
        llm = data.get("llm_assessment", {})

        lines = ["📝 Стилистический анализ:\n"]

        # Статистика
        lines.append("Статистика:")
        lines.append(f"- Слов: {stats.get('total_words', 0)}")
        lines.append(f"- Уникальных слов: {stats.get('unique_words', 0)}")
        lines.append(f"- Предложений: {stats.get('total_sentences', 0)}")
        lines.append(f"- Среднняя длина предложения: {stats.get('avg_sentence_length', 0)} слов")
        lines.append(f"- Богатство словаря: {stats.get('vocabulary_richness', 0):.2%}")

        # LLM оценка
        if isinstance(llm, dict) and "lexical_complexity" in llm:
            lines.append("\nОценка LLM:")
            lines.append(f"- Сложность лексики: {llm.get('lexical_complexity', 'н/д')}")
            lines.append(f"- Длина предложений: {llm.get('sentence_length', 'н/д')}")
            lines.append(f"- Образность: {llm.get('imagery', 'н/д')}")
            lines.append(f"- Эмоциональность: {llm.get('emotionality', 'н/д')}")

        return "\n".join(lines)

    def get_estimated_time(
        self,
        text_length: int,
        mode: AnalysisMode = AnalysisMode.FULL_TEXT
    ) -> float:
        """Оценка времени"""
        return 3.0 + (text_length / 1000) * 0.3
