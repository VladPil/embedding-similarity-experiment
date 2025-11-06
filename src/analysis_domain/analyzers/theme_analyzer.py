"""
Анализатор тем произведения
"""
import json
import re
from typing import Dict, Any, List, Optional
from loguru import logger

from ..entities.base_analyzer import BaseAnalyzer
from ..entities.analysis_result import AnalysisResult
from ..entities.prompt_template import PromptTemplate
from src.text_domain.entities.base_text import BaseText
from src.common.types import AnalysisMode
from src.common.exceptions import AnalysisError


class ThemeAnalyzer(BaseAnalyzer):
    """
    Анализатор основных тем произведения

    Использует:
    - Семплирование чанков (каждый 10-й)
    - LLM для извлечения тем
    - Агрегацию повторяющихся тем

    Возвращает:
    - Список основных тем
    - Описание каждой темы
    - Примеры из текста
    - Частоту встречаемости
    """

    def __init__(
        self,
        llm_service: Optional[Any] = None,
        prompt_template: Optional[PromptTemplate] = None,
        max_themes: int = 5,
        sample_chunks: int = 10
    ):
        """
        Инициализация анализатора тем

        Args:
            llm_service: Сервис для работы с LLM
            prompt_template: Шаблон промпта
            max_themes: Максимальное количество тем
            sample_chunks: Количество чанков для анализа
        """
        self.llm_service = llm_service
        self.prompt_template = prompt_template or self._create_default_prompt()
        self.max_themes = max_themes
        self.sample_chunks = sample_chunks

    @property
    def requires_llm(self) -> bool:
        """Требуется LLM"""
        return True

    @property
    def requires_embeddings(self) -> bool:
        """Embeddings не обязательны"""
        return False

    async def analyze(
        self,
        text: BaseText,
        mode: AnalysisMode,
        **kwargs
    ) -> AnalysisResult:
        """
        Выполнить анализ тем

        Args:
            text: Текст для анализа
            mode: Режим анализа
            **kwargs: Дополнительные параметры

        Returns:
            AnalysisResult: Результат анализа
        """
        try:
            logger.info(f"Начало анализа тем: {text.title}")

            # Получаем чанки
            chunks = kwargs.get('chunks', [])
            if not chunks:
                raise AnalysisError(
                    "Анализ тем требует чанки текста",
                    details={"text_id": text.id}
                )

            # Семплируем чанки (каждый N-й)
            step = max(len(chunks) // self.sample_chunks, 1)
            sample_chunks = chunks[::step][:self.sample_chunks]

            logger.info(f"Анализируем {len(sample_chunks)} чанков для тем")

            # Собираем темы из семплов
            detected_themes = []

            for chunk in sample_chunks:
                try:
                    # Формируем промпт
                    prompt = self._format_theme_prompt(chunk.content)

                    # LLM запрос
                    result = await self.llm_service.generate(
                        prompt=prompt,
                        max_tokens=256,
                        temperature=0.5
                    )

                    # Парсим ответ
                    theme_data = self._parse_theme_response(result)

                    if theme_data and theme_data.get('theme'):
                        theme_data['position'] = chunk.metadata.get('position_ratio', 0.0)
                        theme_data['chunk_index'] = chunk.index
                        detected_themes.append(theme_data)

                except Exception as e:
                    logger.warning(f"Ошибка анализа темы в чанке {chunk.index}: {e}")
                    continue

            # Агрегируем темы
            themes = self._aggregate_themes(detected_themes)

            logger.info(f"Найдено {len(themes)} основных тем")

            return AnalysisResult(
                analyzer_type=self.__class__.__name__,
                text_id=text.id,
                mode=mode,
                data={
                    "themes": themes,
                    "total_themes": len(themes),
                    "chunks_analyzed": len(sample_chunks),
                    "theme_mentions": len(detected_themes)
                }
            )

        except Exception as e:
            logger.error(f"Ошибка анализа тем: {e}")
            raise AnalysisError(
                f"Не удалось проанализировать темы: {str(e)}",
                details={"text_id": text.id}
            )

    def _format_theme_prompt(self, chunk_text: str) -> str:
        """
        Форматировать промпт для анализа тем

        Args:
            chunk_text: Текст чанка

        Returns:
            str: Отформатированный промпт
        """
        prompt = f"""Проанализируй фрагмент текста и определи основную тему.

Фрагмент:
{chunk_text[:1200]}

Задача:
1. Определи ОДНУ основную тему из списка: любовь, дружба, предательство, власть, свобода, семья, одиночество, справедливость, война, смерть, надежда, страх, выбор, жертва, другое
2. Дай краткое описание (1-2 предложения), как эта тема раскрывается в фрагменте
3. Приведи короткую цитату-пример (до 100 символов)

Ответь СТРОГО в формате JSON:
{{
    "theme": "название темы",
    "description": "краткое описание",
    "example": "цитата из текста"
}}

JSON:"""

        return prompt

    def _parse_theme_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Распарсить ответ LLM

        Args:
            llm_response: Ответ LLM

        Returns:
            Dict: Данные темы
        """
        try:
            # Извлекаем JSON
            json_match = re.search(r'\{[^{}]*\}', llm_response, re.DOTALL)

            if json_match:
                parsed = json.loads(json_match.group(0))
                return parsed

            return {}

        except Exception as e:
            logger.warning(f"Ошибка парсинга ответа темы: {e}")
            return {}

    def _aggregate_themes(self, theme_mentions: List[Dict]) -> List[Dict]:
        """
        Агрегировать упоминания тем

        Объединяет одинаковые темы, считает частоту

        Args:
            theme_mentions: Список упоминаний тем

        Returns:
            List[Dict]: Список агрегированных тем
        """
        theme_map = {}

        for mention in theme_mentions:
            theme_name = mention.get('theme', '').strip().lower()
            if not theme_name or theme_name == "другое":
                continue

            if theme_name not in theme_map:
                theme_map[theme_name] = {
                    "theme": theme_name.capitalize(),
                    "descriptions": [],
                    "examples": [],
                    "positions": [],
                    "frequency": 0
                }

            theme = theme_map[theme_name]
            theme['frequency'] += 1

            # Добавляем описание
            desc = mention.get('description', '')
            if desc and desc not in theme['descriptions']:
                theme['descriptions'].append(desc)

            # Добавляем пример
            example = mention.get('example', '')
            if example and example not in theme['examples']:
                theme['examples'].append(example)

            # Добавляем позицию
            position = mention.get('position', 0.0)
            theme['positions'].append(position)

        # Конвертируем в список и сортируем по частоте
        themes = []
        for theme in theme_map.values():
            # Берём 2-3 лучших описания
            theme['descriptions'] = theme['descriptions'][:3]

            # Берём 2-3 примера
            theme['examples'] = theme['examples'][:3]

            # Средняя позиция
            if theme['positions']:
                theme['avg_position'] = sum(theme['positions']) / len(theme['positions'])
            else:
                theme['avg_position'] = 0.0

            # Убираем positions (не нужны в финальном выводе)
            del theme['positions']

            themes.append(theme)

        # Сортируем по частоте
        themes.sort(key=lambda x: x['frequency'], reverse=True)

        # Ограничиваем количество
        return themes[:self.max_themes]

    def interpret_results(self, result: AnalysisResult) -> str:
        """
        Интерпретировать результаты анализа

        Args:
            result: Результат анализа

        Returns:
            str: Человекочитаемая интерпретация
        """
        data = result.data
        themes = data.get('themes', [])
        total = data.get('total_themes', 0)

        if not themes:
            return "📖 Темы не обнаружены в тексте."

        lines = [
            f"📖 **Анализ тем произведения**\n",
            f"🔍 Обнаружено основных тем: {total}\n"
        ]

        for i, theme in enumerate(themes, 1):
            theme_name = theme.get('theme', 'Неизвестная')
            frequency = theme.get('frequency', 0)
            descriptions = theme.get('descriptions', [])
            examples = theme.get('examples', [])

            lines.append(f"**{i}. {theme_name}** (встречается {frequency} раз)")

            if descriptions:
                lines.append(f"   💬 {descriptions[0]}")

            if examples:
                lines.append(f"   📝 Пример: \"{examples[0][:100]}...\"")

            lines.append("")

        # Вывод
        main_themes = [t.get('theme', '') for t in themes[:3]]
        lines.append(
            f"💡 **Вывод**: Произведение исследует темы: "
            f"{', '.join(main_themes).lower()}."
        )

        return '\n'.join(lines)

    def _create_default_prompt(self) -> PromptTemplate:
        """Создать дефолтный шаблон промпта"""
        return PromptTemplate.create_default_theme_prompt()
