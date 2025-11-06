"""
Анализатор персонажей
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


class CharacterAnalyzer(BaseAnalyzer):
    """
    Анализатор персонажей произведения

    Использует:
    - ChunkIndexer для отбора релевантных чанков (диалоги, имена)
    - LLM для извлечения персонажей и их черт
    - Агрегацию упоминаний персонажей

    Возвращает:
    - Список персонажей с ролями (main/secondary/episodic)
    - Черты характера
    - Timeline появлений
    - Development timeline
    """

    def __init__(
        self,
        llm_service: Optional[Any] = None,
        prompt_template: Optional[PromptTemplate] = None,
        max_chunks_to_analyze: int = 30
    ):
        """
        Инициализация анализатора персонажей

        Args:
            llm_service: Сервис для работы с LLM
            prompt_template: Шаблон промпта
            max_chunks_to_analyze: Максимальное количество чанков для анализа
        """
        self.llm_service = llm_service
        self.prompt_template = prompt_template or self._create_default_prompt()
        self.max_chunks_to_analyze = max_chunks_to_analyze
        self.indexer = ChunkIndexer()

    @property
    def requires_llm(self) -> bool:
        """Требуется LLM"""
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
        Выполнить анализ персонажей

        Args:
            text: Текст для анализа
            mode: Режим анализа
            **kwargs: Дополнительные параметры

        Returns:
            AnalysisResult: Результат анализа
        """
        try:
            logger.info(f"Начало анализа персонажей: {text.title}")

            # Получаем чанки
            chunks = kwargs.get('chunks', [])
            if not chunks:
                raise AnalysisError(
                    "Анализ персонажей требует чанки текста",
                    details={"text_id": text.id}
                )

            # Строим индекс персонажных чанков
            char_index = self.indexer.build_character_index(chunks)

            logger.info(
                f"Индекс персонажей: {len(char_index.chunk_indices)} чанков "
                f"({char_index.coverage*100:.1f}% охват)"
            )

            # Получаем релевантные чанки
            relevant_chunks = self.indexer.get_chunk_subset(chunks, 'characters')

            if not relevant_chunks:
                logger.warning("Не найдено персонажных чанков")
                return AnalysisResult(
                    analyzer_type=self.__class__.__name__,
                    text_id=text.id,
                    mode=mode,
                    data={
                        "characters": [],
                        "total_characters": 0,
                        "chunks_analyzed": 0,
                        "coverage": 0.0
                    }
                )

            # Анализируем чанки с LLM
            character_mentions = await self._analyze_chunks_with_llm(relevant_chunks)

            # Агрегируем персонажей
            characters = self._aggregate_characters(character_mentions)

            logger.info(f"Найдено {len(characters)} персонажей")

            return AnalysisResult(
                analyzer_type=self.__class__.__name__,
                text_id=text.id,
                mode=mode,
                data={
                    "characters": characters,
                    "total_characters": len(characters),
                    "chunks_analyzed": min(len(relevant_chunks), self.max_chunks_to_analyze),
                    "coverage": char_index.coverage,
                    "character_mentions": len(character_mentions)
                }
            )

        except Exception as e:
            logger.error(f"Ошибка анализа персонажей: {e}")
            raise AnalysisError(
                f"Не удалось проанализировать персонажей: {str(e)}",
                details={"text_id": text.id}
            )

    async def _analyze_chunks_with_llm(
        self,
        chunks: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Анализировать чанки с помощью LLM

        Args:
            chunks: Список чанков

        Returns:
            List[Dict]: Список упоминаний персонажей
        """
        character_mentions = []

        # Ограничиваем количество чанков
        chunks_to_process = chunks[:self.max_chunks_to_analyze]

        for i, chunk in enumerate(chunks_to_process):
            try:
                # Контекст (предыдущий чанк)
                context_text = ""
                if i > 0:
                    context_text = chunks_to_process[i-1].content[:500]

                # Формируем промпт
                prompt = self._format_character_prompt(chunk.content, context_text)

                # LLM запрос
                result = await self.llm_service.generate(
                    prompt=prompt,
                    max_tokens=512,
                    temperature=0.3
                )

                # Парсим ответ
                char_data = self._parse_character_response(result)

                if char_data and char_data.get('name'):
                    char_data['chunk_index'] = chunk.index
                    char_data['position'] = chunk.metadata.get('position_ratio', 0.0)
                    character_mentions.append(char_data)

            except Exception as e:
                logger.warning(f"Ошибка анализа чанка {chunk.index}: {e}")
                continue

        return character_mentions

    def _format_character_prompt(self, chunk_text: str, context: str = "") -> str:
        """
        Форматировать промпт для анализа персонажей

        Args:
            chunk_text: Текст чанка
            context: Контекст (предыдущий чанк)

        Returns:
            str: Отформатированный промпт
        """
        prompt = f"""Проанализируй фрагмент текста и извлеки информацию о персонажах.

Контекст (предыдущий фрагмент):
{context if context else "Нет контекста"}

Фрагмент для анализа:
{chunk_text[:1500]}

Задача:
1. Найди ОДНОГО наиболее важного персонажа в этом фрагменте
2. Определи его роль: main (главный), secondary (второстепенный), episodic (эпизодический)
3. Извлеки 2-3 черты характера с доказательствами из текста

Ответь СТРОГО в формате JSON:
{{
    "name": "Имя Персонажа",
    "role": "main|secondary|episodic",
    "traits": [
        {{"trait": "черта характера", "evidence": "цитата из текста"}},
        ...
    ]
}}

JSON:"""

        return prompt

    def _parse_character_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Распарсить ответ LLM

        Args:
            llm_response: Ответ LLM

        Returns:
            Dict: Данные персонажа
        """
        try:
            # Извлекаем JSON
            json_match = re.search(
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
                llm_response,
                re.DOTALL
            )

            if json_match:
                parsed = json.loads(json_match.group(0))
                return parsed

            return {}

        except Exception as e:
            logger.warning(f"Ошибка парсинга ответа персонажа: {e}")
            return {}

    def _aggregate_characters(self, mentions: List[Dict]) -> List[Dict]:
        """
        Агрегировать упоминания персонажей

        Объединяет несколько упоминаний одного персонажа в единый профиль

        Args:
            mentions: Список упоминаний

        Returns:
            List[Dict]: Список агрегированных персонажей
        """
        character_map = {}

        for mention in mentions:
            name = mention.get('name', '').strip()
            if not name or name in ["Имя Персонажа", "Неизвестный", "Unknown"]:
                continue

            # Нормализуем имя
            name_key = name.lower()

            if name_key not in character_map:
                character_map[name_key] = {
                    "name": name,
                    "traits": [],
                    "role": mention.get('role', 'episodic'),
                    "appearances": [],
                    "first_appearance": mention.get('position', 0.0)
                }

            char = character_map[name_key]

            # Добавляем черты
            if 'traits' in mention:
                for trait in mention['traits']:
                    if isinstance(trait, dict):
                        char['traits'].append(trait)
                    else:
                        char['traits'].append({"trait": str(trait), "evidence": ""})

            # Добавляем появление
            char['appearances'].append({
                "position": mention.get('position', 0.0),
                "chunk_index": mention.get('chunk_index', 0)
            })

            # Обновляем роль (main > secondary > episodic)
            if mention.get('role') == 'main':
                char['role'] = 'main'
            elif mention.get('role') == 'secondary' and char['role'] != 'main':
                char['role'] = 'secondary'

        # Конвертируем в список
        characters = []
        for char in character_map.values():
            # Дедупликация черт
            unique_traits = {}
            for trait_dict in char['traits']:
                trait_name = trait_dict.get('trait', '').lower()
                if trait_name and trait_name not in unique_traits:
                    unique_traits[trait_name] = trait_dict

            char['traits'] = list(unique_traits.values())[:5]  # Топ-5 черт

            # Сортируем появления
            char['appearances'].sort(key=lambda x: x['position'])

            # Timeline развития
            if len(char['appearances']) > 1:
                char['development_timeline'] = [
                    {
                        "position": char['first_appearance'],
                        "description": "Первое появление"
                    },
                    {
                        "position": char['appearances'][-1]['position'],
                        "description": "Последнее появление"
                    }
                ]
            else:
                char['development_timeline'] = []

            characters.append(char)

        # Сортируем по важности
        role_order = {'main': 0, 'secondary': 1, 'episodic': 2}
        characters.sort(key=lambda x: (
            role_order.get(x['role'], 3),
            -len(x['appearances'])
        ))

        return characters

    def interpret_results(self, result: AnalysisResult) -> str:
        """
        Интерпретировать результаты анализа

        Args:
            result: Результат анализа

        Returns:
            str: Человекочитаемая интерпретация
        """
        data = result.data
        characters = data.get('characters', [])
        total = data.get('total_characters', 0)
        coverage = data.get('coverage', 0)

        if not characters:
            return "👥 Персонажи не обнаружены в тексте."

        # Группируем по ролям
        main_chars = [c for c in characters if c.get('role') == 'main']
        secondary_chars = [c for c in characters if c.get('role') == 'secondary']
        episodic_chars = [c for c in characters if c.get('role') == 'episodic']

        lines = [
            f"👥 **Анализ персонажей произведения**\n",
            f"📊 Обнаружено персонажей: {total} (охват: {coverage*100:.1f}%)\n"
        ]

        # Главные персонажи
        if main_chars:
            lines.append(f"⭐ **Главные персонажи** ({len(main_chars)}):")
            for char in main_chars[:5]:
                name = char.get('name', 'Неизвестный')
                traits = char.get('traits', [])
                appearances = len(char.get('appearances', []))

                trait_names = [t.get('trait', '') for t in traits[:3]]
                trait_text = ', '.join(trait_names) if trait_names else 'нет данных'

                lines.append(
                    f"\n   **{name}**\n"
                    f"   • Появлений: {appearances}\n"
                    f"   • Черты: {trait_text}"
                )

        # Второстепенные
        if secondary_chars:
            lines.append(f"\n👤 **Второстепенные персонажи** ({len(secondary_chars)}):")
            for char in secondary_chars[:3]:
                name = char.get('name', 'Неизвестный')
                appearances = len(char.get('appearances', []))
                lines.append(f"   • {name} ({appearances} появлений)")

        # Эпизодические
        if episodic_chars:
            lines.append(f"\n👥 **Эпизодические персонажи**: {len(episodic_chars)}")

        return '\n'.join(lines)

    def _create_default_prompt(self) -> PromptTemplate:
        """Создать дефолтный шаблон промпта"""
        return PromptTemplate.create_default_character_prompt()
