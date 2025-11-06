"""
Анализатор жанра текста на основе LLM
"""
import json
from typing import Optional, Dict, Any
from loguru import logger

from src.text_domain.entities.base_text import BaseText
from src.text_domain.entities.chunking_strategy import ChunkingStrategy
from ..entities.base_analyzer import BaseAnalyzer
from ..entities.analysis_result import AnalysisResult
from ..entities.prompt_template import PromptTemplate
from src.common.types import AnalysisMode
from src.common.exceptions import AnalysisError


class GenreAnalyzer(BaseAnalyzer):
    """
    Анализатор жанра текста на основе LLM

    Определяет литературный жанр текста используя языковую модель
    """

    def __init__(
        self,
        llm_service=None,
        prompt_template: Optional[PromptTemplate] = None
    ):
        """
        Инициализация анализатора

        Args:
            llm_service: Сервис для работы с LLM
            prompt_template: Шаблон промпта (опционально)
        """
        self.llm_service = llm_service
        self.prompt_template = prompt_template or PromptTemplate.create_default_genre_prompt()

    @property
    def name(self) -> str:
        return "genre"

    @property
    def display_name(self) -> str:
        return "Анализ жанра"

    @property
    def description(self) -> str:
        return """Определяет литературный жанр текста на основе его содержания,
стиля, тематики и структуры. Использует языковую модель для глубокого анализа."""

    @property
    def requires_llm(self) -> bool:
        return True

    @property
    def supports_chunked_mode(self) -> bool:
        # Для жанра лучше анализировать весь текст
        return False

    async def analyze(
        self,
        text: BaseText,
        mode: AnalysisMode = AnalysisMode.FULL_TEXT,
        chunking_strategy: Optional[ChunkingStrategy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Анализировать жанр текста

        Args:
            text: Текст для анализа
            mode: Режим анализа (только FULL_TEXT)
            chunking_strategy: Не используется
            context: Дополнительный контекст

        Returns:
            AnalysisResult: Результат анализа
        """
        try:
            import time
            start_time = time.time()

            # Валидация режима
            self.validate_mode(mode)

            # Получение контента
            content = await text.get_content()

            # Ограничиваем длину текста для анализа (первые 4000 символов)
            # Для определения жанра обычно достаточно начала
            analysis_text = content[:4000] if len(content) > 4000 else content

            # Формируем промпт
            user_prompt = self.prompt_template.format_user_prompt(text=analysis_text)

            # Вызов LLM (будет реализован позже в model_management)
            if self.llm_service:
                llm_response = await self._call_llm(
                    system_prompt=self.prompt_template.system_prompt,
                    user_prompt=user_prompt
                )
            else:
                # Заглушка для тестирования без LLM
                llm_response = self._get_mock_response()

            # Парсим ответ LLM
            result_data = await self._parse_llm_response(llm_response)

            # Добавляем метаданные
            result_data["text_length_analyzed"] = len(analysis_text)
            result_data["full_text_length"] = len(content)

            execution_time = (time.time() - start_time) * 1000

            # Создание результата
            result = AnalysisResult(
                text_id=text.id,
                analyzer_name=self.name,
                data=result_data,
                execution_time_ms=execution_time,
                mode=mode.value,
            )

            # Генерация интерпретации
            result.interpretation = self.interpret_results(result)

            logger.debug(
                f"Анализ жанра завершён для текста {text.id} "
                f"за {execution_time:.0f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Ошибка анализа жанра: {e}")
            raise AnalysisError(
                message=f"Genre analysis failed: {e}",
                details={"text_id": text.id, "error": str(e)}
            )

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Вызов LLM сервиса

        Args:
            system_prompt: Системный промпт
            user_prompt: Пользовательский промпт

        Returns:
            str: Ответ LLM
        """
        # TODO: Реализовать после создания model_management
        # return await self.llm_service.generate(
        #     system_prompt=system_prompt,
        #     user_prompt=user_prompt,
        #     temperature=self.prompt_template.temperature,
        #     max_tokens=self.prompt_template.max_tokens,
        # )
        return self._get_mock_response()

    def _get_mock_response(self) -> str:
        """
        Заглушка для тестирования без LLM

        Returns:
            str: Мок-ответ
        """
        return json.dumps({
            "main_genre": "художественная литература",
            "sub_genres": ["фантастика", "приключения"],
            "confidence": 0.85,
            "reasoning": "Текст содержит элементы фантастики и приключенческого жанра."
        }, ensure_ascii=False)

    async def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Парсинг ответа LLM

        Args:
            response: Ответ от LLM

        Returns:
            Dict: Распарсенные данные
        """
        try:
            # Пытаемся распарсить JSON
            data = json.loads(response)

            # Валидация обязательных полей
            if "main_genre" not in data:
                raise ValueError("Missing 'main_genre' in response")

            if "confidence" not in data:
                data["confidence"] = 0.5

            # Нормализация confidence
            data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))

            # Преобразование sub_genres в список если нужно
            if "sub_genres" not in data:
                data["sub_genres"] = []
            elif isinstance(data["sub_genres"], str):
                data["sub_genres"] = [data["sub_genres"]]

            return data

        except json.JSONDecodeError:
            # Если не JSON, пытаемся извлечь информацию из текста
            logger.warning("LLM response is not valid JSON, parsing as text")
            return {
                "main_genre": "неизвестно",
                "sub_genres": [],
                "confidence": 0.3,
                "reasoning": response[:200],
                "raw_response": response,
            }

    def interpret_results(self, result: AnalysisResult) -> str:
        """
        Интерпретировать результаты анализа жанра

        Args:
            result: Результат анализа

        Returns:
            str: Текстовая интерпретация
        """
        data = result.data

        main_genre = data.get("main_genre", "неизвестно")
        sub_genres = data.get("sub_genres", [])
        confidence = data.get("confidence", 0.0)
        reasoning = data.get("reasoning", "")

        lines = [f"🎭 Жанр: {main_genre}"]

        # Подзанры
        if sub_genres:
            sub_genres_str = ", ".join(sub_genres)
            lines.append(f"Подзанры: {sub_genres_str}")

        # Уверенность
        confidence_pct = confidence * 100
        confidence_level = "высокая" if confidence >= 0.7 else "средняя" if confidence >= 0.5 else "низкая"
        lines.append(f"Уверенность: {confidence_pct:.0f}% ({confidence_level})")

        # Обоснование
        if reasoning:
            lines.append(f"\nОбоснование: {reasoning}")

        return "\n".join(lines)

    def get_estimated_time(
        self,
        text_length: int,
        mode: AnalysisMode = AnalysisMode.FULL_TEXT
    ) -> float:
        """
        Оценка времени выполнения

        Args:
            text_length: Длина текста
            mode: Режим анализа

        Returns:
            float: Время в секундах
        """
        # LLM анализ занимает больше времени
        return 5.0 + min(text_length / 1000, 4.0) * 0.5
