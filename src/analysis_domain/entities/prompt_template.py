"""
Шаблон промпта для LLM
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

from src.common.types import Metadata
from src.common.utils import now_utc


@dataclass
class PromptTemplate:
    """
    Шаблон промпта для LLM анализаторов

    Содержит системный промпт и шаблон пользовательского промпта
    с плейсхолдерами для подстановки данных
    """

    # Основные поля
    id: str
    name: str
    analyzer_type: str  # К какому анализатору относится

    # Промпты
    system_prompt: str  # Системный промпт (инструкции для LLM)
    user_prompt_template: str  # Шаблон пользовательского промпта с {плейсхолдерами}

    # Параметры генерации
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # JSON schema для structured output (опционально)
    output_schema: Optional[Dict[str, Any]] = None

    # Флаги
    is_default: bool = False

    # Метаданные
    metadata: Metadata = field(default_factory=dict)

    # Временные метки
    created_at: datetime = field(default_factory=now_utc)
    updated_at: datetime = field(default_factory=now_utc)

    def format_user_prompt(self, **kwargs) -> str:
        """
        Форматировать пользовательский промпт с подстановкой значений

        Args:
            **kwargs: Значения для подстановки в плейсхолдеры

        Returns:
            str: Отформатированный промпт

        Example:
            >>> template.user_prompt_template = "Проанализируй текст: {text}"
            >>> template.format_user_prompt(text="Пример текста")
            "Проанализируй текст: Пример текста"
        """
        try:
            return self.user_prompt_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(
                f"Missing placeholder value: {e}. "
                f"Available placeholders: {self.get_placeholders()}"
            )

    def get_placeholders(self) -> list:
        """
        Получить список плейсхолдеров в шаблоне

        Returns:
            list: Список имён плейсхолдеров

        Example:
            >>> template.user_prompt_template = "Текст: {text}, Жанр: {genre}"
            >>> template.get_placeholders()
            ['text', 'genre']
        """
        import re
        pattern = r'\{(\w+)\}'
        return re.findall(pattern, self.user_prompt_template)

    def validate(self) -> bool:
        """
        Валидировать шаблон

        Returns:
            bool: True если валиден

        Raises:
            ValueError: Если шаблон невалиден
        """
        # Проверка параметров генерации
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )

        if self.max_tokens <= 0:
            raise ValueError(
                f"max_tokens must be positive, got {self.max_tokens}"
            )

        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(
                f"top_p must be between 0.0 and 1.0, got {self.top_p}"
            )

        # Проверка наличия промптов
        if not self.system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")

        if not self.user_prompt_template.strip():
            raise ValueError("user_prompt_template cannot be empty")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Сериализация в словарь

        Returns:
            Dict: Данные шаблона
        """
        return {
            "id": self.id,
            "name": self.name,
            "analyzer_type": self.analyzer_type,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "placeholders": self.get_placeholders(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "output_schema": self.output_schema,
            "is_default": self.is_default,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def create_default_genre_prompt(cls) -> "PromptTemplate":
        """
        Создать шаблон по умолчанию для анализа жанра

        Returns:
            PromptTemplate: Шаблон для анализа жанра
        """
        return cls(
            id="default_genre",
            name="Анализ жанра (по умолчанию)",
            analyzer_type="genre",
            system_prompt="""Ты - эксперт по литературным жанрам.
Твоя задача - определить жанр текста на основе его содержания, стиля и тематики.
Отвечай в формате JSON с полями: main_genre, sub_genres (список), confidence (0-1), reasoning.""",
            user_prompt_template="""Определи жанр следующего текста:

{text}

Верни результат в формате JSON.""",
            temperature=0.3,  # Низкая температура для более точных результатов
            max_tokens=500,
            output_schema={
                "type": "object",
                "properties": {
                    "main_genre": {"type": "string"},
                    "sub_genres": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                    "reasoning": {"type": "string"}
                },
                "required": ["main_genre", "confidence"]
            },
            is_default=True,
        )

    @classmethod
    def create_default_style_prompt(cls) -> "PromptTemplate":
        """
        Создать шаблон по умолчанию для анализа стиля

        Returns:
            PromptTemplate: Шаблон для анализа стиля
        """
        return cls(
            id="default_style",
            name="Анализ стиля (по умолчанию)",
            analyzer_type="style",
            system_prompt="""Ты - эксперт по литературному стилю.
Проанализируй стилистические особенности текста: лексику, синтаксис, образность, ритм.
Отвечай в формате JSON.""",
            user_prompt_template="""Проанализируй стиль следующего текста:

{text}

Оцени:
- Сложность лексики (простая/средняя/сложная)
- Длина предложений (короткие/средние/длинные)
- Образность языка (низкая/средняя/высокая)
- Эмоциональность (низкая/средняя/высокая)

Верни результат в формате JSON.""",
            temperature=0.5,
            max_tokens=800,
            is_default=True,
        )

    @classmethod
    def create_default_theme_prompt(cls) -> "PromptTemplate":
        """
        Создать шаблон по умолчанию для анализа тем
        """
        return cls(
            id="default_theme",
            name="Анализ тем (по умолчанию)",
            analyzer_type="theme",
            system_prompt="""Ты - эксперт по литературному анализу.
Твоя задача - найти основные темы в тексте.
Отвечай в формате JSON со списком тем и их описаниями.""",
            user_prompt_template="""Найди основные темы в следующем тексте:

{text}

Верни результат в формате JSON со списком тем.""",
            temperature=0.5,
            max_tokens=800,
            is_default=True,
        )

    @classmethod
    def create_default_tension_prompt(cls) -> "PromptTemplate":
        """
        Создать шаблон по умолчанию для анализа напряжения
        """
        return cls(
            id="default_tension",
            name="Анализ напряжения (по умолчанию)",
            analyzer_type="tension",
            system_prompt="""Ты - эксперт по анализу драматического напряжения в тексте.
Оцени уровень напряжения от 0 до 10 и опиши источники напряжения.
Отвечай в формате JSON.""",
            user_prompt_template="""Оцени уровень напряжения в следующем тексте:

{text}

Верни результат в формате JSON с уровнем напряжения (0-10) и описанием.""",
            temperature=0.4,
            max_tokens=600,
            is_default=True,
        )

    @classmethod
    def create_default_character_prompt(cls) -> "PromptTemplate":
        """
        Создать шаблон по умолчанию для анализа персонажей
        """
        return cls(
            id="default_character",
            name="Анализ персонажей (по умолчанию)",
            analyzer_type="character",
            system_prompt="""Ты - эксперт по анализу литературных персонажей.
Найди персонажей в тексте и опиши их роли и характеристики.
Отвечай в формате JSON.""",
            user_prompt_template="""Найди персонажей в следующем тексте:

{text}

Для каждого персонажа укажи имя, роль и краткое описание.
Верни результат в формате JSON.""",
            temperature=0.4,
            max_tokens=800,
            is_default=True,
        )

    def __str__(self) -> str:
        return f"PromptTemplate(name={self.name!r}, analyzer={self.analyzer_type})"

    def __repr__(self) -> str:
        return self.__str__()
