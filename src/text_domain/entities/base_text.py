"""
Базовый класс для всех текстовых сущностей
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

from src.common.types import TextID, Metadata
from src.common.utils import now_utc


@dataclass
class BaseText(ABC):
    """
    Абстрактный базовый класс для любого текста

    Определяет общий интерфейс для работы с текстами различных типов
    """

    # Основные поля
    id: TextID
    title: str
    language: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)

    # Временные метки
    created_at: datetime = field(default_factory=now_utc)
    updated_at: datetime = field(default_factory=now_utc)

    # Кэш контента (ленивая загрузка)
    _content: Optional[str] = field(default=None, repr=False, init=False)
    _length: Optional[int] = field(default=None, repr=False, init=False)

    @abstractmethod
    async def get_content(self) -> str:
        """
        Получить полный текст (может загружаться лениво из БД или файла)

        Returns:
            str: Содержимое текста
        """
        pass

    @abstractmethod
    def get_text_type(self) -> str:
        """
        Получить тип текста

        Returns:
            str: 'plain' или 'fb2'
        """
        pass

    @abstractmethod
    def get_metadata_schema(self) -> Dict[str, str]:
        """
        Получить схему метаданных для этого типа текста

        Returns:
            Dict[str, str]: Схема метаданных (имя поля -> тип)
        """
        pass

    def get_length(self) -> int:
        """
        Получить длину текста в символах

        Returns:
            int: Количество символов
        """
        if self._length is not None:
            return self._length

        if self._content is not None:
            self._length = len(self._content)
            return self._length

        return 0

    def get_word_count(self) -> int:
        """
        Подсчитать количество слов в тексте

        Returns:
            int: Количество слов
        """
        if self._content:
            return len(self._content.split())
        return 0

    def get_line_count(self) -> int:
        """
        Подсчитать количество строк в тексте

        Returns:
            int: Количество строк
        """
        if self._content:
            return len(self._content.splitlines())
        return 0

    def clear_cache(self) -> None:
        """Очистить кэшированный контент"""
        self._content = None
        self._length = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Сериализация в словарь

        Returns:
            Dict[str, Any]: Данные текста
        """
        return {
            "id": self.id,
            "text_type": self.get_text_type(),
            "title": self.title,
            "language": self.language,
            "metadata": self.metadata,
            "length": self.get_length(),
            "word_count": self.get_word_count() if self._content else None,
            "line_count": self.get_line_count() if self._content else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, title={self.title!r})"

    def __repr__(self) -> str:
        return self.__str__()
