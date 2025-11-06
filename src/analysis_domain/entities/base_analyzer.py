"""
Базовый класс для всех анализаторов текста
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from src.text_domain.entities.base_text import BaseText
from src.text_domain.entities.chunking_strategy import ChunkingStrategy
from .analysis_result import AnalysisResult
from src.common.types import AnalysisMode


class BaseAnalyzer(ABC):
    """
    Абстрактный базовый класс для всех анализаторов

    Определяет общий интерфейс для анализа текстов.
    Каждый анализатор реализует свою специфическую логику анализа.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Уникальное имя анализатора

        Returns:
            str: Имя анализатора (например, "genre", "style")
        """
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """
        Человекочитаемое название анализатора

        Returns:
            str: Название для UI (например, "Анализ жанра")
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Описание что делает анализатор

        Returns:
            str: Подробное описание
        """
        pass

    @property
    @abstractmethod
    def requires_llm(self) -> bool:
        """
        Требует ли анализатор LLM модель

        Returns:
            bool: True если нужна LLM
        """
        pass

    @property
    def requires_embeddings(self) -> bool:
        """
        Требует ли анализатор embeddings

        Returns:
            bool: True если нужны embeddings
        """
        return False

    @property
    def supports_chunked_mode(self) -> bool:
        """
        Поддерживает ли анализатор режим chunked

        Returns:
            bool: True если поддерживает
        """
        return True

    @abstractmethod
    async def analyze(
        self,
        text: BaseText,
        mode: AnalysisMode = AnalysisMode.FULL_TEXT,
        chunking_strategy: Optional[ChunkingStrategy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Анализировать текст

        Args:
            text: Текст для анализа
            mode: Режим анализа (FULL_TEXT или CHUNKED)
            chunking_strategy: Стратегия разбиения (для CHUNKED режима)
            context: Дополнительный контекст (embeddings, chunks, etc.)

        Returns:
            AnalysisResult: Результат анализа

        Raises:
            AnalysisError: Если анализ не удался
        """
        pass

    @abstractmethod
    def interpret_results(self, result: AnalysisResult) -> str:
        """
        Интерпретировать результаты анализа в человекочитаемый вид

        Это ОБЯЗАТЕЛЬНЫЙ метод для всех анализаторов!
        Должен возвращать понятное текстовое описание результатов.

        Args:
            result: Результат анализа

        Returns:
            str: Текстовая интерпретация
        """
        pass

    def get_estimated_time(
        self,
        text_length: int,
        mode: AnalysisMode = AnalysisMode.FULL_TEXT
    ) -> float:
        """
        Оценить время выполнения анализа в секундах

        Переопределяется в конкретных анализаторах для точной оценки.

        Args:
            text_length: Длина текста в символах
            mode: Режим анализа

        Returns:
            float: Оценочное время в секундах
        """
        if self.requires_llm:
            # LLM анализ обычно медленнее
            base_time = 5.0
            # +0.5 секунд на каждую 1000 символов
            time_per_chunk = (text_length / 1000) * 0.5
            return base_time + time_per_chunk
        else:
            # Быстрые анализаторы (TF-IDF, статистика)
            base_time = 0.5
            time_per_chunk = (text_length / 10000) * 0.1
            return base_time + time_per_chunk

    def validate_mode(self, mode: AnalysisMode) -> bool:
        """
        Валидировать режим анализа

        Args:
            mode: Режим анализа

        Returns:
            bool: True если режим поддерживается

        Raises:
            ValueError: Если режим не поддерживается
        """
        if mode == AnalysisMode.CHUNKED and not self.supports_chunked_mode:
            raise ValueError(
                f"Analyzer '{self.name}' does not support CHUNKED mode"
            )
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Сериализация информации об анализаторе

        Returns:
            Dict: Информация об анализаторе
        """
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "requires_llm": self.requires_llm,
            "requires_embeddings": self.requires_embeddings,
            "supports_chunked_mode": self.supports_chunked_mode,
        }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()
