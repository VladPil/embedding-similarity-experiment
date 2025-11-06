"""
Базовый класс для всех компараторов (методов сравнения)
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from src.text_domain.entities.base_text import BaseText
from .analysis_result import AnalysisResult
from .comparison_result import ComparisonResult
from .comparison_matrix import ComparisonMatrix
from src.common.types import AnalysisMode, ChunkedComparisonStrategy


class BaseComparator(ABC):
    """
    Абстрактный базовый класс для всех компараторов

    Компаратор сравнивает два текста и вычисляет их сходство
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Уникальное имя компаратора

        Returns:
            str: Имя (например, "cosine", "semantic")
        """
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """
        Человекочитаемое название компаратора

        Returns:
            str: Название для UI (например, "Косинусное сходство")
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Описание метода сравнения

        Returns:
            str: Подробное описание
        """
        pass

    @property
    def supports_chunked_mode(self) -> bool:
        """
        Поддерживает ли компаратор chunked режим

        Returns:
            bool: True если поддерживает
        """
        return True

    @property
    def requires_embeddings(self) -> bool:
        """
        Требует ли компаратор embeddings

        Returns:
            bool: True если нужны embeddings
        """
        return True

    @property
    def requires_llm(self) -> bool:
        """
        Требует ли компаратор LLM

        Returns:
            bool: True если нужна LLM
        """
        return False

    @abstractmethod
    async def compare(
        self,
        text1: BaseText,
        text2: BaseText,
        analysis_results: Dict[str, AnalysisResult],
        mode: AnalysisMode = AnalysisMode.FULL_TEXT,
        strategy: ChunkedComparisonStrategy = ChunkedComparisonStrategy.AGGREGATE_FIRST,
        context: Optional[Dict] = None
    ) -> ComparisonResult:
        """
        Сравнить два текста

        Args:
            text1: Первый текст
            text2: Второй текст
            analysis_results: Результаты анализов текстов (для контекста)
            mode: Режим сравнения
            strategy: Стратегия сравнения в chunked режиме
            context: Дополнительный контекст (embeddings, chunks, etc.)

        Returns:
            ComparisonResult: Результат сравнения

        Raises:
            AnalysisError: Если сравнение не удалось
        """
        pass

    async def compare_all(
        self,
        texts: List[BaseText],
        analysis_results: Dict[str, AnalysisResult],
        mode: AnalysisMode = AnalysisMode.FULL_TEXT,
        strategy: ChunkedComparisonStrategy = ChunkedComparisonStrategy.AGGREGATE_FIRST,
        include_self: bool = False,
        context: Optional[Dict] = None
    ) -> ComparisonMatrix:
        """
        Сравнить все пары текстов и построить матрицу N x N

        Args:
            texts: Список текстов
            analysis_results: Результаты анализов
            mode: Режим сравнения
            strategy: Стратегия в chunked режиме
            include_self: Включать ли сравнение текста с самим собой
            context: Дополнительный контекст

        Returns:
            ComparisonMatrix: Матрица сравнений
        """
        n = len(texts)
        text_ids = [text.id for text in texts]
        matrix = ComparisonMatrix(size=n, text_ids=text_ids)

        # Сравниваем все пары
        for i in range(n):
            for j in range(i if include_self else i + 1, n):
                result = await self.compare(
                    texts[i],
                    texts[j],
                    analysis_results,
                    mode,
                    strategy,
                    context
                )

                # Устанавливаем результат для обоих направлений
                matrix.set(i, j, result)

                if not include_self and i != j:
                    # Симметричная матрица
                    matrix.set(j, i, result)

        return matrix

    def validate_mode(self, mode: AnalysisMode) -> bool:
        """
        Валидировать режим сравнения

        Args:
            mode: Режим

        Returns:
            bool: True если режим поддерживается

        Raises:
            ValueError: Если режим не поддерживается
        """
        if mode == AnalysisMode.CHUNKED and not self.supports_chunked_mode:
            raise ValueError(
                f"Comparator '{self.name}' does not support CHUNKED mode"
            )
        return True

    def get_estimated_time(
        self,
        text1_length: int,
        text2_length: int,
        mode: AnalysisMode = AnalysisMode.FULL_TEXT
    ) -> float:
        """
        Оценить время сравнения в секундах

        Args:
            text1_length: Длина первого текста
            text2_length: Длина второго текста
            mode: Режим

        Returns:
            float: Оценочное время в секундах
        """
        if self.requires_llm:
            # LLM сравнение медленное
            return 10.0 + ((text1_length + text2_length) / 2000) * 1.0
        else:
            # Векторное сравнение быстрое
            return 0.1 + ((text1_length + text2_length) / 100000) * 0.05

    def to_dict(self) -> Dict:
        """
        Сериализация информации о компараторе

        Returns:
            Dict: Информация о компараторе
        """
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "supports_chunked_mode": self.supports_chunked_mode,
            "requires_embeddings": self.requires_embeddings,
            "requires_llm": self.requires_llm,
        }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()
