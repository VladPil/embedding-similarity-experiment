"""
Сущность сессии анализа - главный оркестратор
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.text_domain.entities.base_text import BaseText
from .base_analyzer import BaseAnalyzer
from .base_comparator import BaseComparator
from .analysis_result import AnalysisResult
from .comparison_matrix import ComparisonMatrix
from src.common.types import (
    SessionStatus,
    AnalysisMode,
    ChunkedComparisonStrategy,
    Metadata
)
from src.common.utils import now_utc, calculate_duration
from src.common.exceptions import ValidationError


@dataclass
class AnalysisSession:
    """
    Сессия анализа - главная сущность для управления анализом текстов

    Объединяет тексты, анализаторы, компаратор и результаты.
    Представляет собой полный процесс анализа от начала до конца.
    """

    # Основные поля
    id: str
    name: str

    # Тексты для анализа (от 1 до 5)
    texts: List[BaseText] = field(default_factory=list)

    # Анализаторы для применения
    analyzers: List[BaseAnalyzer] = field(default_factory=list)

    # Компаратор (метод сравнения)
    comparator: Optional[BaseComparator] = None

    # Режим анализа
    mode: AnalysisMode = AnalysisMode.FULL_TEXT
    chunking_strategy_id: Optional[str] = None
    chunked_comparison_strategy: ChunkedComparisonStrategy = ChunkedComparisonStrategy.AGGREGATE_FIRST

    # FAISS настройки
    use_faiss_search: bool = False
    faiss_index_id: Optional[str] = None
    similarity_top_k: int = 10
    similarity_threshold: float = 0.7

    # Состояние
    status: SessionStatus = SessionStatus.DRAFT
    progress: int = 0
    progress_message: str = ""

    # Результаты
    results: Dict[str, AnalysisResult] = field(default_factory=dict)
    comparison_matrix: Optional[ComparisonMatrix] = None
    error: Optional[str] = None

    # Временные метки
    created_at: datetime = field(default_factory=now_utc)
    queued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Пользователь (опционально)
    user_id: Optional[str] = None

    # Метаданные
    metadata: Metadata = field(default_factory=dict)

    def add_text(self, text: BaseText) -> None:
        """
        Добавить текст в сессию

        Args:
            text: Текст для добавления

        Raises:
            ValidationError: Если превышен лимит текстов
        """
        if len(self.texts) >= 5:
            raise ValidationError(
                message="Maximum 5 texts allowed per session",
                details={"current_count": len(self.texts)}
            )

        # Проверка на дубликаты
        if any(t.id == text.id for t in self.texts):
            raise ValidationError(
                message=f"Text '{text.id}' is already in the session",
                details={"text_id": text.id}
            )

        self.texts.append(text)

    def remove_text(self, text_id: str) -> bool:
        """
        Удалить текст из сессии

        Args:
            text_id: ID текста

        Returns:
            bool: True если удалён
        """
        original_count = len(self.texts)
        self.texts = [t for t in self.texts if t.id != text_id]
        return len(self.texts) < original_count

    def add_analyzer(self, analyzer: BaseAnalyzer) -> None:
        """
        Добавить анализатор

        Args:
            analyzer: Анализатор для добавления
        """
        # Проверка на дубликаты
        if any(a.name == analyzer.name for a in self.analyzers):
            raise ValidationError(
                message=f"Analyzer '{analyzer.name}' is already in the session",
                details={"analyzer_name": analyzer.name}
            )

        self.analyzers.append(analyzer)

    def remove_analyzer(self, analyzer_name: str) -> bool:
        """
        Удалить анализатор

        Args:
            analyzer_name: Имя анализатора

        Returns:
            bool: True если удалён
        """
        original_count = len(self.analyzers)
        self.analyzers = [a for a in self.analyzers if a.name != analyzer_name]
        return len(self.analyzers) < original_count

    def set_comparator(self, comparator: BaseComparator) -> None:
        """
        Установить компаратор

        Args:
            comparator: Компаратор для сравнения
        """
        self.comparator = comparator

    def get_text_by_id(self, text_id: str) -> Optional[BaseText]:
        """
        Получить текст по ID

        Args:
            text_id: ID текста

        Returns:
            Optional[BaseText]: Текст или None
        """
        for text in self.texts:
            if text.id == text_id:
                return text
        return None

    def get_analyzer_by_name(self, name: str) -> Optional[BaseAnalyzer]:
        """
        Получить анализатор по имени

        Args:
            name: Имя анализатора

        Returns:
            Optional[BaseAnalyzer]: Анализатор или None
        """
        for analyzer in self.analyzers:
            if analyzer.name == name:
                return analyzer
        return None

    def get_result(self, text_id: str, analyzer_name: str) -> Optional[AnalysisResult]:
        """
        Получить результат анализа

        Args:
            text_id: ID текста
            analyzer_name: Имя анализатора

        Returns:
            Optional[AnalysisResult]: Результат или None
        """
        key = f"{text_id}:{analyzer_name}"
        return self.results.get(key)

    def set_result(self, text_id: str, analyzer_name: str, result: AnalysisResult) -> None:
        """
        Установить результат анализа

        Args:
            text_id: ID текста
            analyzer_name: Имя анализатора
            result: Результат
        """
        key = f"{text_id}:{analyzer_name}"
        self.results[key] = result

    def get_execution_time(self) -> Optional[float]:
        """
        Получить время выполнения в секундах

        Returns:
            Optional[float]: Время выполнения или None
        """
        if self.started_at and self.completed_at:
            return calculate_duration(self.started_at, self.completed_at)
        return None

    def is_completed(self) -> bool:
        """
        Проверить завершена ли сессия

        Returns:
            bool: True если завершена
        """
        return self.status == SessionStatus.COMPLETED

    def is_running(self) -> bool:
        """
        Проверить выполняется ли сессия

        Returns:
            bool: True если выполняется
        """
        return self.status == SessionStatus.RUNNING

    def has_error(self) -> bool:
        """
        Проверить есть ли ошибка

        Returns:
            bool: True если есть ошибка
        """
        return self.status == SessionStatus.FAILED

    def validate(self) -> bool:
        """
        Валидировать сессию перед запуском

        Returns:
            bool: True если сессия валидна

        Raises:
            ValidationError: Если сессия невалидна
        """
        # Проверка наличия текстов
        if not self.texts:
            raise ValidationError(
                message="Session must have at least one text",
                details={"session_id": self.id}
            )

        # Проверка наличия анализаторов
        if not self.analyzers:
            raise ValidationError(
                message="Session must have at least one analyzer",
                details={"session_id": self.id}
            )

        # Проверка компаратора если текстов больше 1
        if len(self.texts) > 1 and not self.comparator:
            raise ValidationError(
                message="Session with multiple texts must have a comparator",
                details={"session_id": self.id, "text_count": len(self.texts)}
            )

        # Проверка chunking_strategy_id для CHUNKED режима
        if self.mode == AnalysisMode.CHUNKED and not self.chunking_strategy_id:
            raise ValidationError(
                message="CHUNKED mode requires chunking_strategy_id",
                details={"session_id": self.id, "mode": self.mode}
            )

        return True

    def estimate_total_time(self) -> float:
        """
        Оценить общее время выполнения в секундах

        Returns:
            float: Оценочное время
        """
        total_time = 0.0

        # Время на анализ каждого текста каждым анализатором
        for text in self.texts:
            text_length = text.get_length()
            for analyzer in self.analyzers:
                total_time += analyzer.get_estimated_time(text_length, self.mode)

        # Время на сравнения если нужно
        if len(self.texts) > 1 and self.comparator:
            num_comparisons = len(self.texts) * (len(self.texts) - 1) // 2
            avg_length = sum(t.get_length() for t in self.texts) / len(self.texts)
            comparison_time = self.comparator.get_estimated_time(
                avg_length, avg_length, self.mode
            )
            total_time += num_comparisons * comparison_time

        return total_time

    def to_dict(self) -> Dict[str, Any]:
        """
        Сериализация в словарь

        Returns:
            Dict: Данные сессии
        """
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "mode": self.mode.value,
            "chunking_strategy_id": self.chunking_strategy_id,
            "chunked_comparison_strategy": self.chunked_comparison_strategy.value,
            "use_faiss_search": self.use_faiss_search,
            "faiss_index_id": self.faiss_index_id,
            "similarity_top_k": self.similarity_top_k,
            "similarity_threshold": self.similarity_threshold,
            "text_count": len(self.texts),
            "text_ids": [t.id for t in self.texts],
            "analyzer_count": len(self.analyzers),
            "analyzer_names": [a.name for a in self.analyzers],
            "has_comparator": self.comparator is not None,
            "comparator_name": self.comparator.name if self.comparator else None,
            "result_count": len(self.results),
            "has_comparison_matrix": self.comparison_matrix is not None,
            "error": self.error,
            "execution_time": self.get_execution_time(),
            "estimated_time": self.estimate_total_time() if self.status == SessionStatus.DRAFT else None,
            "created_at": self.created_at.isoformat(),
            "queued_at": self.queued_at.isoformat() if self.queued_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "user_id": self.user_id,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        return (
            f"AnalysisSession(id={self.id}, name={self.name!r}, "
            f"texts={len(self.texts)}, status={self.status.value})"
        )

    def __repr__(self) -> str:
        return self.__str__()
