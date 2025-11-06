"""
Результат сравнения двух текстов
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime

from src.common.types import JSON, Metadata
from src.common.utils import now_utc


@dataclass
class ComparisonResult:
    """
    Результат сравнения двух текстов компаратором

    Содержит оценку сходства и детальные данные сравнения
    """

    # Сравниваемые тексты
    text1_id: str
    text2_id: str

    # Метод сравнения
    comparator_name: str

    # Оценка сходства (0.0 - 1.0)
    similarity_score: float

    # Детальные данные сравнения
    details: JSON = field(default_factory=dict)

    # Объяснение результата
    explanation: Optional[str] = None

    # Метаданные выполнения
    execution_time_ms: Optional[float] = None
    mode: str = "full_text"

    # Дополнительные метаданные
    metadata: Metadata = field(default_factory=dict)

    # Временная метка
    created_at: datetime = field(default_factory=now_utc)

    def get_similarity_percentage(self) -> float:
        """
        Получить сходство в процентах

        Returns:
            float: Сходство 0-100
        """
        return self.similarity_score * 100

    def is_similar(self, threshold: float = 0.7) -> bool:
        """
        Проверить превышает ли сходство порог

        Args:
            threshold: Пороговое значение (0.0 - 1.0)

        Returns:
            bool: True если похожи
        """
        return self.similarity_score >= threshold

    def get_similarity_level(self) -> str:
        """
        Получить текстовый уровень сходства

        Returns:
            str: Уровень (очень высокий, высокий, средний, низкий, очень низкий)
        """
        score = self.similarity_score

        if score >= 0.9:
            return "очень высокий"
        elif score >= 0.7:
            return "высокий"
        elif score >= 0.5:
            return "средний"
        elif score >= 0.3:
            return "низкий"
        else:
            return "очень низкий"

    def to_dict(self) -> Dict[str, Any]:
        """
        Сериализация в словарь

        Returns:
            Dict: Данные результата
        """
        return {
            "text1_id": self.text1_id,
            "text2_id": self.text2_id,
            "comparator_name": self.comparator_name,
            "similarity_score": self.similarity_score,
            "similarity_percentage": self.get_similarity_percentage(),
            "similarity_level": self.get_similarity_level(),
            "details": self.details,
            "explanation": self.explanation,
            "execution_time_ms": self.execution_time_ms,
            "mode": self.mode,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    def __str__(self) -> str:
        return (
            f"ComparisonResult(texts={self.text1_id}↔{self.text2_id}, "
            f"score={self.similarity_score:.2f})"
        )

    def __repr__(self) -> str:
        return self.__str__()
