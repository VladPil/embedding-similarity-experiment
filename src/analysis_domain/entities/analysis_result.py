"""
Результат анализа текста
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime

from src.common.types import JSON, Metadata
from src.common.utils import now_utc


@dataclass
class AnalysisResult:
    """
    Результат анализа одного текста одним анализатором

    Содержит данные анализа и метаданные о выполнении
    """

    # Основные поля
    text_id: str
    analyzer_name: str

    # Данные результата
    data: JSON = field(default_factory=dict)

    # Интерпретация результата в человекочитаемом виде
    interpretation: Optional[str] = None

    # Метаданные выполнения
    execution_time_ms: Optional[float] = None
    mode: str = "full_text"  # full_text или chunked

    # Дополнительные метаданные
    metadata: Metadata = field(default_factory=dict)

    # Временная метка
    created_at: datetime = field(default_factory=now_utc)

    def get_summary(self) -> str:
        """
        Получить краткое резюме результата

        Returns:
            str: Краткое описание
        """
        if self.interpretation:
            # Берём первые 200 символов интерпретации
            return self.interpretation[:200] + "..." if len(self.interpretation) > 200 else self.interpretation

        return f"Анализ {self.analyzer_name} для текста {self.text_id}"

    def has_error(self) -> bool:
        """
        Проверить наличие ошибки в результате

        Returns:
            bool: True если есть ошибка
        """
        return self.data.get("error") is not None

    def get_error(self) -> Optional[str]:
        """
        Получить сообщение об ошибке

        Returns:
            Optional[str]: Сообщение об ошибке или None
        """
        return self.data.get("error")

    def to_dict(self) -> Dict[str, Any]:
        """
        Сериализация в словарь

        Returns:
            Dict: Данные результата
        """
        return {
            "text_id": self.text_id,
            "analyzer_name": self.analyzer_name,
            "data": self.data,
            "interpretation": self.interpretation,
            "execution_time_ms": self.execution_time_ms,
            "mode": self.mode,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "has_error": self.has_error(),
        }

    def __str__(self) -> str:
        return f"AnalysisResult(text={self.text_id}, analyzer={self.analyzer_name})"

    def __repr__(self) -> str:
        return self.__str__()
