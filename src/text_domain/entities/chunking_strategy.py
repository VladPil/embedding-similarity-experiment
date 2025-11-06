"""
Стратегия разбиения текста на чанки
"""
from dataclasses import dataclass, field
from typing import Optional, Dict
from datetime import datetime

from src.common.types import Metadata
from src.common.utils import now_utc


@dataclass
class ChunkingStrategy:
    """
    Стратегия для адаптивного разбиения текста на чанки

    Поддерживает различные параметры для оптимального разбиения
    в зависимости от типа текста и задачи
    """

    # Основные поля
    id: str
    name: str

    # Базовые параметры размера
    base_chunk_size: int = 2000  # Целевой размер чанка в символах
    min_chunk_size: int = 500    # Минимальный размер
    max_chunk_size: int = 4000   # Максимальный размер
    overlap_percentage: float = 0.1  # Перекрытие между чанками (10%)

    # Адаптивность к структуре текста
    use_sentence_boundaries: bool = True   # Не резать посередине предложения
    use_paragraph_boundaries: bool = True  # Предпочитать границы параграфов
    balance_chunks: bool = True            # Выравнивать размеры чанков

    # Метаданные
    is_default: bool = False
    metadata: Metadata = field(default_factory=dict)

    # Временные метки
    created_at: datetime = field(default_factory=now_utc)
    updated_at: datetime = field(default_factory=now_utc)

    def get_overlap_size(self) -> int:
        """
        Вычислить размер перекрытия в символах

        Returns:
            int: Размер перекрытия
        """
        return int(self.base_chunk_size * self.overlap_percentage)

    def get_effective_chunk_size(self) -> int:
        """
        Получить эффективный размер чанка (с учётом перекрытия)

        Returns:
            int: Эффективный размер
        """
        return self.base_chunk_size - self.get_overlap_size()

    def validate(self) -> bool:
        """
        Валидация параметров стратегии

        Returns:
            bool: True если параметры корректны

        Raises:
            ValueError: Если параметры некорректны
        """
        if self.min_chunk_size >= self.base_chunk_size:
            raise ValueError(
                f"min_chunk_size ({self.min_chunk_size}) должен быть меньше "
                f"base_chunk_size ({self.base_chunk_size})"
            )

        if self.base_chunk_size >= self.max_chunk_size:
            raise ValueError(
                f"base_chunk_size ({self.base_chunk_size}) должен быть меньше "
                f"max_chunk_size ({self.max_chunk_size})"
            )

        if not 0 <= self.overlap_percentage <= 0.5:
            raise ValueError(
                f"overlap_percentage ({self.overlap_percentage}) должен быть в диапазоне [0, 0.5]"
            )

        return True

    def estimate_chunk_count(self, text_length: int) -> int:
        """
        Оценить количество чанков для текста заданной длины

        Args:
            text_length: Длина текста в символах

        Returns:
            int: Примерное количество чанков
        """
        if text_length <= self.base_chunk_size:
            return 1

        effective_size = self.get_effective_chunk_size()
        return max(1, (text_length + effective_size - 1) // effective_size)

    def to_dict(self) -> Dict:
        """
        Сериализация в словарь

        Returns:
            Dict: Параметры стратегии
        """
        return {
            "id": self.id,
            "name": self.name,
            "base_chunk_size": self.base_chunk_size,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "overlap_percentage": self.overlap_percentage,
            "overlap_size": self.get_overlap_size(),
            "effective_chunk_size": self.get_effective_chunk_size(),
            "use_sentence_boundaries": self.use_sentence_boundaries,
            "use_paragraph_boundaries": self.use_paragraph_boundaries,
            "balance_chunks": self.balance_chunks,
            "is_default": self.is_default,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def create_default(cls) -> "ChunkingStrategy":
        """
        Создать стратегию по умолчанию

        Returns:
            ChunkingStrategy: Стратегия с настройками по умолчанию
        """
        return cls(
            id="default",
            name="Default Strategy",
            base_chunk_size=2000,
            min_chunk_size=500,
            max_chunk_size=4000,
            overlap_percentage=0.1,
            use_sentence_boundaries=True,
            use_paragraph_boundaries=True,
            balance_chunks=True,
            is_default=True,
        )

    @classmethod
    def create_small_chunks(cls) -> "ChunkingStrategy":
        """
        Создать стратегию для мелких чанков (для детального анализа)

        Returns:
            ChunkingStrategy: Стратегия с мелкими чанками
        """
        return cls(
            id="small_chunks",
            name="Small Chunks Strategy",
            base_chunk_size=1000,
            min_chunk_size=250,
            max_chunk_size=2000,
            overlap_percentage=0.15,
            use_sentence_boundaries=True,
            use_paragraph_boundaries=True,
            balance_chunks=True,
        )

    @classmethod
    def create_large_chunks(cls) -> "ChunkingStrategy":
        """
        Создать стратегию для крупных чанков (для общего анализа)

        Returns:
            ChunkingStrategy: Стратегия с крупными чанками
        """
        return cls(
            id="large_chunks",
            name="Large Chunks Strategy",
            base_chunk_size=4000,
            min_chunk_size=1000,
            max_chunk_size=8000,
            overlap_percentage=0.05,
            use_sentence_boundaries=True,
            use_paragraph_boundaries=True,
            balance_chunks=True,
        )

    @classmethod
    def create_fixed_size(cls, size: int) -> "ChunkingStrategy":
        """
        Создать стратегию с фиксированным размером (без адаптивности)

        Args:
            size: Фиксированный размер чанка

        Returns:
            ChunkingStrategy: Стратегия с фиксированным размером
        """
        return cls(
            id=f"fixed_{size}",
            name=f"Fixed Size {size} Strategy",
            base_chunk_size=size,
            min_chunk_size=size,
            max_chunk_size=size,
            overlap_percentage=0.0,
            use_sentence_boundaries=False,
            use_paragraph_boundaries=False,
            balance_chunks=False,
        )

    def __str__(self) -> str:
        return f"ChunkingStrategy(name={self.name!r}, base_size={self.base_chunk_size})"

    def __repr__(self) -> str:
        return self.__str__()
