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
    method: str = "fixed_size"  # Метод разбиения: fixed_size, sentence, paragraph

    # Базовые параметры размера (для fixed_size)
    base_chunk_size: int = 2000  # Целевой размер чанка в символах
    min_chunk_size: int = 500    # Минимальный размер
    max_chunk_size: int = 4000   # Максимальный размер
    overlap_percentage: float = 0.0  # Перекрытие между чанками (по умолчанию отключено)

    # Параметры для fixed_size
    chunk_size: Optional[int] = None  # Фиксированный размер
    overlap: Optional[int] = None     # Перекрытие в символах

    # Параметры для sentence-based
    sentences_per_chunk: Optional[int] = None
    overlap_sentences: Optional[int] = None

    # Параметры для paragraph-based
    paragraphs_per_chunk: Optional[int] = None
    overlap_paragraphs: Optional[int] = None

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
        if self.min_chunk_size > self.base_chunk_size:
            raise ValueError(
                f"min_chunk_size ({self.min_chunk_size}) должен быть меньше или равен "
                f"base_chunk_size ({self.base_chunk_size})"
            )

        if self.base_chunk_size > self.max_chunk_size:
            raise ValueError(
                f"base_chunk_size ({self.base_chunk_size}) должен быть меньше или равен "
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
    def create_fixed_size(cls, strategy_id: str, chunk_size: int, overlap: int = 0) -> "ChunkingStrategy":
        """
        Создать стратегию с фиксированным размером

        Args:
            strategy_id: Уникальный ID стратегии
            chunk_size: Фиксированный размер чанка
            overlap: Перекрытие в символах

        Returns:
            ChunkingStrategy: Стратегия с фиксированным размером
        """
        return cls(
            id=strategy_id,
            name=f"Fixed Size {chunk_size} Strategy",
            method="fixed_size",
            base_chunk_size=chunk_size,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=chunk_size,
            max_chunk_size=chunk_size,
            overlap_percentage=overlap / chunk_size if chunk_size > 0 else 0.0,
            use_sentence_boundaries=False,
            use_paragraph_boundaries=False,
            balance_chunks=False,
        )

    @classmethod
    def create_sentence_based(cls, strategy_id: str, sentences_per_chunk: int, overlap_sentences: int = 0) -> "ChunkingStrategy":
        """
        Создать стратегию на основе предложений

        Args:
            strategy_id: Уникальный ID стратегии
            sentences_per_chunk: Количество предложений в чанке
            overlap_sentences: Перекрытие в предложениях

        Returns:
            ChunkingStrategy: Стратегия на основе предложений
        """
        return cls(
            id=strategy_id,
            name=f"Sentence-based {sentences_per_chunk} Strategy",
            method="sentence",
            sentences_per_chunk=sentences_per_chunk,
            overlap_sentences=overlap_sentences,
            use_sentence_boundaries=True,
            use_paragraph_boundaries=False,  # Для sentence-based не используем paragraph boundaries
            balance_chunks=False,  # Для sentence-based не выравниваем чанки
        )

    @classmethod
    def create_paragraph_based(cls, strategy_id: str, paragraphs_per_chunk: int, overlap_paragraphs: int = 0) -> "ChunkingStrategy":
        """
        Создать стратегию на основе параграфов

        Args:
            strategy_id: Уникальный ID стратегии
            paragraphs_per_chunk: Количество параграфов в чанке
            overlap_paragraphs: Перекрытие в параграфах

        Returns:
            ChunkingStrategy: Стратегия на основе параграфов
        """
        return cls(
            id=strategy_id,
            name=f"Paragraph-based {paragraphs_per_chunk} Strategy",
            method="paragraph",
            paragraphs_per_chunk=paragraphs_per_chunk,
            overlap_paragraphs=overlap_paragraphs,
            use_sentence_boundaries=False,  # Для paragraph-based не используем sentence boundaries
            use_paragraph_boundaries=True,
            balance_chunks=False,  # Для paragraph-based не выравниваем чанки
        )

    @classmethod
    def create_adaptive(cls, text_length: int, model_context_length: int = 8192, prompt_overhead: int = 1000) -> "ChunkingStrategy":
        """
        Создать адаптивную стратегию на основе размера текста и контекста модели

        Args:
            text_length: Длина текста в символах
            model_context_length: Максимальная длина контекста модели (токены)
            prompt_overhead: Количество токенов, резервируемых под промпт

        Returns:
            ChunkingStrategy: Адаптивная стратегия
        """
        # Примерно 4 символа на токен для русского языка
        chars_per_token = 4

        # Доступное место для текста в токенах
        available_tokens = model_context_length - prompt_overhead

        # Доступное место для текста в символах
        available_chars = available_tokens * chars_per_token

        # Определяем оптимальное количество чанков
        if text_length <= available_chars:
            # Текст помещается в один чанк
            chunk_size = max(500, text_length)
            num_chunks = 1
        else:
            # Нужно разбить на несколько чанков
            # Стремимся к chunk_size ~= available_chars * 0.8 (оставляем запас)
            optimal_chunk_size = int(available_chars * 0.8)
            num_chunks = max(2, (text_length + optimal_chunk_size - 1) // optimal_chunk_size)
            chunk_size = max(500, text_length // num_chunks)

        # Определяем размеры с запасами
        base_chunk_size = min(chunk_size, available_chars)
        min_chunk_size = max(200, base_chunk_size // 4)
        max_chunk_size = min(available_chars, base_chunk_size * 2)

        # Определяем перекрытие (10% для коротких текстов, 5% для длинных)
        overlap_percentage = 0.1 if text_length < 50000 else 0.05

        return cls(
            id=f"adaptive_{text_length}",
            name=f"Adaptive Strategy for {text_length} chars",
            method="fixed_size",
            base_chunk_size=base_chunk_size,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            overlap_percentage=overlap_percentage,
            chunk_size=base_chunk_size,
            overlap=int(base_chunk_size * overlap_percentage),
            use_sentence_boundaries=True,
            use_paragraph_boundaries=True,
            balance_chunks=True,
            metadata={
                "text_length": text_length,
                "estimated_chunks": num_chunks,
                "model_context_length": model_context_length,
                "prompt_overhead": prompt_overhead,
                "adaptive": True
            }
        )

    def __str__(self) -> str:
        return f"ChunkingStrategy(name={self.name!r}, base_size={self.base_chunk_size})"

    def __repr__(self) -> str:
        return self.__str__()
