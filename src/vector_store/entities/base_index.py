"""
Базовый класс для FAISS индексов
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

from src.common.types import IndexID
from src.common.utils import now_utc


@dataclass
class IndexMetadata:
    """
    Метаданные индекса

    Содержит информацию о векторах и их источниках
    """
    # ID вектора → метаданные
    vector_metadata: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def add_metadata(self, vector_id: int, metadata: Dict[str, Any]) -> None:
        """Добавить метаданные для вектора"""
        self.vector_metadata[vector_id] = metadata

    def get_metadata(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """Получить метаданные вектора"""
        return self.vector_metadata.get(vector_id)

    def remove_metadata(self, vector_id: int) -> None:
        """Удалить метаданные вектора"""
        self.vector_metadata.pop(vector_id, None)

    def clear(self) -> None:
        """Очистить все метаданные"""
        self.vector_metadata.clear()


@dataclass
class IndexStats:
    """
    Статистика работы индекса
    """
    total_vectors: int = 0
    total_searches: int = 0
    total_additions: int = 0
    total_removals: int = 0
    avg_search_time_ms: float = 0.0
    created_at: datetime = field(default_factory=now_utc)
    last_updated_at: datetime = field(default_factory=now_utc)

    def record_search(self, search_time_ms: float) -> None:
        """Записать выполнение поиска"""
        self.total_searches += 1
        # Экспоненциальное сглаживание
        alpha = 0.1
        self.avg_search_time_ms = (
            alpha * search_time_ms +
            (1 - alpha) * self.avg_search_time_ms
        )
        self.last_updated_at = now_utc()

    def record_addition(self, count: int = 1) -> None:
        """Записать добавление векторов"""
        self.total_additions += count
        self.total_vectors += count
        self.last_updated_at = now_utc()

    def record_removal(self, count: int = 1) -> None:
        """Записать удаление векторов"""
        self.total_removals += count
        self.total_vectors -= count
        self.last_updated_at = now_utc()


class BaseIndex(ABC):
    """
    Базовый класс для всех FAISS индексов

    Определяет общий интерфейс для работы с векторными индексами
    """

    def __init__(
        self,
        index_id: IndexID,
        dimension: int,
        metric: str = "cosine",
        use_gpu: bool = False
    ):
        """
        Инициализация индекса

        Args:
            index_id: Уникальный ID индекса
            dimension: Размерность векторов
            metric: Метрика расстояния (cosine, l2, ip)
            use_gpu: Использовать GPU для ускорения
        """
        self.index_id = index_id
        self.dimension = dimension
        self.metric = metric
        self.use_gpu = use_gpu

        # FAISS индекс (будет создан в подклассах)
        self._index = None

        # Метаданные и статистика
        self.metadata = IndexMetadata()
        self.stats = IndexStats()

    @abstractmethod
    def build(self) -> None:
        """
        Построить индекс

        Должен быть вызван после добавления векторов перед поиском
        """
        pass

    @abstractmethod
    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: Optional[List[int]] = None
    ) -> List[int]:
        """
        Добавить векторы в индекс

        Args:
            vectors: Матрица векторов (N x D)
            ids: ID векторов (если None - генерируются автоматически)

        Returns:
            List[int]: ID добавленных векторов
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vectors: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Поиск ближайших соседей

        Args:
            query_vectors: Запросные векторы (N x D)
            k: Количество соседей

        Returns:
            Tuple[np.ndarray, np.ndarray]: (расстояния, ID векторов)
                - distances: (N x k) расстояния до соседей
                - ids: (N x k) ID найденных векторов
        """
        pass

    @abstractmethod
    def remove_vectors(self, ids: List[int]) -> int:
        """
        Удалить векторы из индекса

        Args:
            ids: ID векторов для удаления

        Returns:
            int: Количество удалённых векторов
        """
        pass

    def get_vector_count(self) -> int:
        """
        Получить количество векторов в индексе

        Returns:
            int: Количество векторов
        """
        if self._index is None:
            return 0
        return self._index.ntotal

    def is_trained(self) -> bool:
        """
        Проверить обучен ли индекс

        Returns:
            bool: True если обучен
        """
        if self._index is None:
            return False
        return self._index.is_trained

    def save(self, file_path: str) -> None:
        """
        Сохранить индекс на диск

        Args:
            file_path: Путь к файлу
        """
        import faiss

        if self._index is None:
            raise ValueError("Индекс не инициализирован")

        # Если индекс на GPU - переносим на CPU перед сохранением
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self._index)
            faiss.write_index(cpu_index, file_path)
        else:
            faiss.write_index(self._index, file_path)

    def load(self, file_path: str) -> None:
        """
        Загрузить индекс с диска

        Args:
            file_path: Путь к файлу
        """
        import faiss

        # Загружаем индекс
        self._index = faiss.read_index(file_path)

        # Если нужно GPU - переносим
        if self.use_gpu:
            self._move_to_gpu()

    def _move_to_gpu(self) -> None:
        """Переместить индекс на GPU"""
        import faiss

        if self._index is None:
            raise ValueError("Индекс не инициализирован")

        if not faiss.get_num_gpus():
            raise RuntimeError("GPU недоступен")

        # Создаём GPU ресурсы
        res = faiss.StandardGpuResources()

        # Переносим индекс на GPU
        self._index = faiss.index_cpu_to_gpu(res, 0, self._index)

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Нормализовать векторы (для cosine similarity)

        Args:
            vectors: Матрица векторов

        Returns:
            np.ndarray: Нормализованные векторы
        """
        import faiss

        # FAISS нормализация
        faiss.normalize_L2(vectors)
        return vectors

    def get_stats(self) -> Dict[str, Any]:
        """
        Получить статистику индекса

        Returns:
            Dict: Статистика
        """
        return {
            "index_id": self.index_id,
            "dimension": self.dimension,
            "metric": self.metric,
            "use_gpu": self.use_gpu,
            "total_vectors": self.stats.total_vectors,
            "total_searches": self.stats.total_searches,
            "total_additions": self.stats.total_additions,
            "total_removals": self.stats.total_removals,
            "avg_search_time_ms": self.stats.avg_search_time_ms,
            "is_trained": self.is_trained(),
            "created_at": self.stats.created_at.isoformat(),
            "last_updated_at": self.stats.last_updated_at.isoformat(),
        }

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self.index_id}, "
            f"dim={self.dimension}, vectors={self.get_vector_count()})"
        )

    def __repr__(self) -> str:
        return self.__str__()
