"""
Flat индекс FAISS - точный поиск без сжатия
"""
from typing import Optional, List, Tuple
import numpy as np
import faiss
from loguru import logger

from ..entities.base_index import BaseIndex
from src.common.types import IndexID


class FlatIndex(BaseIndex):
    """
    Flat индекс - точный поиск методом грубой силы

    Преимущества:
    - Точный поиск (100% recall)
    - Простота
    - Не требует обучения

    Недостатки:
    - Медленный на больших датасетах
    - Линейная сложность O(n)

    Рекомендуется для: <100K векторов
    """

    def __init__(
        self,
        index_id: IndexID,
        dimension: int,
        metric: str = "cosine",
        use_gpu: bool = False
    ):
        """
        Инициализация Flat индекса

        Args:
            index_id: ID индекса
            dimension: Размерность векторов
            metric: Метрика (cosine, l2, ip)
            use_gpu: Использовать GPU
        """
        super().__init__(index_id, dimension, metric, use_gpu)
        self._init_index()

    def _init_index(self) -> None:
        """Инициализировать FAISS индекс"""
        # Выбираем тип индекса в зависимости от метрики
        if self.metric == "cosine":
            # Для cosine используем IP (Inner Product) с нормализацией
            self._index = faiss.IndexFlatIP(self.dimension)
        elif self.metric == "l2":
            # L2 расстояние (Euclidean)
            self._index = faiss.IndexFlatL2(self.dimension)
        elif self.metric == "ip":
            # Inner Product (скалярное произведение)
            self._index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Неподдерживаемая метрика: {self.metric}")

        # Переносим на GPU если нужно
        if self.use_gpu:
            self._move_to_gpu()

        logger.info(f"✅ Flat индекс создан: {self.dimension}D, метрика={self.metric}")

    def build(self) -> None:
        """
        Построить индекс

        Для Flat индекса не требуется обучение
        """
        # Flat индекс не требует обучения
        logger.debug("Flat индекс не требует обучения")

    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: Optional[List[int]] = None
    ) -> List[int]:
        """
        Добавить векторы в индекс

        Args:
            vectors: Матрица векторов (N x D)
            ids: ID векторов (игнорируется для Flat индекса)

        Returns:
            List[int]: ID добавленных векторов (автоинкремент от текущего размера)
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Неверная размерность векторов: ожидается {self.dimension}, "
                f"получено {vectors.shape[1]}"
            )

        # Конвертируем в float32 если нужно
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        # Нормализуем для cosine similarity
        if self.metric == "cosine":
            vectors = self._normalize_vectors(vectors.copy())

        # Текущий размер индекса
        start_id = self._index.ntotal

        # Добавляем векторы
        self._index.add(vectors)

        # Генерируем ID
        generated_ids = list(range(start_id, start_id + len(vectors)))

        # Добавляем метаданные если предоставлены
        if ids:
            for gen_id, orig_id in zip(generated_ids, ids):
                self.metadata.add_metadata(gen_id, {"original_id": orig_id})

        # Статистика
        self.stats.record_addition(len(vectors))

        logger.info(f"➕ Добавлено {len(vectors)} векторов в Flat индекс")

        return generated_ids

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
            Tuple[np.ndarray, np.ndarray]: (расстояния, ID)
        """
        import time

        if query_vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Неверная размерность запроса: ожидается {self.dimension}, "
                f"получено {query_vectors.shape[1]}"
            )

        # Конвертируем в float32
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)

        # Нормализуем для cosine
        if self.metric == "cosine":
            query_vectors = self._normalize_vectors(query_vectors.copy())

        # Ограничиваем k размером индекса
        k = min(k, self._index.ntotal)

        # Поиск
        start_time = time.time()
        distances, ids = self._index.search(query_vectors, k)
        search_time_ms = (time.time() - start_time) * 1000

        # Статистика
        self.stats.record_search(search_time_ms)

        logger.debug(
            f"🔍 Поиск завершён: {len(query_vectors)} запросов, "
            f"k={k}, время={search_time_ms:.2f}ms"
        )

        return distances, ids

    def remove_vectors(self, ids: List[int]) -> int:
        """
        Удалить векторы из индекса

        ВНИМАНИЕ: Flat индекс не поддерживает удаление!
        Нужно пересоздавать индекс без удаляемых векторов.

        Args:
            ids: ID векторов для удаления

        Returns:
            int: Количество удалённых векторов (всегда 0)
        """
        logger.warning(
            "Flat индекс не поддерживает удаление векторов. "
            "Для удаления нужно пересоздать индекс."
        )
        return 0

    def reconstruct_vector(self, vector_id: int) -> np.ndarray:
        """
        Восстановить вектор по ID

        Args:
            vector_id: ID вектора

        Returns:
            np.ndarray: Вектор
        """
        return self._index.reconstruct(vector_id)

    def reconstruct_batch(self, vector_ids: List[int]) -> np.ndarray:
        """
        Восстановить несколько векторов

        Args:
            vector_ids: ID векторов

        Returns:
            np.ndarray: Матрица векторов (N x D)
        """
        vectors = np.vstack([
            self._index.reconstruct(vid) for vid in vector_ids
        ])
        return vectors
