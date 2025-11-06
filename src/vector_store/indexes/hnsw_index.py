"""
HNSW индекс - граф для быстрого приближённого поиска
"""
from typing import Optional, List, Tuple
import numpy as np
import faiss
from loguru import logger

from ..entities.base_index import BaseIndex
from src.common.types import IndexID


class HNSWIndex(BaseIndex):
    """
    HNSW (Hierarchical Navigable Small World) индекс

    Строит иерархический граф для быстрого поиска.

    Преимущества:
    - Очень быстрый поиск
    - Высокая точность
    - Не требует обучения
    - Хорошо масштабируется

    Недостатки:
    - Занимает много памяти
    - Медленное добавление векторов
    - Не поддерживает удаление

    Параметры:
    - M: количество связей (16-64, default=32)
    - efConstruction: точность построения (40-500, default=40)
    - efSearch: точность поиска (>= k, default=16)

    Рекомендуется для: любых размеров, когда нужна скорость поиска
    """

    def __init__(
        self,
        index_id: IndexID,
        dimension: int,
        M: int = 32,
        efConstruction: int = 40,
        efSearch: int = 16,
        metric: str = "cosine",
        use_gpu: bool = False
    ):
        """
        Инициализация HNSW индекса

        Args:
            index_id: ID индекса
            dimension: Размерность векторов
            M: Количество связей на узел (больше = точнее но медленнее)
            efConstruction: Параметр точности построения
            efSearch: Параметр точности поиска
            metric: Метрика расстояния
            use_gpu: Использовать GPU (HNSW плохо работает на GPU)
        """
        super().__init__(index_id, dimension, metric, use_gpu)
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch

        if use_gpu:
            logger.warning("HNSW индекс плохо оптимизирован для GPU, используем CPU")
            self.use_gpu = False

        self._init_index()

    def _init_index(self) -> None:
        """Инициализировать FAISS индекс"""
        # HNSW доступен только для L2 и IP метрик в FAISS
        if self.metric == "cosine" or self.metric == "ip":
            self._index = faiss.IndexHNSWFlat(self.dimension, self.M, faiss.METRIC_INNER_PRODUCT)
        elif self.metric == "l2":
            self._index = faiss.IndexHNSWFlat(self.dimension, self.M, faiss.METRIC_L2)
        else:
            raise ValueError(f"Неподдерживаемая метрика для HNSW: {self.metric}")

        # Устанавливаем параметры
        self._index.hnsw.efConstruction = self.efConstruction
        self._index.hnsw.efSearch = self.efSearch

        logger.info(
            f"✅ HNSW индекс создан: {self.dimension}D, "
            f"M={self.M}, efConstruction={self.efConstruction}, efSearch={self.efSearch}"
        )

    def build(self) -> None:
        """
        Построить индекс

        HNSW строится инкрементально при добавлении векторов
        """
        logger.debug("HNSW индекс строится инкрементально при добавлении векторов")

    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: Optional[List[int]] = None
    ) -> List[int]:
        """
        Добавить векторы в индекс

        Args:
            vectors: Матрица векторов (N x D)
            ids: ID векторов (игнорируется)

        Returns:
            List[int]: ID добавленных векторов
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Неверная размерность векторов: ожидается {self.dimension}, "
                f"получено {vectors.shape[1]}"
            )

        # Конвертируем в float32
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        # Нормализуем для cosine
        if self.metric == "cosine":
            vectors = self._normalize_vectors(vectors.copy())

        # Текущий размер
        start_id = self._index.ntotal

        # Добавляем векторы (медленная операция для HNSW!)
        logger.info(f"➕ Добавление {len(vectors)} векторов в HNSW (может быть медленно)...")
        self._index.add(vectors)

        # Генерируем ID
        generated_ids = list(range(start_id, start_id + len(vectors)))

        # Метаданные
        if ids:
            for gen_id, orig_id in zip(generated_ids, ids):
                self.metadata.add_metadata(gen_id, {"original_id": orig_id})

        # Статистика
        self.stats.record_addition(len(vectors))

        logger.info(f"✅ Добавлено {len(vectors)} векторов в HNSW индекс")

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

        # Ограничиваем k
        k = min(k, self._index.ntotal)

        # Поиск
        start_time = time.time()
        distances, ids = self._index.search(query_vectors, k)
        search_time_ms = (time.time() - start_time) * 1000

        # Статистика
        self.stats.record_search(search_time_ms)

        logger.debug(
            f"🔍 Поиск HNSW завершён: {len(query_vectors)} запросов, "
            f"k={k}, efSearch={self.efSearch}, время={search_time_ms:.2f}ms"
        )

        return distances, ids

    def remove_vectors(self, ids: List[int]) -> int:
        """
        Удалить векторы из индекса

        ВНИМАНИЕ: HNSW не поддерживает удаление!

        Args:
            ids: ID векторов для удаления

        Returns:
            int: Количество удалённых векторов (всегда 0)
        """
        logger.warning(
            "HNSW индекс не поддерживает удаление векторов. "
            "Для удаления нужно пересоздать индекс."
        )
        return 0

    def set_efSearch(self, efSearch: int) -> None:
        """
        Изменить параметр точности поиска

        Больше efSearch = выше точность но медленнее
        Рекомендуется: efSearch >= k

        Args:
            efSearch: Новое значение efSearch
        """
        self.efSearch = efSearch
        self._index.hnsw.efSearch = efSearch
        logger.info(f"Установлено efSearch={efSearch}")

    def get_efSearch(self) -> int:
        """Получить текущее значение efSearch"""
        return self._index.hnsw.efSearch

    def get_graph_stats(self) -> dict:
        """
        Получить статистику графа

        Returns:
            dict: Статистика HNSW графа
        """
        hnsw = self._index.hnsw

        return {
            "M": self.M,
            "efConstruction": self.efConstruction,
            "efSearch": self.efSearch,
            "max_level": hnsw.max_level,
            "entry_point": hnsw.entry_point,
            "cum_nneighbor_per_level": list(hnsw.cum_nneighbor_per_level),
        }
