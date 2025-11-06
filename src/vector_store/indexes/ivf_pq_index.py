"""
IVF+PQ индекс - кластеризация + квантизация для компактности
"""
from typing import Optional, List, Tuple
import numpy as np
import faiss
from loguru import logger

from ..entities.base_index import BaseIndex
from src.common.types import IndexID


class IVFPQIndex(BaseIndex):
    """
    IVF+PQ (Inverted File + Product Quantization) индекс

    Комбинирует кластеризацию (IVF) с квантизацией (PQ) для сжатия векторов.

    Преимущества:
    - Очень компактный (экономит память)
    - Быстрый поиск
    - Масштабируется до миллиардов векторов

    Недостатки:
    - Требует обучения
    - Потеря точности из-за квантизации
    - Нельзя восстановить оригинальные векторы

    Параметры:
    - nlist: количество кластеров (как в IVF)
    - m: количество подпространств для PQ (должен делить dimension)
    - nbits: бит на код (4-8, обычно 8)
    - nprobe: количество проверяемых кластеров

    Рекомендуется для: >10M векторов, когда критична память
    """

    def __init__(
        self,
        index_id: IndexID,
        dimension: int,
        nlist: int = 100,
        m: int = 8,
        nbits: int = 8,
        nprobe: int = 10,
        metric: str = "cosine",
        use_gpu: bool = False
    ):
        """
        Инициализация IVF+PQ индекса

        Args:
            index_id: ID индекса
            dimension: Размерность векторов
            nlist: Количество кластеров
            m: Количество подпространств (dimension должна делиться на m)
            nbits: Бит на код PQ
            nprobe: Количество проверяемых кластеров
            metric: Метрика расстояния
            use_gpu: Использовать GPU
        """
        super().__init__(index_id, dimension, metric, use_gpu)

        # Проверяем делимость
        if dimension % m != 0:
            raise ValueError(
                f"Размерность {dimension} должна делиться на m={m}. "
                f"Попробуйте m={self._suggest_m(dimension)}"
            )

        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe
        self._init_index()

    def _suggest_m(self, dimension: int) -> int:
        """Предложить подходящее значение m"""
        # Делители размерности
        for m in [64, 32, 16, 8, 4]:
            if dimension % m == 0:
                return m
        return 1

    def _init_index(self) -> None:
        """Инициализировать FAISS индекс"""
        # Создаём квантизатор для IVF
        if self.metric == "cosine" or self.metric == "ip":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self._index = faiss.IndexIVFPQ(
                quantizer,
                self.dimension,
                self.nlist,
                self.m,
                self.nbits,
                faiss.METRIC_INNER_PRODUCT
            )
        elif self.metric == "l2":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self._index = faiss.IndexIVFPQ(
                quantizer,
                self.dimension,
                self.nlist,
                self.m,
                self.nbits,
                faiss.METRIC_L2
            )
        else:
            raise ValueError(f"Неподдерживаемая метрика: {self.metric}")

        # Устанавливаем nprobe
        self._index.nprobe = self.nprobe

        # Переносим на GPU если нужно
        if self.use_gpu:
            self._move_to_gpu()

        logger.info(
            f"✅ IVF+PQ индекс создан: {self.dimension}D, "
            f"nlist={self.nlist}, m={self.m}, nbits={self.nbits}"
        )

    def build(self) -> None:
        """
        Построить индекс (обучить)

        Должен быть вызван после добавления достаточного количества векторов
        """
        if not self.is_trained():
            logger.warning(
                "IVF+PQ индекс не обучен. "
                "Вызовите train() с достаточным количеством векторов (рекомендуется 30*nlist)"
            )

    def train(self, training_vectors: np.ndarray) -> None:
        """
        Обучить индекс на векторах

        Args:
            training_vectors: Векторы для обучения (рекомендуется >=30*nlist)
        """
        if training_vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Неверная размерность векторов: ожидается {self.dimension}, "
                f"получено {training_vectors.shape[1]}"
            )

        # Проверяем количество векторов
        min_vectors = self.nlist * 30
        if len(training_vectors) < min_vectors:
            logger.warning(
                f"Мало векторов для обучения: {len(training_vectors)}, "
                f"рекомендуется минимум {min_vectors}"
            )

        # Конвертируем в float32
        if training_vectors.dtype != np.float32:
            training_vectors = training_vectors.astype(np.float32)

        # Нормализуем для cosine
        if self.metric == "cosine":
            training_vectors = self._normalize_vectors(training_vectors.copy())

        # Обучаем
        logger.info(
            f"🎓 Обучение IVF+PQ индекса на {len(training_vectors)} векторах "
            f"(это может занять время)..."
        )
        self._index.train(training_vectors)

        logger.info("✅ IVF+PQ индекс обучен")

    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: Optional[List[int]] = None
    ) -> List[int]:
        """
        Добавить векторы в индекс

        Args:
            vectors: Матрица векторов (N x D)
            ids: ID векторов (опционально)

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

        # Обучаем если ещё не обучен
        if not self.is_trained():
            logger.info("Индекс не обучен. Обучаем на добавляемых векторах...")
            self.train(vectors)

        # Нормализуем для cosine
        if self.metric == "cosine":
            vectors = self._normalize_vectors(vectors.copy())

        # Текущий размер
        start_id = self._index.ntotal

        # Добавляем векторы
        self._index.add(vectors)

        # Генерируем ID
        generated_ids = list(range(start_id, start_id + len(vectors)))

        # Метаданные
        if ids:
            for gen_id, orig_id in zip(generated_ids, ids):
                self.metadata.add_metadata(gen_id, {"original_id": orig_id})

        # Статистика
        self.stats.record_addition(len(vectors))

        logger.info(f"➕ Добавлено {len(vectors)} векторов в IVF+PQ индекс")

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

        if not self.is_trained():
            raise RuntimeError("Индекс не обучен. Вызовите train() или add_vectors()")

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
            f"🔍 Поиск IVF+PQ завершён: {len(query_vectors)} запросов, "
            f"k={k}, nprobe={self.nprobe}, время={search_time_ms:.2f}ms"
        )

        return distances, ids

    def remove_vectors(self, ids: List[int]) -> int:
        """
        Удалить векторы из индекса

        Args:
            ids: ID векторов для удаления

        Returns:
            int: Количество удалённых векторов
        """
        if not ids:
            return 0

        try:
            # Создаём селектор для удаления
            id_selector = faiss.IDSelectorBatch(ids)

            # Удаляем
            removed = self._index.remove_ids(id_selector)

            # Обновляем метаданные
            for vid in ids:
                self.metadata.remove_metadata(vid)

            # Статистика
            self.stats.record_removal(removed)

            logger.info(f"🗑️ Удалено {removed} векторов из IVF+PQ индекса")

            return removed

        except Exception as e:
            logger.error(f"Ошибка удаления векторов: {e}")
            return 0

    def set_nprobe(self, nprobe: int) -> None:
        """
        Изменить количество проверяемых кластеров

        Args:
            nprobe: Новое значение nprobe
        """
        self.nprobe = nprobe
        self._index.nprobe = nprobe
        logger.info(f"Установлено nprobe={nprobe}")

    def get_nprobe(self) -> int:
        """Получить текущее значение nprobe"""
        return self._index.nprobe

    def estimate_memory_mb(self) -> float:
        """
        Оценить потребление памяти индексом

        Returns:
            float: Память в МБ
        """
        # Формула для PQ: N * m * nbits / 8 / 1024^2
        n_vectors = self.get_vector_count()
        memory_bytes = n_vectors * self.m * self.nbits / 8

        # Добавляем overhead для кластеризации
        memory_bytes += self.nlist * self.dimension * 4  # float32

        return memory_bytes / (1024 * 1024)
