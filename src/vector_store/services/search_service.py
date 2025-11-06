"""
Сервис для работы с векторным поиском
"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from loguru import logger

from ..entities.base_index import BaseIndex
from ..factory.index_factory import IndexFactory
from src.common.types import IndexID
from src.common.exceptions import ModelError, ValidationError


class SearchService:
    """
    Сервис для векторного поиска

    Управляет несколькими индексами и предоставляет
    высокоуровневый интерфейс для поиска
    """

    def __init__(self):
        """Инициализация сервиса"""
        # Словарь индексов: index_id -> BaseIndex
        self.indexes: Dict[IndexID, BaseIndex] = {}

    def create_index(
        self,
        index_id: IndexID,
        index_type: str,
        dimension: int,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseIndex:
        """
        Создать новый индекс

        Args:
            index_id: ID индекса
            index_type: Тип индекса
            dimension: Размерность векторов
            config: Конфигурация индекса

        Returns:
            BaseIndex: Созданный индекс
        """
        if index_id in self.indexes:
            raise ValidationError(
                f"Индекс {index_id} уже существует",
                details={"index_id": index_id}
            )

        # Создаём индекс через фабрику
        index = IndexFactory.create_index(
            index_type=index_type,
            index_id=index_id,
            dimension=dimension,
            config=config
        )

        # Сохраняем
        self.indexes[index_id] = index

        logger.info(f"✅ Индекс {index_id} создан: {index}")

        return index

    def get_index(self, index_id: IndexID) -> Optional[BaseIndex]:
        """
        Получить индекс по ID

        Args:
            index_id: ID индекса

        Returns:
            Optional[BaseIndex]: Индекс или None
        """
        return self.indexes.get(index_id)

    def delete_index(self, index_id: IndexID) -> bool:
        """
        Удалить индекс

        Args:
            index_id: ID индекса

        Returns:
            bool: True если удалён
        """
        if index_id in self.indexes:
            del self.indexes[index_id]
            logger.info(f"🗑️ Индекс {index_id} удалён")
            return True
        return False

    async def add_vectors(
        self,
        index_id: IndexID,
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """
        Добавить векторы в индекс

        Args:
            index_id: ID индекса
            vectors: Матрица векторов (N x D)
            metadata: Метаданные для каждого вектора

        Returns:
            List[int]: ID добавленных векторов

        Raises:
            ValidationError: Если индекс не найден
        """
        index = self.get_index(index_id)
        if not index:
            raise ValidationError(
                f"Индекс {index_id} не найден",
                details={"index_id": index_id}
            )

        # Добавляем векторы
        vector_ids = index.add_vectors(vectors)

        # Добавляем метаданные если есть
        if metadata:
            for vid, meta in zip(vector_ids, metadata):
                index.metadata.add_metadata(vid, meta)

        logger.info(f"➕ Добавлено {len(vectors)} векторов в индекс {index_id}")

        return vector_ids

    async def search(
        self,
        index_id: IndexID,
        query_vectors: np.ndarray,
        k: int = 10,
        return_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Поиск ближайших соседей

        Args:
            index_id: ID индекса
            query_vectors: Запросные векторы (N x D)
            k: Количество соседей
            return_metadata: Возвращать метаданные найденных векторов

        Returns:
            List[Dict]: Результаты поиска для каждого запроса
                [
                    {
                        "query_index": 0,
                        "results": [
                            {"vector_id": 123, "distance": 0.95, "metadata": {...}},
                            ...
                        ]
                    },
                    ...
                ]

        Raises:
            ValidationError: Если индекс не найден
        """
        index = self.get_index(index_id)
        if not index:
            raise ValidationError(
                f"Индекс {index_id} не найден",
                details={"index_id": index_id}
            )

        # Поиск
        distances, ids = index.search(query_vectors, k)

        # Формируем результаты
        results = []

        for query_idx in range(len(query_vectors)):
            query_result = {
                "query_index": query_idx,
                "results": []
            }

            for i in range(k):
                vector_id = int(ids[query_idx, i])
                distance = float(distances[query_idx, i])

                result_item = {
                    "vector_id": vector_id,
                    "distance": distance
                }

                # Добавляем метаданные если нужно
                if return_metadata:
                    meta = index.metadata.get_metadata(vector_id)
                    if meta:
                        result_item["metadata"] = meta

                query_result["results"].append(result_item)

            results.append(query_result)

        return results

    async def search_single(
        self,
        index_id: IndexID,
        query_vector: np.ndarray,
        k: int = 10,
        return_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Поиск для одного запросного вектора

        Args:
            index_id: ID индекса
            query_vector: Запросный вектор (D,)
            k: Количество соседей
            return_metadata: Возвращать метаданные

        Returns:
            List[Dict]: Результаты поиска
        """
        # Преобразуем в матрицу (1 x D)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Поиск
        results = await self.search(
            index_id=index_id,
            query_vectors=query_vector,
            k=k,
            return_metadata=return_metadata
        )

        # Возвращаем только результаты первого (единственного) запроса
        return results[0]["results"]

    async def remove_vectors(
        self,
        index_id: IndexID,
        vector_ids: List[int]
    ) -> int:
        """
        Удалить векторы из индекса

        Args:
            index_id: ID индекса
            vector_ids: ID векторов для удаления

        Returns:
            int: Количество удалённых векторов
        """
        index = self.get_index(index_id)
        if not index:
            raise ValidationError(
                f"Индекс {index_id} не найден",
                details={"index_id": index_id}
            )

        removed = index.remove_vectors(vector_ids)
        return removed

    def get_index_stats(self, index_id: IndexID) -> Optional[Dict[str, Any]]:
        """
        Получить статистику индекса

        Args:
            index_id: ID индекса

        Returns:
            Optional[Dict]: Статистика или None
        """
        index = self.get_index(index_id)
        if not index:
            return None

        return index.get_stats()

    def list_indexes(self) -> List[Dict[str, Any]]:
        """
        Получить список всех индексов

        Returns:
            List[Dict]: Информация по каждому индексу
        """
        return [
            {
                "index_id": idx_id,
                "type": index.__class__.__name__,
                "dimension": index.dimension,
                "metric": index.metric,
                "vector_count": index.get_vector_count(),
                "is_trained": index.is_trained(),
            }
            for idx_id, index in self.indexes.items()
        ]

    async def save_index(self, index_id: IndexID, file_path: str) -> None:
        """
        Сохранить индекс на диск

        Args:
            index_id: ID индекса
            file_path: Путь к файлу
        """
        index = self.get_index(index_id)
        if not index:
            raise ValidationError(
                f"Индекс {index_id} не найден",
                details={"index_id": index_id}
            )

        index.save(file_path)
        logger.info(f"💾 Индекс {index_id} сохранён: {file_path}")

    async def load_index(
        self,
        index_id: IndexID,
        file_path: str,
        index_type: str,
        dimension: int
    ) -> BaseIndex:
        """
        Загрузить индекс с диска

        Args:
            index_id: ID индекса
            file_path: Путь к файлу
            index_type: Тип индекса
            dimension: Размерность векторов

        Returns:
            BaseIndex: Загруженный индекс
        """
        # Создаём пустой индекс
        index = self.create_index(
            index_id=index_id,
            index_type=index_type,
            dimension=dimension
        )

        # Загружаем данные
        index.load(file_path)

        logger.info(f"📂 Индекс {index_id} загружен: {file_path}")

        return index

    def get_total_vectors(self) -> int:
        """
        Получить общее количество векторов во всех индексах

        Returns:
            int: Общее количество векторов
        """
        return sum(
            index.get_vector_count()
            for index in self.indexes.values()
        )

    def __str__(self) -> str:
        return f"SearchService(indexes={len(self.indexes)}, vectors={self.get_total_vectors()})"
