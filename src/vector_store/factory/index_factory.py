"""
Фабрика для создания FAISS индексов
"""
from typing import Optional, Dict, Any
from loguru import logger

from ..entities.base_index import BaseIndex
from ..indexes.flat_index import FlatIndex
from ..indexes.ivf_flat_index import IVFFlatIndex
from ..indexes.hnsw_index import HNSWIndex
from ..indexes.ivf_pq_index import IVFPQIndex
from src.common.types import IndexID, IndexType
from src.common.exceptions import ValidationError


class IndexFactory:
    """
    Фабрика для создания различных типов FAISS индексов

    Поддерживаемые типы:
    - FLAT: Точный поиск
    - IVF_FLAT: Кластеризация
    - HNSW: Граф
    - IVF_PQ: Кластеризация + квантизация
    """

    @staticmethod
    def create_index(
        index_type: str,
        index_id: IndexID,
        dimension: int,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseIndex:
        """
        Создать индекс заданного типа

        Args:
            index_type: Тип индекса (flat, ivf_flat, hnsw, ivf_pq)
            index_id: ID индекса
            dimension: Размерность векторов
            config: Дополнительная конфигурация

        Returns:
            BaseIndex: Созданный индекс

        Raises:
            ValidationError: Если тип индекса неподдерживаемый
        """
        config = config or {}
        index_type = index_type.lower()

        logger.info(f"🏭 Создание индекса: тип={index_type}, dimension={dimension}")

        if index_type == "flat":
            return IndexFactory._create_flat(index_id, dimension, config)

        elif index_type == "ivf_flat":
            return IndexFactory._create_ivf_flat(index_id, dimension, config)

        elif index_type == "hnsw":
            return IndexFactory._create_hnsw(index_id, dimension, config)

        elif index_type == "ivf_pq":
            return IndexFactory._create_ivf_pq(index_id, dimension, config)

        else:
            raise ValidationError(
                f"Неподдерживаемый тип индекса: {index_type}",
                details={
                    "supported_types": ["flat", "ivf_flat", "hnsw", "ivf_pq"]
                }
            )

    @staticmethod
    def _create_flat(
        index_id: IndexID,
        dimension: int,
        config: Dict[str, Any]
    ) -> FlatIndex:
        """Создать Flat индекс"""
        return FlatIndex(
            index_id=index_id,
            dimension=dimension,
            metric=config.get("metric", "cosine"),
            use_gpu=config.get("use_gpu", False)
        )

    @staticmethod
    def _create_ivf_flat(
        index_id: IndexID,
        dimension: int,
        config: Dict[str, Any]
    ) -> IVFFlatIndex:
        """Создать IVF Flat индекс"""
        # Автоматический подбор nlist если не задан
        nlist = config.get("nlist")
        if nlist is None:
            # Эвристика: sqrt(N), но для начала используем разумное значение
            nlist = 100

        return IVFFlatIndex(
            index_id=index_id,
            dimension=dimension,
            nlist=nlist,
            nprobe=config.get("nprobe", max(1, nlist // 10)),
            metric=config.get("metric", "cosine"),
            use_gpu=config.get("use_gpu", False)
        )

    @staticmethod
    def _create_hnsw(
        index_id: IndexID,
        dimension: int,
        config: Dict[str, Any]
    ) -> HNSWIndex:
        """Создать HNSW индекс"""
        return HNSWIndex(
            index_id=index_id,
            dimension=dimension,
            M=config.get("M", 32),
            efConstruction=config.get("efConstruction", 40),
            efSearch=config.get("efSearch", 16),
            metric=config.get("metric", "cosine"),
            use_gpu=config.get("use_gpu", False)
        )

    @staticmethod
    def _create_ivf_pq(
        index_id: IndexID,
        dimension: int,
        config: Dict[str, Any]
    ) -> IVFPQIndex:
        """Создать IVF+PQ индекс"""
        # Автоматический подбор m если не задан
        m = config.get("m")
        if m is None:
            # Эвристика: dimension / 8, но с проверкой делимости
            for candidate in [64, 32, 16, 8, 4]:
                if dimension % candidate == 0:
                    m = candidate
                    break
            if m is None:
                m = 1

        # Автоматический подбор nlist
        nlist = config.get("nlist", 100)

        return IVFPQIndex(
            index_id=index_id,
            dimension=dimension,
            nlist=nlist,
            m=m,
            nbits=config.get("nbits", 8),
            nprobe=config.get("nprobe", max(1, nlist // 10)),
            metric=config.get("metric", "cosine"),
            use_gpu=config.get("use_gpu", False)
        )

    @staticmethod
    def get_recommended_type(
        vector_count: int,
        dimension: int,
        priority: str = "balanced"
    ) -> str:
        """
        Получить рекомендованный тип индекса для данных

        Args:
            vector_count: Ожидаемое количество векторов
            dimension: Размерность векторов
            priority: Приоритет (speed, accuracy, memory, balanced)

        Returns:
            str: Рекомендованный тип индекса
        """
        if priority == "accuracy":
            # Точность превыше всего
            if vector_count < 100_000:
                return "flat"
            else:
                return "ivf_flat"

        elif priority == "speed":
            # Скорость превыше всего
            if vector_count < 50_000:
                return "flat"
            else:
                return "hnsw"

        elif priority == "memory":
            # Экономия памяти
            if vector_count < 10_000:
                return "flat"
            elif vector_count < 1_000_000:
                return "ivf_flat"
            else:
                return "ivf_pq"

        else:  # balanced
            # Баланс между всеми параметрами
            if vector_count < 10_000:
                return "flat"
            elif vector_count < 100_000:
                return "ivf_flat"
            elif vector_count < 1_000_000:
                return "hnsw"
            else:
                return "ivf_pq"

    @staticmethod
    def get_recommended_config(
        index_type: str,
        vector_count: int,
        dimension: int
    ) -> Dict[str, Any]:
        """
        Получить рекомендованную конфигурацию для типа индекса

        Args:
            index_type: Тип индекса
            vector_count: Количество векторов
            dimension: Размерность

        Returns:
            Dict: Рекомендованная конфигурация
        """
        config = {
            "metric": "cosine",
            "use_gpu": True  # По умолчанию пытаемся использовать GPU
        }

        if index_type == "ivf_flat":
            # nlist = sqrt(N)
            nlist = int(np.sqrt(vector_count)) if vector_count > 0 else 100
            nlist = max(10, min(nlist, 10000))  # Ограничиваем разумными пределами

            config.update({
                "nlist": nlist,
                "nprobe": max(1, nlist // 10)
            })

        elif index_type == "hnsw":
            config.update({
                "M": 32,
                "efConstruction": 40,
                "efSearch": 16
            })

        elif index_type == "ivf_pq":
            nlist = int(np.sqrt(vector_count)) if vector_count > 0 else 100
            nlist = max(10, min(nlist, 10000))

            # Подбираем m
            m = 8
            for candidate in [64, 32, 16, 8, 4]:
                if dimension % candidate == 0:
                    m = candidate
                    break

            config.update({
                "nlist": nlist,
                "nprobe": max(1, nlist // 10),
                "m": m,
                "nbits": 8
            })

        return config

    @staticmethod
    def create_recommended_index(
        index_id: IndexID,
        dimension: int,
        vector_count: int,
        priority: str = "balanced"
    ) -> BaseIndex:
        """
        Создать рекомендованный индекс на основе данных

        Args:
            index_id: ID индекса
            dimension: Размерность векторов
            vector_count: Ожидаемое количество векторов
            priority: Приоритет (speed, accuracy, memory, balanced)

        Returns:
            BaseIndex: Созданный индекс
        """
        # Определяем тип
        index_type = IndexFactory.get_recommended_type(
            vector_count=vector_count,
            dimension=dimension,
            priority=priority
        )

        # Получаем конфигурацию
        config = IndexFactory.get_recommended_config(
            index_type=index_type,
            vector_count=vector_count,
            dimension=dimension
        )

        logger.info(
            f"📊 Рекомендация: тип={index_type}, "
            f"vectors={vector_count}, dim={dimension}, priority={priority}"
        )

        # Создаём индекс
        return IndexFactory.create_index(
            index_type=index_type,
            index_id=index_id,
            dimension=dimension,
            config=config
        )


# Для удобства импорта
import numpy as np
