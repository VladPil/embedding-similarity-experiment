"""
Сервис для работы с Embedding моделями
"""
from typing import Optional, List, Dict, Any
import numpy as np
from loguru import logger

from ..entities.model_config import ModelConfig
from ..entities.model_instance import ModelInstance
from ..scheduler.model_pool import ModelPool
from src.common.exceptions import ModelError
from src.common.types import ModelType


class EmbeddingService:
    """
    Сервис для получения векторных представлений текстов

    Поддерживает батчевую обработку и кэширование
    """

    def __init__(self, model_pool: ModelPool):
        """
        Инициализация сервиса

        Args:
            model_pool: Пул моделей
        """
        self.model_pool = model_pool

    async def encode(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Получить векторные представления для текстов

        Args:
            texts: Список текстов
            model_name: Название модели (если None - дефолтная)
            normalize: Нормализовать векторы (для cosine similarity)
            show_progress: Показывать прогресс

        Returns:
            np.ndarray: Матрица эмбеддингов (N x D)

        Raises:
            ModelError: Если модель недоступна или ошибка кодирования
        """
        if not texts:
            return np.array([])

        # Найти подходящую модель
        config_id = await self._find_embedding_model(model_name)
        if not config_id:
            raise ModelError(
                message=f"Embedding модель не найдена: {model_name}",
                details={"model_name": model_name}
            )

        # Захватить модель
        task_id = f"embed_{id(texts)}"
        instance = await self.model_pool.acquire_model(config_id, task_id)

        if not instance:
            raise ModelError(
                message="Не удалось захватить модель для кодирования",
                details={"config_id": config_id}
            )

        try:
            # Батчевая обработка
            embeddings = []
            batch_size = instance.config.batch_size

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                # Кодирование батча
                batch_embeddings = instance.model.encode(
                    batch,
                    normalize_embeddings=normalize,
                    show_progress_bar=show_progress and i == 0,
                    convert_to_numpy=True
                )

                embeddings.append(batch_embeddings)

            # Объединяем батчи
            all_embeddings = np.vstack(embeddings)

            # Статистика
            instance.record_request(success=True)
            instance.config.update_usage_stats(
                inference_time_ms=0.0,  # TODO: измерить реальное время
                tokens_processed=sum(len(t.split()) for t in texts)
            )

            logger.info(
                f"✅ Embedding завершён: {len(texts)} текстов → "
                f"{all_embeddings.shape}, модель: {instance.config.model_name}"
            )

            return all_embeddings

        except Exception as e:
            instance.record_request(success=False)
            logger.error(f"Ошибка кодирования embedding: {e}")
            raise ModelError(
                message="Ошибка получения векторных представлений",
                details={"error": str(e), "model": instance.config.model_name}
            )

        finally:
            # Освобождаем модель
            await self.model_pool.release_model(config_id)

    async def encode_single(
        self,
        text: str,
        model_name: Optional[str] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Получить векторное представление для одного текста

        Args:
            text: Текст
            model_name: Название модели
            normalize: Нормализовать вектор

        Returns:
            np.ndarray: Вектор эмбеддинга (D,)
        """
        embeddings = await self.encode(
            texts=[text],
            model_name=model_name,
            normalize=normalize
        )

        return embeddings[0]

    async def compute_similarity(
        self,
        text1: str,
        text2: str,
        model_name: Optional[str] = None,
        metric: str = "cosine"
    ) -> float:
        """
        Вычислить схожесть между двумя текстами

        Args:
            text1: Первый текст
            text2: Второй текст
            model_name: Название модели
            metric: Метрика схожести (cosine, euclidean, dot)

        Returns:
            float: Схожесть (0-1 для cosine)
        """
        # Получаем эмбеддинги
        embeddings = await self.encode(
            texts=[text1, text2],
            model_name=model_name,
            normalize=(metric == "cosine")
        )

        emb1, emb2 = embeddings[0], embeddings[1]

        if metric == "cosine":
            # Для нормализованных векторов: dot product = cosine similarity
            similarity = np.dot(emb1, emb2)
        elif metric == "euclidean":
            # Евклидово расстояние (инвертируем для схожести)
            distance = np.linalg.norm(emb1 - emb2)
            similarity = 1.0 / (1.0 + distance)
        elif metric == "dot":
            similarity = np.dot(emb1, emb2)
        else:
            raise ValueError(f"Неизвестная метрика: {metric}")

        return float(similarity)

    async def compute_similarity_matrix(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Вычислить матрицу схожести между всеми текстами

        Args:
            texts: Список текстов
            model_name: Название модели
            metric: Метрика схожести

        Returns:
            np.ndarray: Матрица схожести (N x N)
        """
        # Получаем эмбеддинги
        embeddings = await self.encode(
            texts=texts,
            model_name=model_name,
            normalize=(metric == "cosine")
        )

        if metric == "cosine":
            # Матричное умножение для всех пар
            similarity_matrix = np.dot(embeddings, embeddings.T)
        elif metric == "euclidean":
            # Попарные евклидовы расстояния
            from scipy.spatial.distance import cdist
            distances = cdist(embeddings, embeddings, metric='euclidean')
            similarity_matrix = 1.0 / (1.0 + distances)
        elif metric == "dot":
            similarity_matrix = np.dot(embeddings, embeddings.T)
        else:
            raise ValueError(f"Неизвестная метрика: {metric}")

        return similarity_matrix

    async def get_dimensions(self, model_name: Optional[str] = None) -> int:
        """
        Получить размерность векторов для модели

        Args:
            model_name: Название модели

        Returns:
            int: Размерность векторов
        """
        config_id = await self._find_embedding_model(model_name)
        if not config_id:
            raise ModelError(
                message=f"Embedding модель не найдена: {model_name}",
                details={"model_name": model_name}
            )

        instance = await self.model_pool.get_instance(config_id)
        if not instance:
            raise ModelError(
                message="Модель не загружена",
                details={"config_id": config_id}
            )

        # Получаем размерность из конфигурации или модели
        if instance.config.dimensions:
            return instance.config.dimensions

        # Пробуем получить из модели
        if hasattr(instance.model, 'get_sentence_embedding_dimension'):
            return instance.model.get_sentence_embedding_dimension()

        raise ModelError(
            message="Не удалось определить размерность векторов",
            details={"model": instance.config.model_name}
        )

    async def _find_embedding_model(
        self,
        model_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Найти ID конфигурации Embedding модели

        Args:
            model_name: Название модели или None для дефолтной

        Returns:
            Optional[str]: ID конфигурации или None
        """
        # Получаем все Embedding модели
        available = self.model_pool.get_available_models(model_type="embedding")

        if not available:
            return None

        # Если указано имя - ищем по имени
        if model_name:
            for instance in available:
                if instance.config.model_name == model_name:
                    return instance.config.id

        # Иначе берём первую доступную с наивысшим приоритетом
        best = max(available, key=lambda x: x.config.priority)
        return best.config.id

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Получить список доступных Embedding моделей

        Returns:
            List[Dict]: Список моделей с информацией
        """
        available = self.model_pool.get_available_models(model_type="embedding")

        result = []
        for inst in available:
            info = {
                "config_id": inst.config.id,
                "model_name": inst.config.model_name,
                "dimensions": inst.config.dimensions,
                "batch_size": inst.config.batch_size,
                "memory_mb": inst.config.get_memory_estimate_mb(),
                "priority": inst.config.priority,
                "is_busy": inst.is_busy,
                "total_requests": inst.total_requests,
                "success_rate": inst.get_success_rate(),
            }
            result.append(info)

        return result

    def __str__(self) -> str:
        return f"EmbeddingService(pool={self.model_pool})"
