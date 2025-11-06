"""
Менеджер кэша с двухуровневой стратегией (L1: Redis, L2: PostgreSQL)
"""
from typing import Optional, Any, List
from loguru import logger
import numpy as np

from .redis_client import redis_client
from src.config import settings
from src.common.exceptions import CacheOperationError


class CacheManager:
    """
    Менеджер кэша с двухуровневой стратегией

    L1 (Redis): Быстрый, волатильный, с TTL
    L2 (PostgreSQL): Медленный, персистентный
    """

    def __init__(self):
        """Инициализация менеджера"""
        self.redis = redis_client

    # ===== КЭШ EMBEDDINGS =====

    def _get_embedding_cache_key(self, text_id: str, model_name: str) -> str:
        """
        Получить ключ для кэша embeddings

        Args:
            text_id: ID текста
            model_name: Название модели

        Returns:
            Ключ кэша
        """
        return f"embedding:{text_id}:{model_name}"

    async def get_embedding(
        self,
        text_id: str,
        model_name: str
    ) -> Optional[np.ndarray]:
        """
        Получить embedding из кэша (L1)

        Args:
            text_id: ID текста
            model_name: Название модели

        Returns:
            Numpy array или None если не найден
        """
        try:
            key = self._get_embedding_cache_key(text_id, model_name)
            data = await self.redis.get(key, use_pickle=True)

            if data is not None:
                logger.debug(f"✅ Embedding найден в L1 кэше: {text_id}")
                return data

            return None
        except Exception as e:
            logger.error(f"Ошибка получения embedding из кэша: {e}")
            return None

    async def set_embedding(
        self,
        text_id: str,
        model_name: str,
        embedding: np.ndarray,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Сохранить embedding в кэш (L1)

        Args:
            text_id: ID текста
            model_name: Название модели
            embedding: Numpy array
            ttl: Время жизни (по умолчанию из настроек)

        Returns:
            bool: True если успешно
        """
        try:
            key = self._get_embedding_cache_key(text_id, model_name)
            ttl = ttl or settings.REDIS_TTL_EMBEDDINGS

            success = await self.redis.set(key, embedding, ttl=ttl, use_pickle=True)

            if success:
                logger.debug(f"✅ Embedding сохранён в L1 кэш: {text_id}")

            return success
        except Exception as e:
            logger.error(f"Ошибка сохранения embedding в кэш: {e}")
            return False

    async def delete_embedding(self, text_id: str, model_name: str) -> bool:
        """
        Удалить embedding из кэша

        Args:
            text_id: ID текста
            model_name: Название модели

        Returns:
            bool: True если удалён
        """
        key = self._get_embedding_cache_key(text_id, model_name)
        return await self.redis.delete(key)

    async def delete_all_embeddings_for_text(self, text_id: str) -> int:
        """
        Удалить все embeddings для текста (все модели)

        Args:
            text_id: ID текста

        Returns:
            Количество удалённых ключей
        """
        pattern = f"embedding:{text_id}:*"
        count = await self.redis.delete_pattern(pattern)
        logger.info(f"Удалено {count} embeddings для текста {text_id}")
        return count

    # ===== КЭШ РЕЗУЛЬТАТОВ АНАЛИЗА =====

    def _get_analysis_cache_key(
        self,
        analyzer_name: str,
        text_id: str,
        mode: str = "full_text"
    ) -> str:
        """
        Получить ключ для кэша результатов анализа

        Args:
            analyzer_name: Название анализатора
            text_id: ID текста
            mode: Режим анализа

        Returns:
            Ключ кэша
        """
        return f"analysis:{analyzer_name}:{text_id}:{mode}"

    async def get_analysis_result(
        self,
        analyzer_name: str,
        text_id: str,
        mode: str = "full_text"
    ) -> Optional[dict]:
        """
        Получить результат анализа из кэша

        Args:
            analyzer_name: Название анализатора
            text_id: ID текста
            mode: Режим анализа

        Returns:
            Результат или None
        """
        try:
            key = self._get_analysis_cache_key(analyzer_name, text_id, mode)
            data = await self.redis.get(key, use_pickle=False)

            if data is not None:
                logger.debug(f"✅ Результат анализа найден в кэше: {analyzer_name}/{text_id}")

            return data
        except Exception as e:
            logger.error(f"Ошибка получения результата анализа из кэша: {e}")
            return None

    async def set_analysis_result(
        self,
        analyzer_name: str,
        text_id: str,
        result: dict,
        mode: str = "full_text",
        ttl: Optional[int] = None
    ) -> bool:
        """
        Сохранить результат анализа в кэш

        Args:
            analyzer_name: Название анализатора
            text_id: ID текста
            result: Результат анализа
            mode: Режим анализа
            ttl: Время жизни

        Returns:
            bool: True если успешно
        """
        try:
            key = self._get_analysis_cache_key(analyzer_name, text_id, mode)
            ttl = ttl or settings.REDIS_TTL_ANALYSIS

            success = await self.redis.set(key, result, ttl=ttl, use_pickle=False)

            if success:
                logger.debug(f"✅ Результат анализа сохранён в кэш: {analyzer_name}/{text_id}")

            return success
        except Exception as e:
            logger.error(f"Ошибка сохранения результата анализа в кэш: {e}")
            return False

    async def delete_analysis_result(
        self,
        analyzer_name: str,
        text_id: str,
        mode: str = "full_text"
    ) -> bool:
        """
        Удалить результат анализа из кэша

        Args:
            analyzer_name: Название анализатора
            text_id: ID текста
            mode: Режим анализа

        Returns:
            bool: True если удалён
        """
        key = self._get_analysis_cache_key(analyzer_name, text_id, mode)
        return await self.redis.delete(key)

    # ===== КЭШ СЕССИЙ =====

    def _get_session_cache_key(self, session_id: str) -> str:
        """Ключ для кэша сессии"""
        return f"session:{session_id}"

    async def get_session(self, session_id: str) -> Optional[dict]:
        """Получить сессию из кэша"""
        key = self._get_session_cache_key(session_id)
        return await self.redis.get(key, use_pickle=False)

    async def set_session(
        self,
        session_id: str,
        session_data: dict,
        ttl: int = 3600
    ) -> bool:
        """Сохранить сессию в кэш"""
        key = self._get_session_cache_key(session_id)
        return await self.redis.set(key, session_data, ttl=ttl, use_pickle=False)

    async def delete_session(self, session_id: str) -> bool:
        """Удалить сессию из кэша"""
        key = self._get_session_cache_key(session_id)
        return await self.redis.delete(key)

    # ===== ПРОГРЕСС ЗАДАЧ =====

    def _get_task_progress_key(self, task_id: str) -> str:
        """Ключ для прогресса задачи"""
        return f"task_progress:{task_id}"

    async def set_task_progress(
        self,
        task_id: str,
        progress: int,
        message: str
    ) -> bool:
        """
        Установить прогресс задачи

        Args:
            task_id: ID задачи
            progress: Прогресс (0-100)
            message: Сообщение о текущем шаге

        Returns:
            bool: True если успешно
        """
        key = self._get_task_progress_key(task_id)
        data = {
            "progress": progress,
            "message": message,
            "timestamp": str(np.datetime64('now'))
        }
        return await self.redis.set(key, data, ttl=7200, use_pickle=False)

    async def get_task_progress(self, task_id: str) -> Optional[dict]:
        """Получить прогресс задачи"""
        key = self._get_task_progress_key(task_id)
        return await self.redis.get(key, use_pickle=False)

    # ===== ОБЩИЕ ОПЕРАЦИИ =====

    async def clear_all(self) -> bool:
        """
        Очистить весь кэш (ОСТОРОЖНО!)

        Returns:
            bool: True если успешно
        """
        logger.warning("⚠️ Очистка всего кэша Redis")
        return await self.redis.clear_all()

    async def clear_pattern(self, pattern: str) -> int:
        """
        Очистить кэш по паттерну

        Args:
            pattern: Паттерн (например, "embedding:*")

        Returns:
            Количество удалённых ключей
        """
        count = await self.redis.delete_pattern(pattern)
        logger.info(f"Очищено {count} ключей по паттерну: {pattern}")
        return count

    async def get_stats(self) -> dict:
        """
        Получить статистику кэша

        Returns:
            Словарь со статистикой
        """
        info = await self.redis.get_info()
        return {
            "redis_info": info,
            "is_connected": await self.redis.health_check(),
        }


# Singleton instance
cache_manager = CacheManager()
