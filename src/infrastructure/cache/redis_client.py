"""
Redis клиент для кэширования
"""
import pickle
import json
from typing import Any, Optional, List
from redis import asyncio as aioredis
from loguru import logger

from src.config import settings
from src.common.exceptions import CacheConnectionError, CacheOperationError


class RedisClient:
    """Асинхронный клиент для работы с Redis"""

    def __init__(self):
        """Инициализация клиента"""
        self.redis: Optional[aioredis.Redis] = None
        self._connected = False

    async def connect(self) -> None:
        """Подключение к Redis"""
        try:
            self.redis = await aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=False,  # Для бинарных данных (pickle)
                max_connections=50,
            )
            # Проверка подключения
            await self.redis.ping()
            self._connected = True
            logger.info("✅ Подключено к Redis")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к Redis: {e}")
            raise CacheConnectionError(
                message=f"Failed to connect to Redis: {e}",
                details={"url": settings.redis_url}
            )

    async def disconnect(self) -> None:
        """Отключение от Redis"""
        if self.redis:
            await self.redis.close()
            self._connected = False
            logger.info("Redis отключен")

    async def health_check(self) -> bool:
        """
        Проверка работоспособности Redis

        Returns:
            bool: True если Redis доступен
        """
        try:
            if not self.redis:
                return False
            await self.redis.ping()
            return True
        except Exception:
            return False

    # ===== СТРОКОВЫЕ ОПЕРАЦИИ =====

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        use_pickle: bool = False
    ) -> bool:
        """
        Установить значение в кэш

        Args:
            key: Ключ
            value: Значение
            ttl: Время жизни в секундах (None = бесконечно)
            use_pickle: Использовать pickle для сериализации (для сложных объектов)

        Returns:
            bool: True если успешно
        """
        try:
            # Сериализация
            if use_pickle:
                serialized = pickle.dumps(value)
            else:
                serialized = json.dumps(value)

            # Установка с TTL
            if ttl:
                await self.redis.setex(key, ttl, serialized)
            else:
                await self.redis.set(key, serialized)

            return True
        except Exception as e:
            logger.error(f"Ошибка установки значения в Redis: {e}")
            raise CacheOperationError(
                message=f"Failed to set value in cache: {e}",
                details={"key": key}
            )

    async def get(
        self,
        key: str,
        use_pickle: bool = False
    ) -> Optional[Any]:
        """
        Получить значение из кэша

        Args:
            key: Ключ
            use_pickle: Использовать pickle для десериализации

        Returns:
            Значение или None если не найдено
        """
        try:
            value = await self.redis.get(key)
            if value is None:
                return None

            # Десериализация
            if use_pickle:
                return pickle.loads(value)
            else:
                return json.loads(value)
        except Exception as e:
            logger.error(f"Ошибка получения значения из Redis: {e}")
            return None

    async def delete(self, key: str) -> bool:
        """
        Удалить значение из кэша

        Args:
            key: Ключ

        Returns:
            bool: True если удалено
        """
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Ошибка удаления значения из Redis: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """
        Проверить существование ключа

        Args:
            key: Ключ

        Returns:
            bool: True если существует
        """
        try:
            result = await self.redis.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Ошибка проверки существования ключа: {e}")
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """
        Установить TTL для ключа

        Args:
            key: Ключ
            ttl: Время жизни в секундах

        Returns:
            bool: True если успешно
        """
        try:
            return await self.redis.expire(key, ttl)
        except Exception as e:
            logger.error(f"Ошибка установки TTL: {e}")
            return False

    async def get_ttl(self, key: str) -> Optional[int]:
        """
        Получить TTL ключа

        Args:
            key: Ключ

        Returns:
            TTL в секундах или None если ключа нет или TTL не установлен
        """
        try:
            ttl = await self.redis.ttl(key)
            return ttl if ttl > 0 else None
        except Exception as e:
            logger.error(f"Ошибка получения TTL: {e}")
            return None

    # ===== РАБОТА С МНОЖЕСТВЕННЫМИ КЛЮЧАМИ =====

    async def mget(self, keys: List[str], use_pickle: bool = False) -> List[Optional[Any]]:
        """
        Получить несколько значений

        Args:
            keys: Список ключей
            use_pickle: Использовать pickle

        Returns:
            Список значений (None для отсутствующих)
        """
        try:
            values = await self.redis.mget(keys)
            results = []
            for value in values:
                if value is None:
                    results.append(None)
                else:
                    if use_pickle:
                        results.append(pickle.loads(value))
                    else:
                        results.append(json.loads(value))
            return results
        except Exception as e:
            logger.error(f"Ошибка получения множественных значений: {e}")
            return [None] * len(keys)

    async def delete_pattern(self, pattern: str) -> int:
        """
        Удалить все ключи по паттерну

        Args:
            pattern: Паттерн (например, "embedding:*")

        Returns:
            Количество удалённых ключей
        """
        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Ошибка удаления по паттерну: {e}")
            return 0

    # ===== HASH ОПЕРАЦИИ =====

    async def hset(self, name: str, key: str, value: Any) -> bool:
        """
        Установить значение в hash

        Args:
            name: Имя hash
            key: Ключ в hash
            value: Значение

        Returns:
            bool: True если успешно
        """
        try:
            serialized = json.dumps(value)
            await self.redis.hset(name, key, serialized)
            return True
        except Exception as e:
            logger.error(f"Ошибка установки значения в hash: {e}")
            return False

    async def hget(self, name: str, key: str) -> Optional[Any]:
        """
        Получить значение из hash

        Args:
            name: Имя hash
            key: Ключ в hash

        Returns:
            Значение или None
        """
        try:
            value = await self.redis.hget(name, key)
            if value is None:
                return None
            return json.loads(value)
        except Exception as e:
            logger.error(f"Ошибка получения значения из hash: {e}")
            return None

    async def hgetall(self, name: str) -> dict:
        """
        Получить все значения из hash

        Args:
            name: Имя hash

        Returns:
            Словарь значений
        """
        try:
            data = await self.redis.hgetall(name)
            result = {}
            for key, value in data.items():
                try:
                    result[key.decode('utf-8')] = json.loads(value)
                except:
                    result[key.decode('utf-8')] = value.decode('utf-8')
            return result
        except Exception as e:
            logger.error(f"Ошибка получения hash: {e}")
            return {}

    # ===== СПИСОК ОПЕРАЦИИ =====

    async def lpush(self, key: str, *values: Any) -> int:
        """
        Добавить значения в начало списка

        Args:
            key: Ключ
            values: Значения

        Returns:
            Длина списка после операции
        """
        try:
            serialized = [json.dumps(v) for v in values]
            return await self.redis.lpush(key, *serialized)
        except Exception as e:
            logger.error(f"Ошибка lpush: {e}")
            return 0

    async def rpush(self, key: str, *values: Any) -> int:
        """
        Добавить значения в конец списка

        Args:
            key: Ключ
            values: Значения

        Returns:
            Длина списка после операции
        """
        try:
            serialized = [json.dumps(v) for v in values]
            return await self.redis.rpush(key, *serialized)
        except Exception as e:
            logger.error(f"Ошибка rpush: {e}")
            return 0

    async def lrange(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """
        Получить диапазон элементов списка

        Args:
            key: Ключ
            start: Начальный индекс
            end: Конечный индекс (-1 = до конца)

        Returns:
            Список элементов
        """
        try:
            values = await self.redis.lrange(key, start, end)
            return [json.loads(v) for v in values]
        except Exception as e:
            logger.error(f"Ошибка lrange: {e}")
            return []

    # ===== МНОЖЕСТВО ОПЕРАЦИИ =====

    async def sadd(self, key: str, *members: Any) -> int:
        """
        Добавить элементы в множество

        Args:
            key: Ключ
            members: Элементы

        Returns:
            Количество добавленных элементов
        """
        try:
            serialized = [json.dumps(m) for m in members]
            return await self.redis.sadd(key, *serialized)
        except Exception as e:
            logger.error(f"Ошибка sadd: {e}")
            return 0

    async def smembers(self, key: str) -> set:
        """
        Получить все элементы множества

        Args:
            key: Ключ

        Returns:
            Множество элементов
        """
        try:
            members = await self.redis.smembers(key)
            return {json.loads(m) for m in members}
        except Exception as e:
            logger.error(f"Ошибка smembers: {e}")
            return set()

    # ===== ИНФОРМАЦИЯ =====

    async def get_info(self) -> dict:
        """
        Получить информацию о Redis

        Returns:
            Словарь с информацией
        """
        try:
            info = await self.redis.info()
            return {
                "version": info.get("redis_version", "unknown"),
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_keys": await self.redis.dbsize(),
            }
        except Exception as e:
            logger.error(f"Ошибка получения info: {e}")
            return {}

    async def clear_all(self) -> bool:
        """
        Очистить всю базу Redis (ОСТОРОЖНО!)

        Returns:
            bool: True если успешно
        """
        try:
            await self.redis.flushdb()
            logger.warning("⚠️ Redis база очищена")
            return True
        except Exception as e:
            logger.error(f"Ошибка очистки Redis: {e}")
            return False


# Singleton instance
redis_client = RedisClient()
