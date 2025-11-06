"""
Интерфейс для кэшируемых объектов
"""
from abc import ABC, abstractmethod
from typing import Optional, Any


class ICacheable(ABC):
    """
    Интерфейс для объектов, которые могут быть кэшированы

    Реализующие классы должны предоставить методы сериализации/десериализации
    """

    @abstractmethod
    def get_cache_key(self) -> str:
        """
        Получить уникальный ключ для кэша

        Returns:
            Строковый ключ
        """
        pass

    @abstractmethod
    def to_cache(self) -> Any:
        """
        Сериализовать объект для кэша

        Returns:
            Сериализованные данные (dict, list, str, etc.)
        """
        pass

    @classmethod
    @abstractmethod
    def from_cache(cls, data: Any) -> "ICacheable":
        """
        Десериализовать объект из кэша

        Args:
            data: Данные из кэша

        Returns:
            Восстановленный объект
        """
        pass

    def get_cache_ttl(self) -> Optional[int]:
        """
        Получить TTL для кэша в секундах

        Returns:
            TTL в секундах или None для бесконечного хранения
        """
        return None
