"""
Интерфейс репозитория
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession


# Generic тип для сущности
T = TypeVar('T')


class IRepository(ABC, Generic[T]):
    """
    Базовый интерфейс репозитория для работы с данными

    Реализует паттерн Repository для абстракции доступа к данным
    """

    def __init__(self, session: AsyncSession):
        """
        Args:
            session: Асинхронная сессия SQLAlchemy
        """
        self.session = session

    @abstractmethod
    async def create(self, entity: T) -> T:
        """
        Создать новую сущность

        Args:
            entity: Сущность для создания

        Returns:
            Созданная сущность
        """
        pass

    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """
        Получить сущность по ID

        Args:
            entity_id: ID сущности

        Returns:
            Сущность или None если не найдена
        """
        pass

    @abstractmethod
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[T]:
        """
        Получить все сущности с пагинацией и фильтрами

        Args:
            skip: Сколько пропустить
            limit: Максимальное количество
            filters: Фильтры для запроса

        Returns:
            Список сущностей
        """
        pass

    @abstractmethod
    async def update(self, entity_id: str, data: Dict[str, Any]) -> Optional[T]:
        """
        Обновить сущность

        Args:
            entity_id: ID сущности
            data: Данные для обновления

        Returns:
            Обновлённая сущность или None
        """
        pass

    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """
        Удалить сущность

        Args:
            entity_id: ID сущности

        Returns:
            True если удалена, False если не найдена
        """
        pass

    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Подсчитать количество сущностей

        Args:
            filters: Фильтры для запроса

        Returns:
            Количество сущностей
        """
        pass

    @abstractmethod
    async def exists(self, entity_id: str) -> bool:
        """
        Проверить существование сущности

        Args:
            entity_id: ID сущности

        Returns:
            True если существует
        """
        pass
