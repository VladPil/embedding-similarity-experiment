"""
Подключение к базе данных PostgreSQL
"""
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.pool import NullPool

from src.config import settings


class DatabaseConnection:
    """Менеджер подключения к базе данных"""

    def __init__(self):
        """Инициализация подключения"""
        self.engine: AsyncEngine = create_async_engine(
            settings.database_url,
            echo=settings.DEBUG,
            poolclass=NullPool if settings.DEBUG else None,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # Проверка соединения перед использованием
        )

        self.async_session_maker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Получить асинхронную сессию

        Yields:
            AsyncSession: Сессия базы данных
        """
        async with self.async_session_maker() as session:
            try:
                yield session
            finally:
                await session.close()

    async def close(self):
        """Закрыть все соединения"""
        await self.engine.dispose()

    async def health_check(self) -> bool:
        """
        Проверка работоспособности базы данных

        Returns:
            bool: True если база доступна
        """
        try:
            async with self.async_session_maker() as session:
                await session.execute("SELECT 1")
                return True
        except Exception:
            return False


# Singleton instance
db_connection = DatabaseConnection()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency для FastAPI

    Yields:
        AsyncSession: Сессия базы данных
    """
    async for session in db_connection.get_session():
        yield session
