"""
Конфигурация pytest
"""
import pytest
import asyncio
from typing import AsyncGenerator
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from src.api.v2.app import app
from src.infrastructure.database.base import Base
from src.infrastructure.database.connection import get_db
from src.config import settings


# Тестовая база данных
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"


@pytest.fixture(scope="session")
def event_loop():
    """Создать event loop для сессии"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Создать тестовый engine"""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        future=True
    )

    # Создаем таблицы
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Удаляем таблицы
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def test_db(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Создать тестовую сессию БД"""
    async_session_maker = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    async with async_session_maker() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def client(test_db) -> AsyncGenerator[AsyncClient, None]:
    """Создать тестовый HTTP клиент"""

    # Override dependency
    async def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def fb2_file_path_1() -> str:
    """Путь к первому FB2 файлу"""
    return "tests/data/evolution_hakayna.fb2"


@pytest.fixture
def fb2_file_path_2() -> str:
    """Путь ко второму FB2 файлу"""
    return "tests/data/zona_dremeyet.fb2"
