"""
FastStream брокер для Redis
"""
from faststream import FastStream
from faststream.redis import RedisBroker
from loguru import logger

from src.config import settings


# Создание брокера Redis
broker = RedisBroker(
    url=settings.redis_url,
    apply_types=True,  # Автоматическая валидация типов
)

# Создание FastStream приложения
app = FastStream(
    broker,
    title="Embedding Similarity Queue",
    description="Очередь задач для анализа текстов",
)


@app.on_startup
async def on_startup():
    """Событие при запуске"""
    logger.info("🚀 FastStream запущен")
    logger.info(f"📡 Подключено к Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")


@app.on_shutdown
async def on_shutdown():
    """Событие при остановке"""
    logger.info("🛑 FastStream остановлен")
