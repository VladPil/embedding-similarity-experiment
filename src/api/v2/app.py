"""
Главное приложение FastAPI для API v2
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .routers import texts, analysis, models, monitoring, export

# Создание приложения
app = FastAPI(
    title="Embedding Similarity Experiment API v2",
    description="API для анализа текстов с помощью LLM и embeddings",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware для фронтенда
# По запросу пользователя - разрешаем все домены (для development)
# В production рекомендуется настроить конкретные origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешены все домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Подключение роутеров
app.include_router(texts.router)
app.include_router(analysis.router)
app.include_router(models.router)
app.include_router(monitoring.router)
app.include_router(export.router)


@app.on_event("startup")
async def startup_event():
    """
    Инициализация при запуске приложения
    """
    logger.info("🚀 Запуск API v2...")

    # Подключение к инфраструктуре
    from src.infrastructure.database.connection import db_connection
    from src.infrastructure.cache.redis_client import redis_client

    try:
        # Проверка PostgreSQL
        is_db_healthy = await db_connection.health_check()
        if is_db_healthy:
            logger.info("✅ PostgreSQL подключена")
        else:
            logger.warning("⚠️ PostgreSQL недоступна")

        # Подключение к Redis
        await redis_client.connect()
        is_redis_healthy = await redis_client.health_check()
        if is_redis_healthy:
            logger.info("✅ Redis подключен")
        else:
            logger.warning("⚠️ Redis недоступен")

        # Проверка GPU
        from src.model_management.gpu_monitor import gpu_monitor
        gpu_available = gpu_monitor.is_available()
        if gpu_available:
            stats = gpu_monitor.get_stats(device_id=0)
            logger.info(f"✅ GPU доступна: {stats.memory_free_mb:.0f}MB свободно")
        else:
            logger.warning("⚠️ GPU недоступна, будет использоваться CPU")

        # Запуск MetricsService в фоне (опционально)
        # from src.monitoring.services.metrics_service import metrics_service
        # await metrics_service.start_background_collection()

        logger.info("🎉 API v2 успешно запущено")

    except Exception as e:
        logger.error(f"❌ Ошибка инициализации: {e}")
        # Продолжаем работу даже если что-то не запустилось


@app.on_event("shutdown")
async def shutdown_event():
    """
    Очистка ресурсов при остановке
    """
    logger.info("🛑 Остановка API v2...")

    from src.infrastructure.database.connection import db_connection
    from src.infrastructure.cache.redis_client import redis_client

    try:
        # Закрытие соединений с БД
        await db_connection.close()
        logger.info("✅ PostgreSQL отключена")

        # Закрытие Redis
        await redis_client.disconnect()
        logger.info("✅ Redis отключен")

        # Остановка MetricsService
        # from src.monitoring.services.metrics_service import metrics_service
        # await metrics_service.stop_background_collection()

        logger.info("✅ API v2 остановлено")

    except Exception as e:
        logger.error(f"❌ Ошибка при остановке: {e}")


@app.get("/")
async def root():
    """
    Корневой эндпоинт
    """
    return {
        "name": "Embedding Similarity Experiment API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/monitoring/health"
    }


@app.get("/ping")
async def ping():
    """
    Простая проверка доступности API
    """
    return {"status": "ok"}
