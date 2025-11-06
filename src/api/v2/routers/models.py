"""
Роутер для управления моделями
"""
from fastapi import APIRouter, HTTPException, Query, Depends
from loguru import logger

from ..schemas.model_schemas import (
    ModelConfigRequest,
    ModelConfigResponse,
    ModelLoadRequest,
    ModelInstanceResponse,
    LLMGenerateRequest,
    LLMGenerateResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelPoolStatsResponse,
    GPUStatsResponse
)
from src.infrastructure.database.connection import get_db
from src.infrastructure.database.repositories import ModelConfigRepository
from ..dependencies import get_llm_service, get_embedding_service, get_model_pool, get_gpu_monitor
from src.model_management.services.llm_service import LLMService
from src.model_management.services.embedding_service import EmbeddingService
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/models", tags=["models"])


@router.post("/configs", response_model=ModelConfigResponse, status_code=201)
async def create_model_config(
    request: ModelConfigRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Создать конфигурацию модели

    - **model_name**: Название модели из HuggingFace
    - **model_type**: llm или embedding
    - **quantization**: none, int8, int4
    - **max_memory_gb**: Максимальная память (1-24 GB)
    """
    try:
        logger.info(f"Создание конфигурации модели: {request.model_name}")

        repo = ModelConfigRepository(db)

        # Проверяем что такой модели еще нет
        existing = await repo.get_by_name_and_type(request.model_name, request.model_type)
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Конфигурация для {request.model_name} ({request.model_type}) уже существует"
            )

        # Создаем конфигурацию
        config = await repo.create(
            model_name=request.model_name,
            model_type=request.model_type,
            quantization=request.quantization,
            max_memory_gb=request.max_memory_gb,
            dimensions=request.dimensions,
            batch_size=request.batch_size,
            device=request.device,
            priority=request.priority
        )

        return ModelConfigResponse(
            id=config.id,
            model_name=config.model_name,
            model_type=config.model_type,
            quantization=config.quantization,
            max_memory_gb=config.max_memory_gb,
            device=config.device,
            device_id=0,  # TODO: из config
            priority=config.priority,
            status="not_downloaded",
            is_enabled=config.is_enabled,
            memory_estimate_mb=config.max_memory_gb * 1024 if config.max_memory_gb else 0,
            created_at=config.created_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка создания конфигурации: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/configs")
async def list_model_configs(
    model_type: str = Query(None, description="Фильтр по типу"),
    db: AsyncSession = Depends(get_db)
):
    """
    Получить список конфигураций моделей

    Возвращает все зарегистрированные конфигурации
    """
    try:
        repo = ModelConfigRepository(db)
        configs = await repo.list(model_type=model_type)

        return [
            ModelConfigResponse(
                id=config.id,
                model_name=config.model_name,
                model_type=config.model_type,
                quantization=config.quantization,
                max_memory_gb=config.max_memory_gb,
                device=config.device,
                device_id=0,
                priority=config.priority,
                status="downloaded" if config.model_path else "not_downloaded",
                is_enabled=config.is_enabled,
                memory_estimate_mb=config.max_memory_gb * 1024 if config.max_memory_gb else 0,
                created_at=config.created_at
            )
            for config in configs
        ]

    except Exception as e:
        logger.error(f"Ошибка получения конфигураций: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_model(
    request: ModelLoadRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Загрузить модель в память

    - **config_id**: ID конфигурации модели

    Загружает модель на GPU/CPU и добавляет в пул
    """
    try:
        logger.info(f"Загрузка модели: {request.config_id}")

        repo = ModelConfigRepository(db)
        config = await repo.get_by_id(request.config_id)

        if not config:
            raise HTTPException(status_code=404, detail="Config not found")

        # Загружаем через model_pool
        pool = get_model_pool()
        success = await pool.load_model(
            model_name=config.model_name,
            model_type=config.model_type,
            quantization=config.quantization,
            max_memory_gb=config.max_memory_gb,
            device=config.device
        )

        if success:
            return {
                "status": "loaded",
                "config_id": request.config_id,
                "message": "Модель успешно загружена"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unload/{config_id}")
async def unload_model(
    config_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Выгрузить модель из памяти

    - **config_id**: ID конфигурации модели
    """
    try:
        logger.info(f"Выгрузка модели: {config_id}")

        repo = ModelConfigRepository(db)
        config = await repo.get_by_id(config_id)

        if not config:
            raise HTTPException(status_code=404, detail="Config not found")

        # Выгружаем через model_pool
        await model_pool.unload_model(config.model_name)

        return {
            "status": "unloaded",
            "config_id": config_id,
            "message": "Модель выгружена"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка выгрузки модели: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/instances")
async def list_loaded_models():
    """
    Получить список загруженных моделей

    Возвращает экземпляры моделей, загруженные в память
    """
    try:
        loaded = model_pool.get_loaded_models()

        return [
            ModelInstanceResponse(
                config_id=model_name,  # TODO: получать настоящий config_id
                model_name=model_name,
                model_type=info.get("type", "unknown"),
                is_busy=info.get("busy", False),
                current_task_id=None,
                total_requests=info.get("total_requests", 0),
                failed_requests=info.get("failed_requests", 0),
                success_rate=info.get("success_rate", 0.0),
                allocated_memory_mb=info.get("memory_mb", 0.0),
                peak_memory_mb=info.get("peak_memory_mb", 0.0),
                uptime_seconds=info.get("uptime_seconds", 0.0),
                loaded_at=info.get("loaded_at")
            )
            for model_name, info in loaded.items()
        ]

    except Exception as e:
        logger.error(f"Ошибка получения загруженных моделей: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm/generate", response_model=LLMGenerateResponse)
async def llm_generate(
    request: LLMGenerateRequest,
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Сгенерировать текст с помощью LLM

    - **prompt**: Промпт для генерации
    - **model_name**: Опционально, название модели
    - **max_tokens**: Максимум токенов (1-4096)
    - **temperature**: Температура (0-2)
    """
    try:
        logger.info(f"LLM генерация: {len(request.prompt)} символов")

        # Генерируем через сервис
        result = await llm_service.generate(
            prompt=request.prompt,
            model_name=request.model_name,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop_sequences=request.stop_sequences
        )

        return LLMGenerateResponse(
            generated_text=result["text"],
            model_used=result.get("model_name", request.model_name or "default"),
            tokens_generated=result.get("tokens", 0),
            generation_time_seconds=result.get("time_seconds", 0.0)
        )

    except Exception as e:
        logger.error(f"Ошибка генерации LLM: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(
    request: EmbeddingRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    Получить векторные представления текстов

    - **texts**: Список текстов (1+)
    - **model_name**: Опционально, название модели
    - **normalize**: Нормализовать векторы (для cosine similarity)
    """
    try:
        logger.info(f"Получение embeddings для {len(request.texts)} текстов")

        # Получаем embeddings через сервис
        result = await embedding_service.get_embeddings(
            texts=request.texts,
            model_name=request.model_name,
            normalize=request.normalize
        )

        return EmbeddingResponse(
            embeddings=result["embeddings"],
            model_used=result.get("model_name", request.model_name or "default"),
            dimension=result.get("dimension", 768),
            texts_count=len(request.texts)
        )

    except Exception as e:
        logger.error(f"Ошибка получения embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pool/stats", response_model=ModelPoolStatsResponse)
async def get_pool_stats():
    """
    Получить статистику пула моделей

    Возвращает информацию о загруженных моделях и использовании памяти
    """
    try:
        stats = model_pool.get_stats()

        return ModelPoolStatsResponse(
            total_models=stats.get("total_models", 0),
            llm_models=stats.get("llm_models", 0),
            embedding_models=stats.get("embedding_models", 0),
            busy_models=stats.get("busy_models", 0),
            available_models=stats.get("available_models", 0),
            total_memory_gb=stats.get("total_memory_gb", 0.0),
            max_memory_gb=stats.get("max_memory_gb", 24.0),
            memory_usage_percent=stats.get("memory_usage_percent", 0.0)
        )

    except Exception as e:
        logger.error(f"Ошибка получения статистики пула: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gpu/stats", response_model=GPUStatsResponse)
async def get_gpu_stats(device_id: int = Query(0, ge=0)):
    """
    Получить статистику GPU

    - **device_id**: ID GPU устройства (по умолчанию 0)
    """
    try:
        if not gpu_monitor.is_available():
            return GPUStatsResponse(
                device_id=device_id,
                memory_used_mb=0.0,
                memory_total_mb=0.0,
                memory_free_mb=0.0,
                memory_usage_percent=0.0,
                utilization_percent=0.0,
                temperature_celsius=0.0,
                is_available=False
            )

        stats = gpu_monitor.get_stats(device_id=device_id)

        return GPUStatsResponse(
            device_id=device_id,
            memory_used_mb=stats.memory_used_mb,
            memory_total_mb=stats.memory_total_mb,
            memory_free_mb=stats.memory_free_mb,
            memory_usage_percent=stats.memory_usage_percent,
            utilization_percent=stats.utilization_percent,
            temperature_celsius=stats.temperature_celsius,
            is_available=True
        )

    except Exception as e:
        logger.error(f"Ошибка получения статистики GPU: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/download/{config_id}")
async def download_model(
    config_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Скачать модель из HuggingFace

    - **config_id**: ID конфигурации модели

    Скачивает модель в локальный кэш
    """
    try:
        logger.info(f"Скачивание модели: {config_id}")

        repo = ModelConfigRepository(db)
        config = await repo.get_by_id(config_id)

        if not config:
            raise HTTPException(status_code=404, detail="Config not found")

        # TODO: Реализация загрузки через ModelDownloader
        # Пока заглушка
        logger.warning("ModelDownloader пока не реализован")

        return {
            "status": "downloading",
            "config_id": config_id,
            "message": "Скачивание началось (заглушка)"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка скачивания модели: {e}")
        raise HTTPException(status_code=500, detail=str(e))
