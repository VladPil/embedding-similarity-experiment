"""
Репозиторий для работы с конфигурациями моделей
"""
from typing import Optional, List
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from ..models import ModelConfigModel, ModelMetricsModel
from src.common.exceptions import DatabaseOperationError
from src.common.utils import generate_id, now_utc


class ModelConfigRepository:
    """Репозиторий для работы с конфигурациями моделей"""

    def __init__(self, session: AsyncSession):
        """
        Args:
            session: Сессия SQLAlchemy
        """
        self.session = session

    async def create(
        self,
        model_name: str,
        model_type: str,
        quantization: Optional[str] = None,
        max_memory_gb: Optional[float] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 32,
        device: str = "cuda",
        priority: int = 0,
        **kwargs
    ) -> ModelConfigModel:
        """
        Создать конфигурацию модели

        Args:
            model_name: Название модели
            model_type: Тип модели (llm/embedding)
            quantization: Квантизация (none/int8/int4)
            max_memory_gb: Максимальная память
            dimensions: Размерность векторов (для embedding)
            batch_size: Размер батча
            device: Устройство (cuda/cpu)
            priority: Приоритет
            **kwargs: Дополнительные параметры

        Returns:
            Созданная конфигурация
        """
        try:
            config = ModelConfigModel(
                id=generate_id("model_config"),
                model_name=model_name,
                model_type=model_type,
                quantization=quantization,
                max_memory_gb=max_memory_gb,
                dimensions=dimensions,
                batch_size=batch_size,
                device=device,
                priority=priority,
                **kwargs
            )

            self.session.add(config)
            await self.session.commit()
            await self.session.refresh(config)

            logger.info(f"Создана конфигурация модели: {config.id} - {model_name}")
            return config

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка создания конфигурации модели: {e}")
            raise DatabaseOperationError(
                message=f"Failed to create model config: {e}",
                details={"model_name": model_name}
            )

    async def get_by_id(self, config_id: str) -> Optional[ModelConfigModel]:
        """
        Получить конфигурацию по ID

        Args:
            config_id: ID конфигурации

        Returns:
            Конфигурация или None
        """
        try:
            stmt = select(ModelConfigModel).where(ModelConfigModel.id == config_id)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Ошибка получения конфигурации {config_id}: {e}")
            return None

    async def get_by_name_and_type(
        self,
        model_name: str,
        model_type: str
    ) -> Optional[ModelConfigModel]:
        """
        Получить конфигурацию по названию и типу

        Args:
            model_name: Название модели
            model_type: Тип модели

        Returns:
            Конфигурация или None
        """
        try:
            stmt = select(ModelConfigModel).where(
                and_(
                    ModelConfigModel.model_name == model_name,
                    ModelConfigModel.model_type == model_type
                )
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Ошибка получения конфигурации {model_name}: {e}")
            return None

    async def list(
        self,
        model_type: Optional[str] = None,
        is_enabled: Optional[bool] = None
    ) -> List[ModelConfigModel]:
        """
        Получить список конфигураций

        Args:
            model_type: Фильтр по типу
            is_enabled: Фильтр по активности

        Returns:
            Список конфигураций
        """
        try:
            where_clause = []
            if model_type:
                where_clause.append(ModelConfigModel.model_type == model_type)
            if is_enabled is not None:
                where_clause.append(ModelConfigModel.is_enabled == is_enabled)

            stmt = select(ModelConfigModel)
            if where_clause:
                stmt = stmt.where(and_(*where_clause))

            stmt = stmt.order_by(
                ModelConfigModel.priority.desc(),
                ModelConfigModel.created_at.desc()
            )

            result = await self.session.execute(stmt)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Ошибка получения списка конфигураций: {e}")
            return []

    async def update(self, config_id: str, **kwargs) -> Optional[ModelConfigModel]:
        """
        Обновить конфигурацию

        Args:
            config_id: ID конфигурации
            **kwargs: Поля для обновления

        Returns:
            Обновлённая конфигурация или None
        """
        try:
            config = await self.get_by_id(config_id)
            if not config:
                return None

            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            await self.session.commit()
            await self.session.refresh(config)

            logger.info(f"Обновлена конфигурация: {config_id}")
            return config

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка обновления конфигурации {config_id}: {e}")
            return None

    async def delete(self, config_id: str) -> bool:
        """
        Удалить конфигурацию

        Args:
            config_id: ID конфигурации

        Returns:
            True если удалена
        """
        try:
            config = await self.get_by_id(config_id)
            if not config:
                return False

            await self.session.delete(config)
            await self.session.commit()

            logger.info(f"Удалена конфигурация: {config_id}")
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка удаления конфигурации {config_id}: {e}")
            return False

    async def update_usage_stats(
        self,
        config_id: str,
        inference_time_ms: float,
        success: bool = True
    ) -> bool:
        """
        Обновить статистику использования модели

        Args:
            config_id: ID конфигурации
            inference_time_ms: Время инференса
            success: Успешность

        Returns:
            True если обновлено
        """
        try:
            config = await self.get_by_id(config_id)
            if not config:
                return False

            config.usage_count += 1
            config.last_used_at = now_utc()

            # Обновляем среднее время
            if config.avg_inference_time_ms is None:
                config.avg_inference_time_ms = inference_time_ms
            else:
                # Экспоненциальное скользящее среднее
                alpha = 0.1
                config.avg_inference_time_ms = (
                    alpha * inference_time_ms +
                    (1 - alpha) * config.avg_inference_time_ms
                )

            await self.session.commit()
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка обновления статистики: {e}")
            return False

    async def save_metrics(
        self,
        config_id: str,
        gpu_memory_used_mb: Optional[float] = None,
        gpu_utilization_percent: Optional[float] = None,
        inference_time_ms: Optional[float] = None,
        task_id: Optional[str] = None,
        task_type: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        success: bool = True,
        error: Optional[str] = None
    ) -> bool:
        """
        Сохранить метрики использования модели

        Args:
            config_id: ID конфигурации
            gpu_memory_used_mb: Использованная GPU память
            gpu_utilization_percent: Утилизация GPU
            inference_time_ms: Время инференса
            task_id: ID задачи
            task_type: Тип задачи
            input_tokens: Входные токены
            output_tokens: Выходные токены
            success: Успешность
            error: Ошибка

        Returns:
            True если сохранено
        """
        try:
            metrics = ModelMetricsModel(
                id=generate_id("metrics"),
                model_config_id=config_id,
                gpu_memory_used_mb=gpu_memory_used_mb,
                gpu_utilization_percent=gpu_utilization_percent,
                inference_time_ms=inference_time_ms,
                task_id=task_id,
                task_type=task_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=success,
                error=error
            )

            self.session.add(metrics)
            await self.session.commit()

            return True

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка сохранения метрик: {e}")
            return False

    async def get_default_llm(self) -> Optional[ModelConfigModel]:
        """
        Получить LLM модель по умолчанию

        Returns:
            Конфигурация LLM с наивысшим приоритетом или None
        """
        try:
            stmt = select(ModelConfigModel).where(
                and_(
                    ModelConfigModel.model_type == "llm",
                    ModelConfigModel.is_enabled == True
                )
            ).order_by(ModelConfigModel.priority.desc()).limit(1)

            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()

        except Exception as e:
            logger.error(f"Ошибка получения default LLM: {e}")
            return None

    async def get_default_embedding(self) -> Optional[ModelConfigModel]:
        """
        Получить embedding модель по умолчанию

        Returns:
            Конфигурация embedding с наивысшим приоритетом или None
        """
        try:
            stmt = select(ModelConfigModel).where(
                and_(
                    ModelConfigModel.model_type == "embedding",
                    ModelConfigModel.is_enabled == True
                )
            ).order_by(ModelConfigModel.priority.desc()).limit(1)

            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()

        except Exception as e:
            logger.error(f"Ошибка получения default embedding: {e}")
            return None
