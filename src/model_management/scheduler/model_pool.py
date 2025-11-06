"""
Пул моделей для управления загруженными моделями
"""
import asyncio
from typing import Dict, Optional, List
from loguru import logger

from ..entities.model_config import ModelConfig
from ..entities.model_instance import ModelInstance
from src.common.exceptions import ModelError, GPUMemoryError


class ModelPool:
    """
    Пул загруженных моделей с балансировкой нагрузки

    Управляет загрузкой, выгрузкой и распределением задач между моделями
    """

    def __init__(
        self,
        max_concurrent_llm: int = 2,
        max_concurrent_embedding: int = 4,
        max_gpu_memory_gb: float = 20.0
    ):
        """
        Инициализация пула

        Args:
            max_concurrent_llm: Максимум одновременных LLM задач
            max_concurrent_embedding: Максимум одновременных embedding задач
            max_gpu_memory_gb: Максимальная память GPU
        """
        self.max_concurrent_llm = max_concurrent_llm
        self.max_concurrent_embedding = max_concurrent_embedding
        self.max_gpu_memory_gb = max_gpu_memory_gb

        # Словарь загруженных моделей: config_id -> ModelInstance
        self.instances: Dict[str, ModelInstance] = {}

        # Очереди задач
        self.llm_queue: asyncio.Queue = asyncio.Queue()
        self.embedding_queue: asyncio.Queue = asyncio.Queue()

        # Locks для синхронизации
        self._lock = asyncio.Lock()

    async def load_model(
        self,
        config: ModelConfig,
        model: any,
        tokenizer: Optional[any] = None
    ) -> ModelInstance:
        """
        Загрузить модель в пул

        Args:
            config: Конфигурация модели
            model: Объект модели
            tokenizer: Токенизатор (опционально)

        Returns:
            ModelInstance: Экземпляр модели

        Raises:
            GPUMemoryError: Если недостаточно памяти
        """
        async with self._lock:
            # Проверка памяти
            current_memory = self.get_total_memory_usage()
            required_memory = config.get_memory_estimate_mb() / 1024  # В GB

            if current_memory + required_memory > self.max_gpu_memory_gb:
                # Попытка освободить память
                freed = await self._free_memory(required_memory)
                if not freed:
                    raise GPUMemoryError(
                        message=f"Insufficient GPU memory: need {required_memory:.1f}GB, "
                                f"available {self.max_gpu_memory_gb - current_memory:.1f}GB",
                        details={
                            "required_gb": required_memory,
                            "available_gb": self.max_gpu_memory_gb - current_memory,
                            "model_name": config.model_name
                        }
                    )

            # Создаём экземпляр
            instance = ModelInstance(
                config=config,
                model=model,
                tokenizer=tokenizer,
            )

            # Добавляем в пул
            self.instances[config.id] = instance

            logger.info(
                f"✅ Модель загружена в пул: {config.model_name} "
                f"(память: {required_memory:.1f}GB)"
            )

            return instance

    async def unload_model(self, config_id: str) -> bool:
        """
        Выгрузить модель из пула

        Args:
            config_id: ID конфигурации модели

        Returns:
            bool: True если выгружена
        """
        async with self._lock:
            if config_id not in self.instances:
                return False

            instance = self.instances[config_id]

            # Проверяем что модель не занята
            if instance.is_busy:
                logger.warning(
                    f"Попытка выгрузить занятую модель: {instance.config.model_name}"
                )
                return False

            # Удаляем из пула
            del self.instances[config_id]

            # Освобождаем память
            del instance.model
            if instance.tokenizer:
                del instance.tokenizer

            logger.info(f"🗑️ Модель выгружена: {instance.config.model_name}")

            return True

    async def get_instance(self, config_id: str) -> Optional[ModelInstance]:
        """
        Получить экземпляр модели

        Args:
            config_id: ID конфигурации

        Returns:
            Optional[ModelInstance]: Экземпляр или None
        """
        return self.instances.get(config_id)

    async def acquire_model(
        self,
        config_id: str,
        task_id: str,
        timeout: float = 30.0
    ) -> Optional[ModelInstance]:
        """
        Захватить модель для выполнения задачи

        Args:
            config_id: ID конфигурации модели
            task_id: ID задачи
            timeout: Таймаут ожидания в секундах

        Returns:
            Optional[ModelInstance]: Захваченный экземпляр или None
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            # Проверяем таймаут
            if asyncio.get_event_loop().time() - start_time > timeout:
                logger.warning(
                    f"Таймаут ожидания модели {config_id} для задачи {task_id}"
                )
                return None

            # Пытаемся захватить модель
            instance = self.instances.get(config_id)
            if instance and instance.acquire(task_id):
                return instance

            # Ждём немного перед следующей попыткой
            await asyncio.sleep(0.1)

    async def release_model(self, config_id: str) -> None:
        """
        Освободить модель после выполнения задачи

        Args:
            config_id: ID конфигурации модели
        """
        instance = self.instances.get(config_id)
        if instance:
            instance.release()

    def get_available_models(self, model_type: Optional[str] = None) -> List[ModelInstance]:
        """
        Получить список доступных моделей

        Args:
            model_type: Тип модели (llm/embedding) или None для всех

        Returns:
            List[ModelInstance]: Список доступных моделей
        """
        result = []
        for instance in self.instances.values():
            if not instance.is_busy:
                if model_type is None or instance.config.model_type.value == model_type:
                    result.append(instance)
        return result

    def get_total_memory_usage(self) -> float:
        """
        Получить общее потребление памяти в GB

        Returns:
            float: Память в GB
        """
        total_mb = sum(
            instance.config.get_memory_estimate_mb()
            for instance in self.instances.values()
        )
        return total_mb / 1024

    async def _free_memory(self, required_gb: float) -> bool:
        """
        Попытка освободить память выгрузкой моделей

        Args:
            required_gb: Требуемая память в GB

        Returns:
            bool: True если удалось освободить
        """
        # Сортируем модели по последнему использованию
        sorted_instances = sorted(
            self.instances.values(),
            key=lambda x: x.last_used_at
        )

        freed_gb = 0.0

        for instance in sorted_instances:
            if instance.is_busy:
                continue

            # Выгружаем модель
            config_id = instance.config.id
            if await self.unload_model(config_id):
                freed_gb += instance.config.get_memory_estimate_mb() / 1024
                logger.info(f"Освобождено {freed_gb:.1f}GB памяти")

                if freed_gb >= required_gb:
                    return True

        return freed_gb >= required_gb

    def get_stats(self) -> dict:
        """
        Получить статистику пула

        Returns:
            dict: Статистика
        """
        llm_models = [i for i in self.instances.values() if i.config.is_llm()]
        embedding_models = [i for i in self.instances.values() if i.config.is_embedding()]

        return {
            "total_models": len(self.instances),
            "llm_models": len(llm_models),
            "embedding_models": len(embedding_models),
            "busy_models": sum(1 for i in self.instances.values() if i.is_busy),
            "available_models": sum(1 for i in self.instances.values() if not i.is_busy),
            "total_memory_gb": self.get_total_memory_usage(),
            "max_memory_gb": self.max_gpu_memory_gb,
            "memory_usage_percent": (self.get_total_memory_usage() / self.max_gpu_memory_gb) * 100,
        }

    async def health_check(self) -> dict:
        """
        Проверка здоровья пула

        Returns:
            dict: Результаты проверки
        """
        stats = self.get_stats()

        # Проверки
        issues = []

        # Проверка памяти
        if stats["memory_usage_percent"] > 90:
            issues.append("Критическое использование памяти (>90%)")

        # Проверка успешности запросов
        for instance in self.instances.values():
            success_rate = instance.get_success_rate()
            if success_rate < 0.8:
                issues.append(
                    f"Низкий процент успеха для {instance.config.model_name}: {success_rate:.0%}"
                )

        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "stats": stats,
        }

    def __str__(self) -> str:
        stats = self.get_stats()
        return (
            f"ModelPool(models={stats['total_models']}, "
            f"memory={stats['total_memory_gb']:.1f}/{stats['max_memory_gb']:.1f}GB)"
        )

    def __repr__(self) -> str:
        return self.__str__()
