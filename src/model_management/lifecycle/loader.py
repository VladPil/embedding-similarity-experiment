"""
Загрузчик моделей в память
"""
from typing import Optional, Tuple, Any
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

from ..entities.model_config import ModelConfig
from ..entities.model_instance import ModelInstance
from ..scheduler.model_pool import ModelPool
from src.common.exceptions import ModelError, GPUMemoryError
from src.common.types import ModelType, QuantizationType, DeviceType


class ModelLoader:
    """
    Загрузчик моделей в память с поддержкой квантизации

    Загружает LLM и Embedding модели на GPU/CPU
    """

    def __init__(self, model_pool: ModelPool):
        """
        Инициализация загрузчика

        Args:
            model_pool: Пул моделей
        """
        self.model_pool = model_pool

    async def load_model(self, config: ModelConfig) -> ModelInstance:
        """
        Загрузить модель в память

        Args:
            config: Конфигурация модели

        Returns:
            ModelInstance: Загруженный экземпляр модели

        Raises:
            ModelError: Если загрузка не удалась
            GPUMemoryError: Если недостаточно памяти
        """
        logger.info(f"🔄 Загрузка модели: {config.model_name} ({config.model_type.value})")

        try:
            if config.is_llm():
                model, tokenizer = await self._load_llm(config)
            elif config.is_embedding():
                model, tokenizer = await self._load_embedding(config), None
            else:
                raise ModelError(
                    message=f"Неподдерживаемый тип модели: {config.model_type}",
                    details={"model_type": config.model_type.value}
                )

            # Добавляем в пул
            instance = await self.model_pool.load_model(config, model, tokenizer)

            logger.info(
                f"✅ Модель загружена: {config.model_name} "
                f"(память: {config.get_memory_estimate_mb():.0f}MB)"
            )

            return instance

        except GPUMemoryError:
            raise

        except Exception as e:
            logger.error(f"Ошибка загрузки модели {config.model_name}: {e}")
            raise ModelError(
                message=f"Не удалось загрузить модель {config.model_name}",
                details={"error": str(e), "config_id": config.id}
            )

    async def _load_llm(self, config: ModelConfig) -> Tuple[Any, Any]:
        """
        Загрузить LLM модель

        Args:
            config: Конфигурация модели

        Returns:
            Tuple[Any, Any]: (model, tokenizer)
        """
        # Определяем устройство
        device = self._get_device(config)

        # Настройка квантизации
        quantization_config = self._get_quantization_config(config)

        # Параметры загрузки
        load_kwargs = {
            "pretrained_model_name_or_path": config.model_path or config.model_name,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "device_map": "auto" if device == "cuda" else None,
            "trust_remote_code": True,
        }

        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config

        # Загружаем модель
        logger.info(f"Загрузка LLM на {device}...")
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        # Загружаем токенизатор
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path or config.model_name,
            trust_remote_code=True
        )

        # Настраиваем токенизатор
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    async def _load_embedding(self, config: ModelConfig) -> Any:
        """
        Загрузить Embedding модель

        Args:
            config: Конфигурация модели

        Returns:
            Any: Модель SentenceTransformer
        """
        # Определяем устройство
        device = self._get_device(config)

        logger.info(f"Загрузка Embedding модели на {device}...")

        # Загружаем через SentenceTransformer
        model = SentenceTransformer(
            model_name_or_path=config.model_path or config.model_name,
            device=device,
            trust_remote_code=True
        )

        # Обновляем размерность в конфиге если не задана
        if config.dimensions is None:
            config.dimensions = model.get_sentence_embedding_dimension()

        return model

    def _get_device(self, config: ModelConfig) -> str:
        """
        Получить устройство для загрузки

        Args:
            config: Конфигурация модели

        Returns:
            str: Устройство ("cuda:0", "cpu")
        """
        if config.device == DeviceType.CUDA:
            if torch.cuda.is_available():
                return f"cuda:{config.device_id}"
            else:
                logger.warning("CUDA недоступна, используется CPU")
                return "cpu"
        else:
            return "cpu"

    def _get_quantization_config(
        self,
        config: ModelConfig
    ) -> Optional[BitsAndBytesConfig]:
        """
        Получить конфигурацию квантизации

        Args:
            config: Конфигурация модели

        Returns:
            Optional[BitsAndBytesConfig]: Конфигурация или None
        """
        if config.quantization == QuantizationType.NONE:
            return None

        if config.quantization == QuantizationType.INT8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )

        elif config.quantization == QuantizationType.INT4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        else:
            logger.warning(f"Неподдерживаемая квантизация: {config.quantization}")
            return None

    async def unload_model(self, config_id: str) -> bool:
        """
        Выгрузить модель из памяти

        Args:
            config_id: ID конфигурации модели

        Returns:
            bool: True если успешно выгружена
        """
        success = await self.model_pool.unload_model(config_id)

        if success:
            # Очищаем CUDA кэш
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return success

    async def reload_model(self, config: ModelConfig) -> ModelInstance:
        """
        Перезагрузить модель (выгрузить и загрузить заново)

        Args:
            config: Конфигурация модели

        Returns:
            ModelInstance: Новый экземпляр модели
        """
        logger.info(f"🔄 Перезагрузка модели: {config.model_name}")

        # Выгружаем если загружена
        await self.unload_model(config.id)

        # Загружаем заново
        return await self.load_model(config)

    def get_memory_stats(self) -> dict:
        """
        Получить статистику памяти

        Returns:
            dict: Статистика по GPU/CPU памяти
        """
        stats = {}

        # GPU память
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)

                stats[f"gpu_{i}"] = {
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "total_gb": total,
                    "free_gb": total - allocated,
                    "usage_percent": (allocated / total) * 100
                }

        # CPU память (через psutil если доступен)
        try:
            import psutil
            mem = psutil.virtual_memory()
            stats["cpu"] = {
                "used_gb": mem.used / (1024 ** 3),
                "available_gb": mem.available / (1024 ** 3),
                "total_gb": mem.total / (1024 ** 3),
                "usage_percent": mem.percent
            }
        except ImportError:
            pass

        return stats

    def __str__(self) -> str:
        return f"ModelLoader(pool={self.model_pool})"
