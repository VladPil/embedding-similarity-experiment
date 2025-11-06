"""
Проверка работоспособности моделей
"""
from typing import Dict, List, Optional
from loguru import logger
import asyncio

from ..entities.model_config import ModelConfig
from ..entities.model_instance import ModelInstance
from ..scheduler.model_pool import ModelPool
from src.common.exceptions import ModelError


class ModelHealthChecker:
    """
    Проверка работоспособности загруженных моделей

    Выполняет тестовые запросы для проверки функционирования
    """

    def __init__(self, model_pool: ModelPool):
        """
        Инициализация checker'а

        Args:
            model_pool: Пул моделей
        """
        self.model_pool = model_pool

    async def check_model(self, config_id: str) -> Dict:
        """
        Проверить конкретную модель

        Args:
            config_id: ID конфигурации модели

        Returns:
            Dict: Результаты проверки
        """
        instance = await self.model_pool.get_instance(config_id)

        if not instance:
            return {
                "healthy": False,
                "error": "Модель не найдена в пуле",
                "config_id": config_id
            }

        logger.info(f"🔍 Проверка модели: {instance.config.model_name}")

        try:
            if instance.config.is_llm():
                result = await self._check_llm(instance)
            elif instance.config.is_embedding():
                result = await self._check_embedding(instance)
            else:
                result = {
                    "healthy": False,
                    "error": f"Неподдерживаемый тип модели: {instance.config.model_type}"
                }

            result["config_id"] = config_id
            result["model_name"] = instance.config.model_name

            if result["healthy"]:
                logger.info(f"✅ Модель {instance.config.model_name} работает корректно")
            else:
                logger.warning(
                    f"⚠️ Модель {instance.config.model_name} не прошла проверку: "
                    f"{result.get('error', 'Unknown error')}"
                )

            return result

        except Exception as e:
            logger.error(f"Ошибка проверки модели {instance.config.model_name}: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "config_id": config_id,
                "model_name": instance.config.model_name
            }

    async def _check_llm(self, instance: ModelInstance) -> Dict:
        """
        Проверить LLM модель

        Args:
            instance: Экземпляр модели

        Returns:
            Dict: Результаты проверки
        """
        # Тестовый промпт
        test_prompt = "Hello, how are you?"

        try:
            # Токенизация
            inputs = instance.tokenizer(
                test_prompt,
                return_tensors="pt",
                padding=True
            )

            # Перенос на устройство
            if instance.config.device.value == "cuda":
                inputs = {k: v.to(instance.config.device_id) for k, v in inputs.items()}

            # Генерация (короткая)
            outputs = instance.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=instance.tokenizer.pad_token_id
            )

            # Декодирование
            generated = instance.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return {
                "healthy": True,
                "test_prompt": test_prompt,
                "generated_length": len(generated),
                "output_sample": generated[:100]
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

    async def _check_embedding(self, instance: ModelInstance) -> Dict:
        """
        Проверить Embedding модель

        Args:
            instance: Экземпляр модели

        Returns:
            Dict: Результаты проверки
        """
        # Тестовые тексты
        test_texts = ["Hello world", "Test embedding"]

        try:
            # Кодирование
            embeddings = instance.model.encode(
                test_texts,
                normalize_embeddings=True,
                convert_to_numpy=True
            )

            # Проверяем размерность
            expected_dim = instance.config.dimensions
            actual_dim = embeddings.shape[1]

            if expected_dim and actual_dim != expected_dim:
                return {
                    "healthy": False,
                    "error": f"Неверная размерность: ожидалось {expected_dim}, получено {actual_dim}"
                }

            return {
                "healthy": True,
                "test_texts_count": len(test_texts),
                "embedding_dimension": actual_dim,
                "embedding_sample": embeddings[0][:5].tolist()  # Первые 5 значений
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

    async def check_all_models(self) -> Dict:
        """
        Проверить все модели в пуле

        Returns:
            Dict: Сводные результаты проверки
        """
        logger.info("🔍 Проверка всех моделей в пуле...")

        results = []

        for config_id in self.model_pool.instances.keys():
            result = await self.check_model(config_id)
            results.append(result)

        healthy_count = sum(1 for r in results if r["healthy"])
        total_count = len(results)

        summary = {
            "total_models": total_count,
            "healthy_models": healthy_count,
            "unhealthy_models": total_count - healthy_count,
            "all_healthy": healthy_count == total_count,
            "results": results
        }

        logger.info(
            f"📊 Результаты проверки: {healthy_count}/{total_count} моделей работают корректно"
        )

        return summary

    async def run_continuous_check(
        self,
        interval_seconds: int = 300,
        max_iterations: Optional[int] = None
    ) -> None:
        """
        Запустить непрерывную проверку моделей

        Args:
            interval_seconds: Интервал между проверками
            max_iterations: Максимальное количество итераций (None = бесконечно)
        """
        logger.info(
            f"🔄 Запуск непрерывной проверки моделей (интервал: {interval_seconds}s)"
        )

        iteration = 0

        while True:
            if max_iterations and iteration >= max_iterations:
                break

            # Проверяем все модели
            await self.check_all_models()

            # Ждём до следующей проверки
            await asyncio.sleep(interval_seconds)

            iteration += 1

    async def get_model_diagnostics(self, config_id: str) -> Dict:
        """
        Получить расширенную диагностику модели

        Args:
            config_id: ID конфигурации модели

        Returns:
            Dict: Диагностическая информация
        """
        instance = await self.model_pool.get_instance(config_id)

        if not instance:
            return {"error": "Модель не найдена"}

        # Базовая информация
        diagnostics = {
            "config_id": config_id,
            "model_name": instance.config.model_name,
            "model_type": instance.config.model_type.value,
            "status": {
                "is_busy": instance.is_busy,
                "current_task_id": instance.current_task_id,
                "uptime_seconds": instance.get_uptime_seconds()
            },
            "statistics": {
                "total_requests": instance.total_requests,
                "failed_requests": instance.failed_requests,
                "success_rate": instance.get_success_rate(),
                "last_used_at": instance.last_used_at.isoformat() if instance.last_used_at else None
            },
            "memory": {
                "allocated_mb": instance.allocated_memory_mb,
                "peak_mb": instance.peak_memory_mb,
                "estimated_mb": instance.config.get_memory_estimate_mb()
            },
            "config": {
                "device": instance.config.device.value,
                "device_id": instance.config.device_id,
                "quantization": instance.config.quantization.value,
                "priority": instance.config.priority
            }
        }

        # Проверка работоспособности
        health_check = await self.check_model(config_id)
        diagnostics["health_check"] = health_check

        return diagnostics

    def __str__(self) -> str:
        return f"ModelHealthChecker(pool={self.model_pool})"
