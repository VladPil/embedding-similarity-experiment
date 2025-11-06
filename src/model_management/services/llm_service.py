"""
Сервис для работы с LLM моделями
"""
from typing import Optional, List, Dict, Any
from loguru import logger

from ..entities.model_config import ModelConfig
from ..entities.model_instance import ModelInstance
from ..scheduler.model_pool import ModelPool
from src.common.exceptions import ModelError
from src.common.types import ModelType


class LLMService:
    """
    Сервис для работы с языковыми моделями

    Предоставляет высокоуровневый интерфейс для генерации текста
    """

    def __init__(self, model_pool: ModelPool):
        """
        Инициализация сервиса

        Args:
            model_pool: Пул моделей
        """
        self.model_pool = model_pool

    async def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Сгенерировать текст на основе промпта

        Args:
            prompt: Текст промпта
            model_name: Название модели (если None - используется дефолтная)
            max_tokens: Максимальное количество токенов
            temperature: Температура сэмплирования (0-1)
            top_p: Nucleus sampling параметр
            stop_sequences: Последовательности для остановки генерации
            **kwargs: Дополнительные параметры

        Returns:
            str: Сгенерированный текст

        Raises:
            ModelError: Если модель недоступна или ошибка генерации
        """
        # Найти подходящую модель
        config_id = await self._find_llm_model(model_name)
        if not config_id:
            raise ModelError(
                message=f"LLM модель не найдена: {model_name}",
                details={"model_name": model_name}
            )

        # Захватить модель
        task_id = f"llm_gen_{id(prompt)}"
        instance = await self.model_pool.acquire_model(config_id, task_id)

        if not instance:
            raise ModelError(
                message="Не удалось захватить модель для генерации",
                details={"config_id": config_id}
            )

        try:
            # Подготовка промпта
            inputs = instance.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )

            # Перенос на GPU если нужно
            if instance.config.device.value == "cuda":
                inputs = {k: v.to(instance.config.device_id) for k, v in inputs.items()}

            # Генерация
            outputs = instance.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=instance.tokenizer.pad_token_id,
                eos_token_id=instance.tokenizer.eos_token_id,
                **kwargs
            )

            # Декодирование
            generated_text = instance.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Обрезка по stop sequences
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
                        break

            # Статистика
            instance.record_request(success=True)
            instance.config.update_usage_stats(
                inference_time_ms=0.0,  # TODO: измерить реальное время
                tokens_processed=outputs.shape[1]
            )

            logger.info(
                f"✅ LLM генерация завершена: {len(generated_text)} символов, "
                f"модель: {instance.config.model_name}"
            )

            return generated_text.strip()

        except Exception as e:
            instance.record_request(success=False)
            logger.error(f"Ошибка генерации LLM: {e}")
            raise ModelError(
                message="Ошибка генерации текста",
                details={"error": str(e), "model": instance.config.model_name}
            )

        finally:
            # Освобождаем модель
            await self.model_pool.release_model(config_id)

    async def generate_batch(
        self,
        prompts: List[str],
        model_name: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """
        Сгенерировать тексты для батча промптов

        Args:
            prompts: Список промптов
            model_name: Название модели
            **kwargs: Параметры генерации

        Returns:
            List[str]: Список сгенерированных текстов
        """
        results = []

        # TODO: Оптимизировать для реального батчинга
        for prompt in prompts:
            result = await self.generate(prompt, model_name=model_name, **kwargs)
            results.append(result)

        return results

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Чат-генерация в формате OpenAI

        Args:
            messages: Список сообщений [{"role": "user", "content": "..."}]
            model_name: Название модели
            **kwargs: Параметры генерации

        Returns:
            str: Ответ модели
        """
        # Конвертируем в промпт
        prompt = self._format_chat_messages(messages)

        return await self.generate(prompt, model_name=model_name, **kwargs)

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Форматировать чат-сообщения в промпт

        Args:
            messages: Список сообщений

        Returns:
            str: Отформатированный промпт
        """
        # Простой формат для Qwen
        formatted = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                formatted.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                formatted.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        # Добавляем префикс для ответа
        formatted.append("<|im_start|>assistant\n")

        return "\n".join(formatted)

    async def _find_llm_model(self, model_name: Optional[str] = None) -> Optional[str]:
        """
        Найти ID конфигурации LLM модели

        Args:
            model_name: Название модели или None для дефолтной

        Returns:
            Optional[str]: ID конфигурации или None
        """
        # Получаем все LLM модели
        available = self.model_pool.get_available_models(model_type="llm")

        if not available:
            return None

        # Если указано имя - ищем по имени
        if model_name:
            for instance in available:
                if instance.config.model_name == model_name:
                    return instance.config.id

        # Иначе берём первую доступную с наивысшим приоритетом
        best = max(available, key=lambda x: x.config.priority)
        return best.config.id

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Получить список доступных LLM моделей

        Returns:
            List[Dict]: Список моделей с информацией
        """
        available = self.model_pool.get_available_models(model_type="llm")

        return [
            {
                "config_id": inst.config.id,
                "model_name": inst.config.model_name,
                "quantization": inst.config.quantization.value,
                "memory_mb": inst.config.get_memory_estimate_mb(),
                "priority": inst.config.priority,
                "is_busy": inst.is_busy,
                "total_requests": inst.total_requests,
                "success_rate": inst.get_success_rate(),
            }
            for inst in available
        ]

    async def estimate_tokens(self, text: str, model_name: Optional[str] = None) -> int:
        """
        Оценить количество токенов в тексте

        Args:
            text: Текст для оценки
            model_name: Название модели

        Returns:
            int: Примерное количество токенов
        """
        # Простая оценка: 1 токен ≈ 4 символа для английского, ≈ 2 для русского
        # TODO: Использовать реальный токенизатор
        return len(text) // 3

    def __str__(self) -> str:
        return f"LLMService(pool={self.model_pool})"
