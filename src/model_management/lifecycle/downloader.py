"""
Загрузчик моделей из HuggingFace
"""
import os
from pathlib import Path
from typing import Optional, Callable
from loguru import logger
from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from src.common.exceptions import ModelError
from src.config import settings


class ModelDownloader:
    """
    Загрузчик моделей из HuggingFace Hub

    Поддерживает инкрементальную загрузку и возобновление
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Инициализация загрузчика

        Args:
            cache_dir: Директория для кэша моделей
        """
        self.cache_dir = cache_dir or settings.models_cache_dir
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """Создать директорию для кэша если не существует"""
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Директория кэша моделей: {self.cache_dir}")

    async def download_model(
        self,
        model_name: str,
        revision: str = "main",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> str:
        """
        Загрузить модель из HuggingFace

        Args:
            model_name: Название модели (например, "Qwen/Qwen2.5-3B-Instruct")
            revision: Версия модели (ветка/тег/коммит)
            progress_callback: Callback для отслеживания прогресса (current, total)

        Returns:
            str: Путь к загруженной модели

        Raises:
            ModelError: Если загрузка не удалась
        """
        logger.info(f"⬇️ Начало загрузки модели: {model_name} (revision: {revision})")

        try:
            # Загружаем всю модель (снепшот репозитория)
            model_path = snapshot_download(
                repo_id=model_name,
                revision=revision,
                cache_dir=self.cache_dir,
                resume_download=True,  # Возобновление загрузки
                local_files_only=False,
            )

            logger.info(f"✅ Модель загружена: {model_path}")
            return model_path

        except HfHubHTTPError as e:
            logger.error(f"Ошибка HTTP при загрузке модели: {e}")
            raise ModelError(
                message=f"Не удалось загрузить модель {model_name}",
                details={
                    "model_name": model_name,
                    "error": str(e),
                    "status_code": e.response.status_code if hasattr(e, 'response') else None
                }
            )

        except Exception as e:
            logger.error(f"Неожиданная ошибка при загрузке модели: {e}")
            raise ModelError(
                message=f"Ошибка загрузки модели {model_name}",
                details={"model_name": model_name, "error": str(e)}
            )

    async def download_file(
        self,
        model_name: str,
        filename: str,
        revision: str = "main"
    ) -> str:
        """
        Загрузить конкретный файл из модели

        Args:
            model_name: Название модели
            filename: Имя файла (например, "config.json")
            revision: Версия модели

        Returns:
            str: Путь к загруженному файлу

        Raises:
            ModelError: Если загрузка не удалась
        """
        logger.info(f"⬇️ Загрузка файла {filename} из {model_name}")

        try:
            file_path = hf_hub_download(
                repo_id=model_name,
                filename=filename,
                revision=revision,
                cache_dir=self.cache_dir,
                resume_download=True
            )

            logger.info(f"✅ Файл загружен: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Ошибка загрузки файла: {e}")
            raise ModelError(
                message=f"Не удалось загрузить файл {filename}",
                details={
                    "model_name": model_name,
                    "filename": filename,
                    "error": str(e)
                }
            )

    def is_model_cached(self, model_name: str, revision: str = "main") -> bool:
        """
        Проверить кэширована ли модель локально

        Args:
            model_name: Название модели
            revision: Версия модели

        Returns:
            bool: True если модель в кэше
        """
        try:
            # Пытаемся загрузить без доступа к сети
            snapshot_download(
                repo_id=model_name,
                revision=revision,
                cache_dir=self.cache_dir,
                local_files_only=True
            )
            return True

        except Exception:
            return False

    def get_model_path(self, model_name: str, revision: str = "main") -> Optional[str]:
        """
        Получить путь к кэшированной модели

        Args:
            model_name: Название модели
            revision: Версия модели

        Returns:
            Optional[str]: Путь к модели или None если не найдена
        """
        if not self.is_model_cached(model_name, revision):
            return None

        try:
            path = snapshot_download(
                repo_id=model_name,
                revision=revision,
                cache_dir=self.cache_dir,
                local_files_only=True
            )
            return path

        except Exception:
            return None

    def get_cache_size_mb(self) -> float:
        """
        Получить размер кэша моделей в МБ

        Returns:
            float: Размер в МБ
        """
        total_size = 0

        for dirpath, dirnames, filenames in os.walk(self.cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)

        return total_size / (1024 * 1024)

    def clear_cache(self) -> None:
        """
        Очистить кэш моделей

        ВНИМАНИЕ: Удалит все загруженные модели!
        """
        import shutil

        if os.path.exists(self.cache_dir):
            logger.warning(f"🗑️ Очистка кэша моделей: {self.cache_dir}")
            shutil.rmtree(self.cache_dir)
            self._ensure_cache_dir()
            logger.info("✅ Кэш очищен")

    def list_cached_models(self) -> list[str]:
        """
        Получить список кэшированных моделей

        Returns:
            list[str]: Список путей к моделям
        """
        models = []

        if not os.path.exists(self.cache_dir):
            return models

        # HuggingFace хранит модели в виде snapshots
        snapshots_dir = os.path.join(self.cache_dir, "models--*", "snapshots")

        for root, dirs, files in os.walk(self.cache_dir):
            if "snapshots" in root and "config.json" in files:
                models.append(root)

        return models

    def __str__(self) -> str:
        cache_size = self.get_cache_size_mb()
        return f"ModelDownloader(cache={self.cache_dir}, size={cache_size:.1f}MB)"
