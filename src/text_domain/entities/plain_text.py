"""
Обычный текст (plain text)
"""
import aiofiles
from dataclasses import dataclass, field
from typing import Optional, Dict
from pathlib import Path

from .base_text import BaseText
from src.common.types import TextStorageType
from src.common.exceptions import FileReadError, TextProcessingError


@dataclass
class PlainText(BaseText):
    """
    Обычный текст

    Может храниться как в базе данных (для коротких текстов),
    так и в файловой системе (для длинных текстов)
    """

    # Тип хранения
    storage_type: TextStorageType = TextStorageType.DATABASE

    # Хранение в БД (для коротких текстов < 1000 символов)
    content: Optional[str] = None

    # Хранение в файле (для длинных текстов > 1000 символов)
    file_path: Optional[str] = None

    async def get_content(self) -> str:
        """
        Получить содержимое текста

        Загружает из БД или файла в зависимости от storage_type

        Returns:
            str: Содержимое текста

        Raises:
            TextProcessingError: Если не удалось загрузить текст
        """
        # Если уже загружен в кэш
        if self._content is not None:
            return self._content

        # Загрузка из БД
        if self.storage_type == TextStorageType.DATABASE:
            if self.content is None:
                raise TextProcessingError(
                    message=f"Text content is None for database storage (id: {self.id})",
                    details={"text_id": self.id, "storage_type": "database"}
                )
            self._content = self.content
            self._length = len(self._content)
            return self._content

        # Загрузка из файла
        if self.storage_type == TextStorageType.FILE:
            if self.file_path is None:
                raise TextProcessingError(
                    message=f"File path is None for file storage (id: {self.id})",
                    details={"text_id": self.id, "storage_type": "file"}
                )

            try:
                file_path = Path(self.file_path)
                if not file_path.exists():
                    raise FileReadError(str(file_path))

                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    self._content = await f.read()
                    self._length = len(self._content)
                    return self._content

            except Exception as e:
                raise FileReadError(
                    message=f"Failed to read text from file: {e}",
                    details={"text_id": self.id, "file_path": self.file_path}
                )

        raise TextProcessingError(
            message=f"Unknown storage type: {self.storage_type}",
            details={"text_id": self.id, "storage_type": self.storage_type}
        )

    def get_text_type(self) -> str:
        """
        Получить тип текста

        Returns:
            str: 'plain'
        """
        return "plain"

    def get_metadata_schema(self) -> Dict[str, str]:
        """
        Получить схему метаданных для обычного текста

        Returns:
            Dict[str, str]: Схема метаданных
        """
        return {
            "storage_type": "string",
            "length": "integer",
            "encoding": "string",
            "source": "string",
        }

    def to_dict(self) -> Dict:
        """
        Сериализация в словарь

        Returns:
            Dict: Данные текста
        """
        data = super().to_dict()
        data.update({
            "storage_type": self.storage_type.value,
            "file_path": self.file_path,
            "has_content": self.content is not None,
        })
        return data

    @classmethod
    def create_from_string(
        cls,
        text_id: str,
        title: str,
        content: str,
        language: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> "PlainText":
        """
        Создать PlainText из строки

        Args:
            text_id: ID текста
            title: Название
            content: Содержимое
            language: Язык
            metadata: Метаданные

        Returns:
            PlainText: Созданный объект
        """
        storage_type = (
            TextStorageType.DATABASE
            if len(content) < 1000
            else TextStorageType.FILE
        )

        return cls(
            id=text_id,
            title=title,
            content=content if storage_type == TextStorageType.DATABASE else None,
            storage_type=storage_type,
            language=language,
            metadata=metadata or {},
        )

    @classmethod
    def create_from_file(
        cls,
        text_id: str,
        title: str,
        file_path: str,
        language: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> "PlainText":
        """
        Создать PlainText из файла

        Args:
            text_id: ID текста
            title: Название
            file_path: Путь к файлу
            language: Язык
            metadata: Метаданные

        Returns:
            PlainText: Созданный объект
        """
        return cls(
            id=text_id,
            title=title,
            file_path=file_path,
            storage_type=TextStorageType.FILE,
            language=language,
            metadata=metadata or {},
        )
