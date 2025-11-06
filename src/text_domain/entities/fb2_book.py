"""
FB2 книга
"""
import aiofiles
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path

from .base_text import BaseText
from src.common.exceptions import FileReadError, FB2ParseError


@dataclass
class FB2Book(BaseText):
    """
    FB2 книга с расширенными метаданными

    FB2 (FictionBook) - XML формат для электронных книг
    """

    # Путь к FB2 файлу (всегда файловое хранение)
    file_path: str

    # Метаданные книги
    author: Optional[str] = None
    genre: List[str] = field(default_factory=list)
    year: Optional[int] = None
    publisher: Optional[str] = None
    isbn: Optional[str] = None
    series: Optional[str] = None
    series_number: Optional[int] = None

    # Структура книги
    annotation: Optional[str] = None
    cover_image: Optional[str] = None  # Путь к изображению обложки

    async def get_content(self) -> str:
        """
        Получить текстовое содержимое книги

        Парсит FB2 файл и извлекает чистый текст

        Returns:
            str: Текстовое содержимое

        Raises:
            FB2ParseError: Если не удалось распарсить FB2
        """
        # Если уже загружен в кэш
        if self._content is not None:
            return self._content

        try:
            # Импорт здесь чтобы избежать циклических зависимостей
            from ..services.fb2_parser_service import FB2ParserService

            parser = FB2ParserService()
            self._content = await parser.extract_text(self.file_path)
            self._length = len(self._content)

            return self._content

        except Exception as e:
            raise FB2ParseError(
                message=f"Failed to parse FB2 file: {e}",
                details={
                    "text_id": self.id,
                    "file_path": self.file_path,
                    "error": str(e)
                }
            )

    async def get_annotation(self) -> Optional[str]:
        """
        Получить аннотацию книги

        Returns:
            Optional[str]: Аннотация или None
        """
        if self.annotation is not None:
            return self.annotation

        try:
            from ..services.fb2_parser_service import FB2ParserService
            parser = FB2ParserService()
            metadata = await parser.extract_metadata(self.file_path)
            self.annotation = metadata.get("annotation")
            return self.annotation
        except Exception:
            return None

    async def get_cover(self) -> Optional[bytes]:
        """
        Получить изображение обложки

        Returns:
            Optional[bytes]: Данные изображения или None
        """
        if not self.cover_image:
            return None

        try:
            cover_path = Path(self.cover_image)
            if not cover_path.exists():
                return None

            async with aiofiles.open(cover_path, 'rb') as f:
                return await f.read()
        except Exception:
            return None

    def get_text_type(self) -> str:
        """
        Получить тип текста

        Returns:
            str: 'fb2'
        """
        return "fb2"

    def get_metadata_schema(self) -> Dict[str, str]:
        """
        Получить схему метаданных для FB2 книги

        Returns:
            Dict[str, str]: Схема метаданных
        """
        return {
            "author": "string",
            "genre": "array",
            "year": "integer",
            "publisher": "string",
            "isbn": "string",
            "series": "string",
            "series_number": "integer",
            "annotation": "string",
            "cover_image": "string",
        }

    def to_dict(self) -> Dict:
        """
        Сериализация в словарь

        Returns:
            Dict: Данные книги
        """
        data = super().to_dict()
        data.update({
            "file_path": self.file_path,
            "author": self.author,
            "genre": self.genre,
            "year": self.year,
            "publisher": self.publisher,
            "isbn": self.isbn,
            "series": self.series,
            "series_number": self.series_number,
            "has_annotation": self.annotation is not None,
            "has_cover": self.cover_image is not None,
        })
        return data

    def get_author_name(self) -> str:
        """
        Получить имя автора или "Неизвестный автор"

        Returns:
            str: Имя автора
        """
        return self.author or "Неизвестный автор"

    def get_genre_list(self) -> List[str]:
        """
        Получить список жанров

        Returns:
            List[str]: Список жанров
        """
        return self.genre or []

    def get_primary_genre(self) -> Optional[str]:
        """
        Получить основной жанр (первый в списке)

        Returns:
            Optional[str]: Основной жанр или None
        """
        return self.genre[0] if self.genre else None

    def get_full_series_title(self) -> Optional[str]:
        """
        Получить полное название серии с номером

        Returns:
            Optional[str]: "Серия #N" или None
        """
        if not self.series:
            return None

        if self.series_number:
            return f"{self.series} #{self.series_number}"

        return self.series

    @classmethod
    def create_from_file(
        cls,
        text_id: str,
        title: str,
        file_path: str,
        author: Optional[str] = None,
        genre: Optional[List[str]] = None,
        year: Optional[int] = None,
        publisher: Optional[str] = None,
        language: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> "FB2Book":
        """
        Создать FB2Book из файла

        Args:
            text_id: ID текста
            title: Название книги
            file_path: Путь к FB2 файлу
            author: Автор
            genre: Жанры
            year: Год издания
            publisher: Издательство
            language: Язык
            metadata: Дополнительные метаданные

        Returns:
            FB2Book: Созданный объект
        """
        return cls(
            id=text_id,
            title=title,
            file_path=file_path,
            author=author,
            genre=genre or [],
            year=year,
            publisher=publisher,
            language=language,
            metadata=metadata or {},
        )
