"""
Репозиторий для работы с текстами
"""
from typing import Optional, List
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from ..models import TextModel, EmbeddingCacheModel
from src.common.exceptions import DatabaseOperationError
from src.common.utils import generate_id


class TextRepository:
    """Репозиторий для работы с текстами"""

    def __init__(self, session: AsyncSession):
        """
        Args:
            session: Сессия SQLAlchemy
        """
        self.session = session

    async def create(
        self,
        title: str,
        text_type: str = "plain",
        storage_type: str = "database",
        content: Optional[str] = None,
        file_path: Optional[str] = None,
        metadata: Optional[dict] = None,
        **kwargs
    ) -> TextModel:
        """
        Создать новый текст

        Args:
            title: Название текста
            text_type: Тип текста (plain/fb2)
            storage_type: Тип хранения (database/file)
            content: Содержимое (для database)
            file_path: Путь к файлу (для file)
            metadata: Дополнительные метаданные
            **kwargs: Дополнительные поля (author, genre, year и т.д.)

        Returns:
            Созданный текст
        """
        try:
            text = TextModel(
                id=generate_id("text"),
                title=title,
                text_type=text_type,
                storage_type=storage_type,
                content=content,
                file_path=file_path,
                length=len(content) if content else 0,
                metadata=metadata or {},
                **kwargs
            )

            self.session.add(text)
            await self.session.commit()
            await self.session.refresh(text)

            logger.info(f"Создан текст: {text.id} - {title}")
            return text

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка создания текста: {e}")
            raise DatabaseOperationError(
                message=f"Failed to create text: {e}",
                details={"title": title}
            )

    async def get_by_id(self, text_id: str) -> Optional[TextModel]:
        """
        Получить текст по ID

        Args:
            text_id: ID текста

        Returns:
            Текст или None
        """
        try:
            stmt = select(TextModel).where(TextModel.id == text_id)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Ошибка получения текста {text_id}: {e}")
            return None

    async def list(
        self,
        offset: int = 0,
        limit: int = 50,
        text_type: Optional[str] = None
    ) -> tuple[List[TextModel], int]:
        """
        Получить список текстов с пагинацией

        Args:
            offset: Смещение
            limit: Количество
            text_type: Фильтр по типу

        Returns:
            (Список текстов, общее количество)
        """
        try:
            # Запрос с фильтром
            where_clause = []
            if text_type:
                where_clause.append(TextModel.text_type == text_type)

            # Общее количество
            count_stmt = select(func.count(TextModel.id))
            if where_clause:
                count_stmt = count_stmt.where(and_(*where_clause))

            count_result = await self.session.execute(count_stmt)
            total = count_result.scalar()

            # Тексты с пагинацией
            stmt = select(TextModel)
            if where_clause:
                stmt = stmt.where(and_(*where_clause))

            stmt = stmt.order_by(TextModel.created_at.desc())
            stmt = stmt.offset(offset).limit(limit)

            result = await self.session.execute(stmt)
            texts = result.scalars().all()

            return list(texts), total

        except Exception as e:
            logger.error(f"Ошибка получения списка текстов: {e}")
            return [], 0

    async def delete(self, text_id: str) -> bool:
        """
        Удалить текст

        Args:
            text_id: ID текста

        Returns:
            True если удалён
        """
        try:
            text = await self.get_by_id(text_id)
            if not text:
                return False

            await self.session.delete(text)
            await self.session.commit()

            logger.info(f"Удалён текст: {text_id}")
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка удаления текста {text_id}: {e}")
            return False

    async def update(self, text_id: str, **kwargs) -> Optional[TextModel]:
        """
        Обновить текст

        Args:
            text_id: ID текста
            **kwargs: Поля для обновления

        Returns:
            Обновлённый текст или None
        """
        try:
            text = await self.get_by_id(text_id)
            if not text:
                return None

            for key, value in kwargs.items():
                if hasattr(text, key):
                    setattr(text, key, value)

            await self.session.commit()
            await self.session.refresh(text)

            logger.info(f"Обновлён текст: {text_id}")
            return text

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка обновления текста {text_id}: {e}")
            return None

    async def get_content(self, text_id: str) -> Optional[str]:
        """
        Получить содержимое текста

        Args:
            text_id: ID текста

        Returns:
            Содержимое текста или None
        """
        text = await self.get_by_id(text_id)
        if not text:
            return None

        if text.storage_type == "database":
            return text.content
        elif text.storage_type == "file":
            # Читаем из файла
            try:
                with open(text.file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Ошибка чтения файла {text.file_path}: {e}")
                return None
        else:
            return None

    async def get_embedding(
        self,
        text_id: str,
        model_name: str
    ) -> Optional[List[float]]:
        """
        Получить embedding из L2 кэша

        Args:
            text_id: ID текста
            model_name: Название модели

        Returns:
            Вектор или None
        """
        try:
            stmt = select(EmbeddingCacheModel).where(
                and_(
                    EmbeddingCacheModel.text_id == text_id,
                    EmbeddingCacheModel.model_name == model_name
                )
            )
            result = await self.session.execute(stmt)
            cache_entry = result.scalar_one_or_none()

            if cache_entry:
                return cache_entry.embedding["vector"]
            return None

        except Exception as e:
            logger.error(f"Ошибка получения embedding: {e}")
            return None

    async def save_embedding(
        self,
        text_id: str,
        model_name: str,
        embedding: List[float]
    ) -> bool:
        """
        Сохранить embedding в L2 кэш

        Args:
            text_id: ID текста
            model_name: Название модели
            embedding: Вектор

        Returns:
            True если сохранён
        """
        try:
            cache_entry = EmbeddingCacheModel(
                text_id=text_id,
                model_name=model_name,
                embedding={"vector": embedding},
                dimensions=len(embedding)
            )

            self.session.add(cache_entry)
            await self.session.commit()

            logger.debug(f"Сохранён embedding для текста {text_id}")
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка сохранения embedding: {e}")
            return False
