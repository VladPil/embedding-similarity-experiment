"""
Repository for Text model.
Handles text storage strategy (DB vs File).
"""

import os
import hashlib
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc
import aiofiles
import logging

from server.repositories.base import BaseRepository
from server.db.models import Text, TextStorageType
from server.config import settings

logger = logging.getLogger(__name__)


class TextRepository(BaseRepository[Text]):
    """
    Repository for text management.
    Implements smart storage strategy:
    - Short texts (<1000 chars) in database
    - Long texts in files with DB reference
    """

    STORAGE_THRESHOLD = 1000  # Characters threshold for DB storage

    def __init__(self, db: Session):
        """Initialize text repository."""
        super().__init__(Text, db)
        self.data_dir = settings.data_texts_dir
        os.makedirs(self.data_dir, exist_ok=True)

    async def create_text(
        self,
        title: str,
        content: str,
        language: Optional[str] = None
    ) -> Text:
        """
        Create text with smart storage strategy.

        Args:
            title: Text title
            content: Text content
            language: Text language (optional)

        Returns:
            Created Text object
        """
        # Generate unique ID
        text_id = self._generate_id(content)

        # Check if already exists
        existing = self.get(text_id)
        if existing:
            logger.info(f"Text already exists: {text_id}")
            return existing

        # Calculate metadata
        length = len(content)
        lines = content.count('\n') + 1

        # Determine storage strategy
        if length <= self.STORAGE_THRESHOLD:
            # Store in database
            storage_type = TextStorageType.DATABASE
            file_path = None
            db_content = content
        else:
            # Store in file
            storage_type = TextStorageType.FILE
            file_path = await self._save_to_file(text_id, content)
            db_content = None

        # Create database record
        text = self.create(
            id=text_id,
            title=title,
            storage_type=storage_type,
            content=db_content,
            file_path=file_path,
            length=length,
            lines=lines,
            language=language
        )

        logger.info(
            f"Created text: {text_id}, storage: {storage_type.value}, "
            f"size: {length} chars"
        )
        return text

    async def get_content(self, text_id: str) -> Optional[str]:
        """
        Get text content regardless of storage type.

        Args:
            text_id: Text ID

        Returns:
            Text content or None
        """
        text = self.get(text_id)
        if not text:
            return None

        if text.storage_type == TextStorageType.DATABASE:
            return text.content
        else:
            return await self._read_from_file(text.file_path)

    async def delete_text(self, text_id: str) -> bool:
        """
        Delete text and its file if exists.
        All related data (embeddings, tasks) will be cascade deleted.

        Args:
            text_id: Text ID

        Returns:
            True if deleted
        """
        text = self.get(text_id)
        if not text:
            return False

        # Delete file if exists
        if text.storage_type == TextStorageType.FILE and text.file_path:
            try:
                if os.path.exists(text.file_path):
                    os.remove(text.file_path)
                    logger.info(f"Deleted file: {text.file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {text.file_path}: {e}")

        # Delete from database (cascade will handle embeddings)
        result = self.delete(text_id)
        if result:
            logger.info(f"Deleted text from DB: {text_id}")
        return result

    async def get_recent_texts(
        self,
        limit: int = 10,
        skip: int = 0
    ) -> List[Text]:
        """
        Get recent texts with metadata.

        Args:
            limit: Maximum number of texts
            skip: Number to skip

        Returns:
            List of Text model objects
        """
        texts = self.db.query(Text).order_by(
            desc(Text.created_at)
        ).offset(skip).limit(limit).all()

        return texts

    async def search_texts(
        self,
        query: str,
        limit: int = 10
    ) -> List[Text]:
        """
        Search texts by title.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching texts
        """
        return self.db.query(Text).filter(
            Text.title.ilike(f"%{query}%")
        ).limit(limit).all()

    async def update_text_content(
        self,
        text_id: str,
        new_content: str
    ) -> Optional[Text]:
        """
        Update text content with storage strategy adjustment.

        Args:
            text_id: Text ID
            new_content: New content

        Returns:
            Updated Text or None
        """
        text = self.get(text_id)
        if not text:
            return None

        old_length = text.length
        new_length = len(new_content)
        new_lines = new_content.count('\n') + 1

        # Check if storage strategy needs to change
        if old_length <= self.STORAGE_THRESHOLD and new_length > self.STORAGE_THRESHOLD:
            # Move from DB to file
            file_path = await self._save_to_file(text_id, new_content)
            self.update(
                text_id,
                storage_type=TextStorageType.FILE,
                content=None,
                file_path=file_path,
                length=new_length,
                lines=new_lines
            )
        elif old_length > self.STORAGE_THRESHOLD and new_length <= self.STORAGE_THRESHOLD:
            # Move from file to DB
            if text.file_path and os.path.exists(text.file_path):
                os.remove(text.file_path)
            self.update(
                text_id,
                storage_type=TextStorageType.DATABASE,
                content=new_content,
                file_path=None,
                length=new_length,
                lines=new_lines
            )
        else:
            # Same storage type
            if text.storage_type == TextStorageType.DATABASE:
                self.update(text_id, content=new_content, length=new_length, lines=new_lines)
            else:
                await self._save_to_file(text_id, new_content)
                self.update(text_id, length=new_length, lines=new_lines)

        return self.get(text_id)

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for text based on content hash."""
        hash_obj = hashlib.sha256(content.encode())
        timestamp = str(int(datetime.now().timestamp()))
        return f"{hash_obj.hexdigest()[:16]}_{timestamp}"

    async def _save_to_file(self, text_id: str, content: str) -> str:
        """Save text content to file."""
        file_path = os.path.join(self.data_dir, f"{text_id}.txt")
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(content)
        return file_path

    async def _read_from_file(self, file_path: str) -> Optional[str]:
        """Read text content from file."""
        if not file_path or not os.path.exists(file_path):
            return None
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            return await f.read()

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        db_count = self.count({"storage_type": TextStorageType.DATABASE})
        file_count = self.count({"storage_type": TextStorageType.FILE})
        total_count = self.count()

        return {
            "total_texts": total_count,
            "database_stored": db_count,
            "file_stored": file_count,
            "storage_threshold": self.STORAGE_THRESHOLD
        }