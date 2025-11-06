"""
Service for text management with business logic.
"""

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
import logging
import os

from server.services.base import BaseService
from server.repositories.text_repository import TextRepository
from server.repositories.embedding_repository import EmbeddingRepository
from server.core.texts.fb2_parser import FB2Parser
from server.db.models import Text

logger = logging.getLogger(__name__)


class TextService(BaseService):
    """
    Service for text operations.
    Handles business logic and coordinates repositories.
    """

    def __init__(self, db: Session):
        """Initialize text service."""
        super().__init__(db)
        self.text_repo = TextRepository(db)
        self.embedding_repo = EmbeddingRepository(db)
        self.fb2_parser = FB2Parser()

    async def initialize(self):
        """Initialize service."""
        self.log_info("Text service initialized")

    async def cleanup(self):
        """Cleanup service resources."""
        self.log_info("Text service cleaned up")

    async def create_text(
        self,
        title: str,
        content: str,
        language: Optional[str] = None
    ) -> Text:
        """
        Create text with smart storage.

        Args:
            title: Text title
            content: Text content
            language: Text language

        Returns:
            Text model object
        """
        try:
            # Validate input
            if not title or not content:
                raise ValueError("Title and content are required")

            if len(title) > 500:
                raise ValueError("Title too long (max 500 characters)")

            # Create text using repository
            text = await self.text_repo.create_text(title, content, language)

            self.log_info(f"Created text: {text.id}")

            return text

        except Exception as e:
            self.log_error(f"Error creating text", e)
            raise

    async def upload_fb2(
        self,
        file_content: bytes,
        custom_title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload FB2 file.

        Args:
            file_content: FB2 file content
            custom_title: Custom title (optional)

        Returns:
            Text information dictionary
        """
        try:
            # Parse FB2
            parsed = self.fb2_parser.parse(file_content)

            if not parsed:
                raise ValueError("Failed to parse FB2 file")

            title = custom_title or parsed.get("title", "Untitled")
            content = parsed.get("content", "")

            # Create text
            return await self.create_text(title, content, language="ru")

        except Exception as e:
            self.log_error(f"Error uploading FB2", e)
            raise

    def get_text(self, text_id: str) -> Optional[Text]:
        """
        Get text by ID.

        Args:
            text_id: Text ID

        Returns:
            Text object or None
        """
        return self.text_repo.get(text_id)

    def get_text_content_sync(self, text_id: str) -> Optional[str]:
        """
        Get text content synchronously (for background tasks).

        Args:
            text_id: Text ID

        Returns:
            Text content or None
        """
        text = self.text_repo.get(text_id)
        if not text:
            return None

        if text.storage_type.value == "DATABASE":
            return text.content
        elif text.storage_type.value == "FILE":
            if text.file_path and os.path.exists(text.file_path):
                with open(text.file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        return None

    async def get_text_content(self, text_id: str) -> Optional[str]:
        """
        Get text content.

        Args:
            text_id: Text ID

        Returns:
            Text content or None
        """
        return await self.text_repo.get_content(text_id)

    async def get_text_info(self, text_id: str) -> Optional[Dict[str, Any]]:
        """
        Get text information.

        Args:
            text_id: Text ID

        Returns:
            Text info dictionary or None
        """
        text = self.text_repo.get(text_id)
        if not text:
            return None

        return {
            "id": text.id,
            "title": text.title,
            "length": text.length,
            "lines": text.lines,
            "storage_type": text.storage_type.value,
            "created_at": text.created_at.isoformat(),
            "has_embeddings": len(text.embeddings) > 0
        }

    async def list_texts(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List texts with metadata.

        Args:
            limit: Maximum number
            offset: Number to skip

        Returns:
            List of text info dictionaries
        """
        texts = await self.text_repo.get_recent_texts(limit, offset)
        return [
            {
                "id": text.id,
                "title": text.title,
                "length": text.length,
                "lines": text.lines,
                "storage_type": text.storage_type.value if hasattr(text, 'storage_type') else None,
                "created_at": text.created_at.isoformat() if hasattr(text, 'created_at') else None
            }
            for text in texts
        ]

    async def update_text(
        self,
        text_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None
    ) -> Optional[Text]:
        """
        Update text information.

        Args:
            text_id: Text ID
            title: New title (optional)
            content: New content (optional)

        Returns:
            Updated text object or None
        """
        try:
            # Get existing text
            text = self.text_repo.get(text_id)
            if not text:
                return None

            # Update title if provided
            if title is not None:
                if len(title) > 500:
                    raise ValueError("Title too long (max 500 characters)")
                text.title = title

            # Update content if provided
            if content is not None:
                # Recalculate storage strategy
                await self.text_repo.update_text_content(text_id, content)
                # Get updated text
                text = self.text_repo.get(text_id)

                # Clear embeddings cache for updated text
                await self.embedding_repo.delete_text_embeddings(text_id)

            elif title is not None:
                # Only title updated, just save
                text = self.text_repo.get(text_id)
                if text:
                    text.title = title
                    self.db.commit()
                    self.db.refresh(text)

            self.log_info(f"Updated text: {text_id}")
            return text

        except Exception as e:
            self.log_error(f"Error updating text {text_id}", e)
            raise

    async def delete_text(self, text_id: str) -> bool:
        """
        Delete text and all related data.

        Args:
            text_id: Text ID

        Returns:
            True if deleted
        """
        try:
            # Delete embeddings first (cascade will handle it, but let's be explicit)
            await self.embedding_repo.delete_text_embeddings(text_id)

            # Delete text (will cascade delete embeddings and tasks)
            result = await self.text_repo.delete_text(text_id)

            if result:
                self.log_info(f"Deleted text and related data: {text_id}")

            return result

        except Exception as e:
            self.log_error(f"Error deleting text {text_id}", e)
            return False

    async def search_texts(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search texts by title.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching texts
        """
        texts = await self.text_repo.search_texts(query, limit)
        return [
            {
                "id": text.id,
                "title": text.title,
                "length": text.length,
                "lines": text.lines
            }
            for text in texts
        ]

    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Storage stats dictionary
        """
        text_stats = await self.text_repo.get_storage_stats()
        cache_stats = await self.embedding_repo.get_cache_stats()

        return {
            "texts": text_stats,
            "cache": cache_stats
        }

    async def validate_text_exists(self, text_id: str) -> bool:
        """
        Validate that text exists.

        Args:
            text_id: Text ID

        Returns:
            True if exists
        """
        return await self.text_repo.exists(text_id)

    async def validate_texts_exist(self, text_ids: List[str]) -> Dict[str, bool]:
        """
        Validate multiple texts exist.

        Args:
            text_ids: List of text IDs

        Returns:
            Dictionary mapping text_id to existence
        """
        results = {}
        for text_id in text_ids:
            results[text_id] = await self.text_repo.exists(text_id)
        return results