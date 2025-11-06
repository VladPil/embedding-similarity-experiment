"""
Text storage and management with async file I/O.
"""

import aiofiles
from pathlib import Path
import hashlib
from datetime import datetime
from typing import List, Dict, Optional
from loguru import logger

from server.config import settings


class TextManager:
    """Manage stored texts with async file operations."""

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize text storage.

        Args:
            storage_dir: Directory to store texts (None for settings default)
        """
        self.storage_dir = Path(storage_dir or settings.storage_texts_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Text storage initialized at {self.storage_dir}")

    async def save_text(self, text: str, title: Optional[str] = None, text_id: Optional[str] = None) -> str:
        """
        Save text to storage.

        Args:
            text: Text content
            title: Optional title for the text
            text_id: Optional text ID (if not provided, will be generated)

        Returns:
            Text ID (filename without extension)
        """
        # Generate ID from content hash if not provided
        if not text_id:
            text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            text_id = f"{timestamp}_{text_hash}"

        # Save text
        file_path = self.storage_dir / f"{text_id}.txt"
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(text)

        # Save metadata
        if title:
            meta_path = self.storage_dir / f"{text_id}.meta"
            async with aiofiles.open(meta_path, 'w', encoding='utf-8') as f:
                await f.write(f"title: {title}\n")
                await f.write(f"length: {len(text)}\n")
                await f.write(f"lines: {text.count(chr(10)) + 1}\n")
                await f.write(f"created_at: {datetime.now().isoformat()}\n")

        logger.info(f"Saved text {text_id} ({len(text)} chars)")
        return text_id

    async def update_text(self, text_id: str, text: str, title: Optional[str] = None) -> bool:
        """
        Update existing text without changing its ID.

        Args:
            text_id: Text identifier
            text: New text content
            title: Optional new title for the text

        Returns:
            True if updated successfully
        """
        file_path = self.storage_dir / f"{text_id}.txt"

        if not file_path.exists():
            return False

        # Update text file
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(text)

        # Update metadata
        meta_path = self.storage_dir / f"{text_id}.meta"
        async with aiofiles.open(meta_path, 'w', encoding='utf-8') as f:
            if title:
                await f.write(f"title: {title}\n")
            await f.write(f"length: {len(text)}\n")
            await f.write(f"lines: {text.count(chr(10)) + 1}\n")
            await f.write(f"updated_at: {datetime.now().isoformat()}\n")

        logger.info(f"Updated text {text_id} ({len(text)} chars)")
        return True

    async def get_text(self, text_id: str) -> Optional[str]:
        """
        Load text from storage.

        Args:
            text_id: Text ID

        Returns:
            Text content or None if not found
        """
        file_path = self.storage_dir / f"{text_id}.txt"

        if not file_path.exists():
            return None

        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            return await f.read()

    async def get_metadata(self, text_id: str) -> Dict:
        """
        Get text metadata.

        Args:
            text_id: Text ID

        Returns:
            Metadata dictionary
        """
        meta_path = self.storage_dir / f"{text_id}.meta"
        file_path = self.storage_dir / f"{text_id}.txt"

        metadata = {
            'id': text_id,
            'title': text_id,
            'length': 0,
            'lines': 0,
            'created_at': None
        }

        # Load from meta file if exists
        if meta_path.exists():
            async with aiofiles.open(meta_path, 'r', encoding='utf-8') as f:
                async for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        metadata[key.strip()] = value.strip()

        # Get length from actual file
        if file_path.exists():
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                metadata['length'] = int(len(content))
                metadata['lines'] = int(content.count('\n') + 1)

        return metadata

    async def list_texts(self) -> List[Dict]:
        """
        List all stored texts.

        Returns:
            List of text metadata dictionaries
        """
        texts = []

        # Use synchronous glob since it's fast
        for file_path in self.storage_dir.glob("*.txt"):
            text_id = file_path.stem
            metadata = await self.get_metadata(text_id)
            texts.append(metadata)

        # Sort by ID (timestamp)
        texts.sort(key=lambda x: x['id'], reverse=True)

        return texts

    async def delete_text(self, text_id: str) -> bool:
        """
        Delete text from storage.

        Args:
            text_id: Text ID

        Returns:
            True if deleted, False if not found
        """
        file_path = self.storage_dir / f"{text_id}.txt"
        meta_path = self.storage_dir / f"{text_id}.meta"

        deleted = False

        if file_path.exists():
            file_path.unlink()
            deleted = True
            logger.info(f"Deleted text {text_id}")

        if meta_path.exists():
            meta_path.unlink()

        return deleted

    async def clear_all(self):
        """Delete all stored texts."""
        count = 0
        for file_path in self.storage_dir.glob("*.txt"):
            file_path.unlink()
            count += 1

        for file_path in self.storage_dir.glob("*.meta"):
            file_path.unlink()

        logger.info(f"Cleared all texts ({count} files)")

    async def text_exists(self, text_id: str) -> bool:
        """
        Check if text exists.

        Args:
            text_id: Text ID

        Returns:
            True if text exists
        """
        file_path = self.storage_dir / f"{text_id}.txt"
        return file_path.exists()
