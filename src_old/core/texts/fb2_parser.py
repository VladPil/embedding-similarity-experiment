"""
FB2 file parser for extracting text content.
"""

from lxml import etree
from typing import Optional
import re
from loguru import logger


class FB2Parser:
    """Parser for FB2 ebook format."""

    def __init__(self):
        self.namespaces = {
            'fb2': 'http://www.gribuser.ru/xml/fictionbook/2.0'
        }

    def parse_file(self, file_path: str) -> str:
        """
        Parse FB2 file and extract text content.

        Args:
            file_path: Path to FB2 file

        Returns:
            Extracted text content
        """
        try:
            tree = etree.parse(file_path)
            return self._extract_text(tree)
        except Exception as e:
            logger.error(f"Failed to parse FB2 file: {e}")
            raise ValueError(f"Failed to parse FB2 file: {str(e)}")

    def parse_bytes(self, content: bytes) -> str:
        """
        Parse FB2 content from bytes.

        Args:
            content: FB2 file content as bytes

        Returns:
            Extracted text content
        """
        try:
            tree = etree.fromstring(content)
            return self._extract_text(tree)
        except Exception as e:
            logger.error(f"Failed to parse FB2 content: {e}")
            raise ValueError(f"Failed to parse FB2 content: {str(e)}")

    def _extract_text(self, tree) -> str:
        """
        Extract text from parsed XML tree.

        Args:
            tree: Parsed XML tree

        Returns:
            Extracted text
        """
        # Try with namespace
        body = tree.find('.//fb2:body', self.namespaces)

        # Try without namespace if not found
        if body is None:
            body = tree.find('.//body')

        if body is None:
            raise ValueError("Could not find body element in FB2 file")

        # Extract all text from paragraphs
        text_parts = []

        # Try with namespace
        paragraphs = body.findall('.//fb2:p', self.namespaces)

        # Try without namespace if not found
        if not paragraphs:
            paragraphs = body.findall('.//p')

        for p in paragraphs:
            text = ''.join(p.itertext())
            if text.strip():
                text_parts.append(text.strip())

        # Join with newlines
        full_text = '\n'.join(text_parts)

        # Clean up excessive whitespace
        full_text = re.sub(r'\n\n+', '\n\n', full_text)
        full_text = re.sub(r' +', ' ', full_text)

        logger.debug(f"Extracted {len(full_text)} characters from FB2")
        return full_text

    @staticmethod
    def get_book_title(file_path: str) -> Optional[str]:
        """
        Extract book title from FB2 file.

        Args:
            file_path: Path to FB2 file

        Returns:
            Book title or None
        """
        try:
            tree = etree.parse(file_path)
            namespaces = {'fb2': 'http://www.gribuser.ru/xml/fictionbook/2.0'}

            # Try with namespace
            title = tree.find('.//fb2:book-title', namespaces)

            # Try without namespace
            if title is None:
                title = tree.find('.//book-title')

            if title is not None and title.text:
                return title.text.strip()

            return None
        except Exception as e:
            logger.error(f"Failed to get book title: {e}")
            return None
