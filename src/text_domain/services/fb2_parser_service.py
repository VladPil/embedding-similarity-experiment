"""
Сервис для парсинга FB2 файлов
"""
import re
from typing import Optional, Dict, List
from pathlib import Path
import xml.etree.ElementTree as ET
from loguru import logger

from src.common.exceptions import FB2ParseError, FileReadError


class FB2ParserService:
    """
    Сервис для парсинга FB2 (FictionBook) файлов

    Извлекает текстовое содержимое и метаданные из XML формата FB2
    """

    # Namespace для FB2 XML
    FB2_NAMESPACE = {
        'fb': 'http://www.gribuser.ru/xml/fictionbook/2.0',
        'l': 'http://www.w3.org/1999/xlink'
    }

    def __init__(self):
        """Инициализация парсера"""
        pass

    async def extract_text(self, file_path: str) -> str:
        """
        Извлечь чистый текст из FB2 файла

        Args:
            file_path: Путь к FB2 файлу

        Returns:
            str: Извлечённый текст

        Raises:
            FB2ParseError: Если не удалось распарсить файл
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileReadError(file_path)

            # Парсинг XML
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Извлечение текста из <body>
            body_elements = root.findall('.//fb:body', self.FB2_NAMESPACE)

            if not body_elements:
                # Попытка без namespace
                body_elements = root.findall('.//body')

            if not body_elements:
                logger.warning(f"No body elements found in {file_path}")
                return ""

            # Собираем текст из всех <body> (может быть несколько: основной текст, комментарии)
            texts = []
            for body in body_elements:
                # Пропускаем body с атрибутом name="notes" (сноски)
                if body.get('name') == 'notes':
                    continue

                body_text = self._extract_text_from_element(body)
                if body_text:
                    texts.append(body_text)

            result = "\n\n".join(texts)

            # Очистка текста
            result = self._clean_text(result)

            logger.debug(f"Извлечено {len(result)} символов из {file_path}")
            return result

        except ET.ParseError as e:
            raise FB2ParseError(
                message=f"XML parse error: {e}",
                details={"file_path": file_path}
            )
        except Exception as e:
            raise FB2ParseError(
                message=f"Failed to parse FB2 file: {e}",
                details={"file_path": file_path, "error": str(e)}
            )

    async def extract_metadata(self, file_path: str) -> Dict:
        """
        Извлечь метаданные из FB2 файла

        Args:
            file_path: Путь к FB2 файлу

        Returns:
            Dict: Словарь с метаданными

        Raises:
            FB2ParseError: Если не удалось распарсить файл
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileReadError(file_path)

            tree = ET.parse(file_path)
            root = tree.getroot()

            # Поиск секции description
            desc = root.find('.//fb:description', self.FB2_NAMESPACE)
            if desc is None:
                desc = root.find('.//description')

            if desc is None:
                logger.warning(f"No description found in {file_path}")
                return {}

            metadata = {}

            # Извлечение title-info
            title_info = desc.find('.//fb:title-info', self.FB2_NAMESPACE)
            if title_info is None:
                title_info = desc.find('.//title-info')

            if title_info is not None:
                # Жанры
                genres = []
                for genre in title_info.findall('.//fb:genre', self.FB2_NAMESPACE):
                    if genre.text:
                        genres.append(genre.text.strip())
                if not genres:
                    for genre in title_info.findall('.//genre'):
                        if genre.text:
                            genres.append(genre.text.strip())
                if genres:
                    metadata['genre'] = genres

                # Автор
                author = title_info.find('.//fb:author', self.FB2_NAMESPACE)
                if author is None:
                    author = title_info.find('.//author')
                if author is not None:
                    author_parts = []
                    for part in ['first-name', 'middle-name', 'last-name']:
                        elem = author.find(f'.//fb:{part}', self.FB2_NAMESPACE)
                        if elem is None:
                            elem = author.find(f'.//{part}')
                        if elem is not None and elem.text:
                            author_parts.append(elem.text.strip())
                    if author_parts:
                        metadata['author'] = ' '.join(author_parts)

                # Название книги
                book_title = title_info.find('.//fb:book-title', self.FB2_NAMESPACE)
                if book_title is None:
                    book_title = title_info.find('.//book-title')
                if book_title is not None and book_title.text:
                    metadata['title'] = book_title.text.strip()

                # Аннотация
                annotation = title_info.find('.//fb:annotation', self.FB2_NAMESPACE)
                if annotation is None:
                    annotation = title_info.find('.//annotation')
                if annotation is not None:
                    ann_text = self._extract_text_from_element(annotation)
                    if ann_text:
                        metadata['annotation'] = ann_text.strip()

                # Язык
                lang = title_info.find('.//fb:lang', self.FB2_NAMESPACE)
                if lang is None:
                    lang = title_info.find('.//lang')
                if lang is not None and lang.text:
                    metadata['language'] = lang.text.strip()

                # Серия
                sequence = title_info.find('.//fb:sequence', self.FB2_NAMESPACE)
                if sequence is None:
                    sequence = title_info.find('.//sequence')
                if sequence is not None:
                    series_name = sequence.get('name')
                    series_number = sequence.get('number')
                    if series_name:
                        metadata['series'] = series_name
                    if series_number:
                        try:
                            metadata['series_number'] = int(series_number)
                        except ValueError:
                            pass

            # Извлечение publish-info
            publish_info = desc.find('.//fb:publish-info', self.FB2_NAMESPACE)
            if publish_info is None:
                publish_info = desc.find('.//publish-info')

            if publish_info is not None:
                # Издательство
                publisher = publish_info.find('.//fb:publisher', self.FB2_NAMESPACE)
                if publisher is None:
                    publisher = publish_info.find('.//publisher')
                if publisher is not None and publisher.text:
                    metadata['publisher'] = publisher.text.strip()

                # Год издания
                year = publish_info.find('.//fb:year', self.FB2_NAMESPACE)
                if year is None:
                    year = publish_info.find('.//year')
                if year is not None and year.text:
                    try:
                        metadata['year'] = int(year.text.strip())
                    except ValueError:
                        pass

                # ISBN
                isbn = publish_info.find('.//fb:isbn', self.FB2_NAMESPACE)
                if isbn is None:
                    isbn = publish_info.find('.//isbn')
                if isbn is not None and isbn.text:
                    metadata['isbn'] = isbn.text.strip()

            logger.debug(f"Извлечены метаданные из {file_path}: {list(metadata.keys())}")
            return metadata

        except ET.ParseError as e:
            raise FB2ParseError(
                message=f"XML parse error: {e}",
                details={"file_path": file_path}
            )
        except Exception as e:
            raise FB2ParseError(
                message=f"Failed to extract metadata: {e}",
                details={"file_path": file_path, "error": str(e)}
            )

    def _extract_text_from_element(self, element: ET.Element) -> str:
        """
        Рекурсивно извлечь текст из XML элемента

        Args:
            element: XML элемент

        Returns:
            str: Извлечённый текст
        """
        texts = []

        # Добавляем текст самого элемента
        if element.text:
            texts.append(element.text.strip())

        # Рекурсивно обрабатываем дочерние элементы
        for child in element:
            # Пропускаем некоторые элементы
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag

            if tag in ['binary', 'image']:
                continue

            child_text = self._extract_text_from_element(child)
            if child_text:
                texts.append(child_text)

            # Добавляем tail (текст после закрывающего тега)
            if child.tail:
                texts.append(child.tail.strip())

        return ' '.join(texts)

    def _clean_text(self, text: str) -> str:
        """
        Очистить текст от лишних пробелов и символов

        Args:
            text: Исходный текст

        Returns:
            str: Очищенный текст
        """
        # Удаление множественных пробелов
        text = re.sub(r'\s+', ' ', text)

        # Удаление пробелов в начале и конце строк
        lines = [line.strip() for line in text.split('\n')]

        # Удаление пустых строк
        lines = [line for line in lines if line]

        # Объединение с переносами строк
        text = '\n'.join(lines)

        return text.strip()

    def validate_fb2(self, file_path: str) -> bool:
        """
        Проверить является ли файл корректным FB2

        Args:
            file_path: Путь к файлу

        Returns:
            bool: True если файл корректный FB2
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return False

            tree = ET.parse(file_path)
            root = tree.getroot()

            # Проверка корневого элемента
            tag = root.tag.split('}')[-1] if '}' in root.tag else root.tag
            return tag == 'FictionBook'

        except Exception:
            return False
