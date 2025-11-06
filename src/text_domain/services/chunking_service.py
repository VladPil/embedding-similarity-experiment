"""
Сервис для адаптивного разбиения текста на чанки
"""
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger

from ..entities.chunking_strategy import ChunkingStrategy
from src.common.exceptions import ChunkingError


@dataclass
class TextChunk:
    """
    Чанк текста с метаданными

    Представляет отдельный фрагмент текста с информацией о его позиции
    """
    content: str              # Содержимое чанка
    start_pos: int           # Начальная позиция в исходном тексте
    end_pos: int             # Конечная позиция
    chunk_index: int         # Индекс чанка (начиная с 0)
    total_chunks: int        # Общее количество чанков
    overlap_start: int = 0   # Размер перекрытия в начале
    overlap_end: int = 0     # Размер перекрытия в конце
    strategy_id: str = ""    # ID стратегии, которая создала этот чанк

    def get_length(self) -> int:
        """Получить длину чанка"""
        return len(self.content)

    def get_unique_content(self) -> str:
        """Получить содержимое без перекрытий"""
        return self.content[self.overlap_start:len(self.content) - self.overlap_end]

    def to_dict(self) -> dict:
        """Сериализация в словарь"""
        return {
            "content": self.content,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "length": self.get_length(),
            "overlap_start": self.overlap_start,
            "overlap_end": self.overlap_end,
            "strategy_id": self.strategy_id,
        }


class ChunkingService:
    """
    Сервис для адаптивного разбиения текста на чанки

    Поддерживает различные стратегии разбиения с учётом границ предложений,
    параграфов и выравнивания размеров чанков
    """

    # Регулярные выражения для поиска границ
    SENTENCE_ENDINGS = re.compile(r'[.!?]+[\s\n]+')
    PARAGRAPH_BREAKS = re.compile(r'\n\s*\n')

    def __init__(self):
        """Инициализация сервиса"""
        pass

    async def chunk_text(
        self,
        text: str,
        strategy: ChunkingStrategy
    ) -> List[TextChunk]:
        """
        Разбить текст на чанки согласно стратегии

        Args:
            text: Исходный текст
            strategy: Стратегия чанковки

        Returns:
            List[TextChunk]: Список чанков

        Raises:
            ChunkingError: Если не удалось разбить текст
        """
        try:
            # Валидация стратегии
            strategy.validate()

            # Выбор метода разбиения в зависимости от стратегии
            if strategy.use_paragraph_boundaries:
                chunks = await self._chunk_by_paragraphs(text, strategy)
            elif strategy.use_sentence_boundaries:
                chunks = await self._chunk_by_sentences(text, strategy)
            else:
                # Только для fixed_size стратегий проверяем, нужно ли разбивать короткий текст
                if len(text) <= strategy.base_chunk_size:
                    return [TextChunk(
                        content=text,
                        start_pos=0,
                        end_pos=len(text),
                        chunk_index=0,
                        total_chunks=1,
                        strategy_id=strategy.id,
                    )]
                chunks = await self._chunk_by_fixed_size(text, strategy)

            # Выравнивание размеров если требуется
            if strategy.balance_chunks and len(chunks) > 1:
                chunks = await self._balance_chunks(text, chunks, strategy)

            # Добавление перекрытий
            if strategy.overlap_percentage > 0:
                chunks = await self._add_overlaps(text, chunks, strategy)

            logger.debug(
                f"Текст разбит на {len(chunks)} чанков "
                f"(стратегия: {strategy.name}, длина: {len(text)})"
            )

            return chunks

        except Exception as e:
            raise ChunkingError(
                message=f"Failed to chunk text: {e}",
                details={
                    "text_length": len(text),
                    "strategy": strategy.name,
                    "error": str(e)
                }
            )

    async def _chunk_by_paragraphs(
        self,
        text: str,
        strategy: ChunkingStrategy
    ) -> List[TextChunk]:
        """
        Разбить текст по границам параграфов

        Args:
            text: Исходный текст
            strategy: Стратегия

        Returns:
            List[TextChunk]: Список чанков
        """
        # Поиск границ параграфов
        paragraph_positions = [0]
        for match in self.PARAGRAPH_BREAKS.finditer(text):
            paragraph_positions.append(match.end())
        paragraph_positions.append(len(text))

        chunks = []
        paragraphs_per_chunk = getattr(strategy, 'paragraphs_per_chunk', None)

        if paragraphs_per_chunk:
            # Разбиваем по количеству параграфов
            para_idx = 0
            while para_idx < len(paragraph_positions) - 1:
                # Берем нужное количество параграфов
                end_para_idx = min(para_idx + paragraphs_per_chunk, len(paragraph_positions) - 1)

                chunk_start = paragraph_positions[para_idx]
                chunk_end = paragraph_positions[end_para_idx]

                # Создаём чанк
                chunk_text = text[chunk_start:chunk_end]
                if chunk_text.strip():  # Только непустые чанки
                    chunks.append(TextChunk(
                        content=chunk_text,
                        start_pos=chunk_start,
                        end_pos=chunk_end,
                        chunk_index=len(chunks),
                        total_chunks=0,
                        strategy_id=strategy.id,
                    ))

                para_idx = end_para_idx
        else:
            # Старая логика по размеру (для обратной совместимости)
            current_start = 0
            current_content = []
            current_length = 0

            for i in range(len(paragraph_positions) - 1):
                para_start = paragraph_positions[i]
                para_end = paragraph_positions[i + 1]
                para_text = text[para_start:para_end]
                para_length = len(para_text)

                # Если добавление параграфа превысит max_chunk_size
                if current_length + para_length > strategy.max_chunk_size and current_content:
                    # Создаём чанк
                    chunk_text = ''.join(current_content)
                    chunks.append(TextChunk(
                        content=chunk_text,
                        start_pos=current_start,
                        end_pos=current_start + len(chunk_text),
                        chunk_index=len(chunks),
                        total_chunks=0,  # Обновим позже
                        strategy_id=strategy.id,
                    ))

                    # Начинаем новый чанк
                    current_start = para_start
                    current_content = [para_text]
                    current_length = para_length

                # Если текущий параграф сам больше max_chunk_size
                elif para_length > strategy.max_chunk_size:
                    # Сохраняем текущий чанк если есть
                    if current_content:
                        chunk_text = ''.join(current_content)
                        chunks.append(TextChunk(
                            content=chunk_text,
                            start_pos=current_start,
                            end_pos=current_start + len(chunk_text),
                            chunk_index=len(chunks),
                            total_chunks=0,
                            strategy_id=strategy.id,
                        ))

                    # Разбиваем большой параграф по предложениям
                    para_chunks = await self._chunk_by_sentences(para_text, strategy)
                    for pc in para_chunks:
                        pc.start_pos += para_start
                        pc.end_pos += para_start
                        pc.chunk_index = len(chunks)
                        pc.strategy_id = strategy.id
                        chunks.append(pc)

                    # Начинаем новый чанк
                    current_start = para_end
                    current_content = []
                    current_length = 0

                else:
                    # Добавляем параграф к текущему чанку
                    current_content.append(para_text)
                    current_length += para_length

            # Добавляем последний чанк (для старой логики)
            if current_content:
                chunk_text = ''.join(current_content)
                chunks.append(TextChunk(
                    content=chunk_text,
                    start_pos=current_start,
                    end_pos=current_start + len(chunk_text),
                    chunk_index=len(chunks),
                    total_chunks=0,
                    strategy_id=strategy.id,
                ))

        # Обновляем total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        return chunks

    async def _chunk_by_sentences(
        self,
        text: str,
        strategy: ChunkingStrategy
    ) -> List[TextChunk]:
        """
        Разбить текст по границам предложений

        Args:
            text: Исходный текст
            strategy: Стратегия

        Returns:
            List[TextChunk]: Список чанков
        """
        # Поиск границ предложений
        sentence_positions = [0]
        for match in self.SENTENCE_ENDINGS.finditer(text):
            sentence_positions.append(match.end())
        if sentence_positions[-1] != len(text):
            sentence_positions.append(len(text))

        chunks = []
        sentences_per_chunk = getattr(strategy, 'sentences_per_chunk', None)

        if sentences_per_chunk:
            # Разбиваем по количеству предложений
            sentence_idx = 0
            while sentence_idx < len(sentence_positions) - 1:
                # Берем нужное количество предложений
                end_sentence_idx = min(sentence_idx + sentences_per_chunk, len(sentence_positions) - 1)

                chunk_start = sentence_positions[sentence_idx]
                chunk_end = sentence_positions[end_sentence_idx]

                # Создаём чанк
                chunk_text = text[chunk_start:chunk_end]
                if chunk_text.strip():  # Только непустые чанки
                    chunks.append(TextChunk(
                        content=chunk_text,
                        start_pos=chunk_start,
                        end_pos=chunk_end,
                        chunk_index=len(chunks),
                        total_chunks=0,
                        strategy_id=strategy.id,
                    ))

                sentence_idx = end_sentence_idx
        else:
            # Старая логика по размеру (для обратной совместимости)
            i = 0
            while i < len(sentence_positions) - 1:
                # Начинаем новый чанк
                chunk_start = sentence_positions[i]
                chunk_end = sentence_positions[i]
                j = i

                # Добавляем предложения пока не достигнем целевого размера
                while j < len(sentence_positions) - 1:
                    next_end = sentence_positions[j + 1]
                    chunk_length = next_end - chunk_start

                    # Проверка размера
                    if chunk_length <= strategy.max_chunk_size:
                        chunk_end = next_end
                        j += 1

                        # Если достигли base_chunk_size - хватит
                        if chunk_length >= strategy.base_chunk_size:
                            break
                    else:
                        # Превысили max_chunk_size
                        break

                # Если чанк слишком маленький и не последний
                if chunk_end - chunk_start < strategy.min_chunk_size and j < len(sentence_positions) - 1:
                    j += 1
                    chunk_end = sentence_positions[j] if j < len(sentence_positions) else len(text)

                # Создаём чанк
                chunk_text = text[chunk_start:chunk_end]
                chunks.append(TextChunk(
                    content=chunk_text,
                    start_pos=chunk_start,
                    end_pos=chunk_end,
                    chunk_index=len(chunks),
                    total_chunks=0,
                    strategy_id=strategy.id,
                ))

                i = j

        # Обновляем total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        return chunks

    async def _chunk_by_fixed_size(
        self,
        text: str,
        strategy: ChunkingStrategy
    ) -> List[TextChunk]:
        """
        Разбить текст на чанки фиксированного размера

        Args:
            text: Исходный текст
            strategy: Стратегия

        Returns:
            List[TextChunk]: Список чанков
        """
        chunks = []
        text_length = len(text)
        chunk_size = strategy.base_chunk_size
        pos = 0

        while pos < text_length:
            end_pos = min(pos + chunk_size, text_length)
            chunk_text = text[pos:end_pos]

            chunks.append(TextChunk(
                content=chunk_text,
                start_pos=pos,
                end_pos=end_pos,
                chunk_index=len(chunks),
                total_chunks=0,
                    strategy_id=strategy.id,
            ))

            pos = end_pos

        # Обновляем total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        return chunks

    async def _balance_chunks(
        self,
        text: str,
        chunks: List[TextChunk],
        strategy: ChunkingStrategy
    ) -> List[TextChunk]:
        """
        Выровнять размеры чанков

        Args:
            text: Исходный текст
            chunks: Список чанков
            strategy: Стратегия

        Returns:
            List[TextChunk]: Выровненные чанки
        """
        # Пока просто возвращаем как есть
        # TODO: Реализовать выравнивание размеров
        return chunks

    async def _add_overlaps(
        self,
        text: str,
        chunks: List[TextChunk],
        strategy: ChunkingStrategy
    ) -> List[TextChunk]:
        """
        Добавить перекрытия между чанками

        Args:
            text: Исходный текст
            chunks: Список чанков
            strategy: Стратегия

        Returns:
            List[TextChunk]: Чанки с перекрытиями
        """
        overlap_size = strategy.get_overlap_size()

        for i in range(len(chunks)):
            chunk = chunks[i]

            # Добавляем перекрытие в начало (из предыдущего чанка)
            if i > 0:
                prev_chunk = chunks[i - 1]
                overlap_start = max(0, prev_chunk.end_pos - overlap_size)
                overlap_text = text[overlap_start:chunk.start_pos]

                chunk.content = overlap_text + chunk.content
                chunk.start_pos = overlap_start
                chunk.overlap_start = len(overlap_text)

            # Добавляем перекрытие в конец (из следующего чанка)
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                overlap_end = min(len(text), chunk.end_pos + overlap_size)
                overlap_text = text[chunk.end_pos:overlap_end]

                chunk.content = chunk.content + overlap_text
                chunk.end_pos = overlap_end
                chunk.overlap_end = len(overlap_text)

        return chunks

    def get_chunk_stats(self, chunks: List[TextChunk]) -> dict:
        """
        Получить статистику по чанкам

        Args:
            chunks: Список чанков

        Returns:
            dict: Статистика
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_length": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
            }

        lengths = [chunk.get_length() for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_length": sum(lengths),
            "avg_chunk_size": sum(lengths) / len(lengths),
            "min_chunk_size": min(lengths),
            "max_chunk_size": max(lengths),
        }
