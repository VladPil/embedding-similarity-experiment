"""
Text chunking manager.
Splits texts into manageable chunks for analysis.
"""

from typing import List, Dict, Optional
import re
from loguru import logger


class ChunkManager:
    """Manages text chunking operations."""

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 100,
        split_by: str = "sentences"
    ):
        """
        Initialize chunk manager.

        Args:
            chunk_size: Target size for each chunk (in characters or sentences)
            overlap: Overlap between chunks (in characters or sentences)
            split_by: Splitting strategy ('sentences', 'paragraphs', 'characters')
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.split_by = split_by

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with nltk or spacy)
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.

        Args:
            text: Input text

        Returns:
            List of paragraphs
        """
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]

    def chunk_by_sentences(self, text: str) -> List[Dict]:
        """
        Chunk text by sentences with overlap.

        Args:
            text: Input text

        Returns:
            List of chunk dictionaries
        """
        sentences = self.split_into_sentences(text)
        chunks = []

        i = 0
        chunk_index = 0

        while i < len(sentences):
            # Take chunk_size sentences
            end = min(i + self.chunk_size, len(sentences))
            chunk_sentences = sentences[i:end]
            chunk_text = ' '.join(chunk_sentences)

            chunks.append({
                'index': chunk_index,
                'text': chunk_text,
                'start_sentence': i,
                'end_sentence': end,
                'sentence_count': len(chunk_sentences),
                'char_count': len(chunk_text)
            })

            # Move forward with overlap
            i += self.chunk_size - self.overlap
            chunk_index += 1

            if i >= len(sentences):
                break

        return chunks

    def chunk_by_paragraphs(self, text: str) -> List[Dict]:
        """
        Chunk text by paragraphs with overlap.

        Args:
            text: Input text

        Returns:
            List of chunk dictionaries
        """
        paragraphs = self.split_into_paragraphs(text)
        chunks = []

        i = 0
        chunk_index = 0

        while i < len(paragraphs):
            # Take chunk_size paragraphs
            end = min(i + self.chunk_size, len(paragraphs))
            chunk_paragraphs = paragraphs[i:end]
            chunk_text = '\n\n'.join(chunk_paragraphs)

            chunks.append({
                'index': chunk_index,
                'text': chunk_text,
                'start_paragraph': i,
                'end_paragraph': end,
                'paragraph_count': len(chunk_paragraphs),
                'char_count': len(chunk_text)
            })

            # Move forward with overlap
            i += self.chunk_size - self.overlap
            chunk_index += 1

            if i >= len(paragraphs):
                break

        return chunks

    def chunk_by_characters(self, text: str) -> List[Dict]:
        """
        Chunk text by character count with overlap.

        Args:
            text: Input text

        Returns:
            List of chunk dictionaries
        """
        chunks = []
        text_length = len(text)

        i = 0
        chunk_index = 0

        while i < text_length:
            # Take chunk_size characters
            end = min(i + self.chunk_size, text_length)
            chunk_text = text[i:end]

            chunks.append({
                'index': chunk_index,
                'text': chunk_text,
                'start_char': i,
                'end_char': end,
                'char_count': len(chunk_text)
            })

            # Move forward with overlap
            i += self.chunk_size - self.overlap
            chunk_index += 1

            if i >= text_length:
                break

        return chunks

    def chunk_text(self, text: str) -> List[Dict]:
        """
        Chunk text using configured strategy.

        Args:
            text: Input text

        Returns:
            List of chunk dictionaries
        """
        if self.split_by == 'sentences':
            return self.chunk_by_sentences(text)
        elif self.split_by == 'paragraphs':
            return self.chunk_by_paragraphs(text)
        elif self.split_by == 'characters':
            return self.chunk_by_characters(text)
        else:
            raise ValueError(f"Unknown split_by strategy: {self.split_by}")

    def compare_chunks(
        self,
        chunks1: List[Dict],
        chunks2: List[Dict],
        similarity_fn
    ) -> List[Dict]:
        """
        Compare chunks from two texts.

        Args:
            chunks1: Chunks from first text
            chunks2: Chunks from second text
            similarity_fn: Function to calculate similarity between two chunk texts

        Returns:
            List of chunk comparison results
        """
        results = []

        for chunk1 in chunks1:
            best_match = None
            best_similarity = 0

            for chunk2 in chunks2:
                similarity = similarity_fn(chunk1['text'], chunk2['text'])

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = chunk2

            results.append({
                'chunk1_index': chunk1['index'],
                'chunk1_text': chunk1['text'][:100] + '...' if len(chunk1['text']) > 100 else chunk1['text'],
                'chunk2_index': best_match['index'] if best_match else -1,
                'chunk2_text': (best_match['text'][:100] + '...' if len(best_match['text']) > 100 else best_match['text']) if best_match else '',
                'similarity': best_similarity
            })

        return results

    def get_most_similar_chunks(
        self,
        comparisons: List[Dict],
        top_n: int = 5
    ) -> List[Dict]:
        """
        Get the most similar chunk pairs.

        Args:
            comparisons: Chunk comparison results
            top_n: Number of top results to return

        Returns:
            Top N most similar chunk pairs
        """
        sorted_comparisons = sorted(
            comparisons,
            key=lambda x: x['similarity'],
            reverse=True
        )
        return sorted_comparisons[:top_n]

    def adaptive_chunk_parameters(
        self,
        text: str,
        min_chunk_size: int = 500,
        max_chunk_size: int = 5000
    ) -> Dict:
        """
        Автоматически подбирает оптимальные параметры чанкования.

        Args:
            text: Текст для анализа
            min_chunk_size: Минимальный размер чанка
            max_chunk_size: Максимальный размер чанка

        Returns:
            Словарь с рекомендуемыми параметрами
        """
        text_length = len(text)

        if text_length <= max_chunk_size:
            return {
                'chunk_size': text_length,
                'overlap': 0,
                'split_by': 'characters',
                'estimated_chunks': 1
            }

        # Определяем стратегию в зависимости от длины текста
        if text_length < 10000:
            # Короткие тексты - по предложениям
            sentences = self.split_into_sentences(text)
            chunk_size = max(5, len(sentences) // 10)
            return {
                'chunk_size': chunk_size,
                'overlap': min(2, chunk_size // 4),
                'split_by': 'sentences',
                'estimated_chunks': len(sentences) // chunk_size + 1
            }
        elif text_length < 50000:
            # Средние тексты - по параграфам
            paragraphs = self.split_into_paragraphs(text)
            chunk_size = max(3, len(paragraphs) // 10)
            return {
                'chunk_size': chunk_size,
                'overlap': min(1, chunk_size // 4),
                'split_by': 'paragraphs',
                'estimated_chunks': len(paragraphs) // chunk_size + 1
            }
        else:
            # Длинные тексты - по символам
            optimal_chunk_size = min(max_chunk_size, max(min_chunk_size, text_length // 100))
            chunk_count = max(1, text_length // optimal_chunk_size)
            return {
                'chunk_size': optimal_chunk_size,
                'overlap': optimal_chunk_size // 10,
                'split_by': 'characters',
                'estimated_chunks': chunk_count
            }

    def adaptive_chunk_text(self, text: str) -> List[Dict]:
        """
        Автоматически разбивает текст на чанки с оптимальными параметрами.

        Args:
            text: Текст для разбиения

        Returns:
            Список чанков
        """
        params = self.adaptive_chunk_parameters(text)
        self.chunk_size = params['chunk_size']
        self.overlap = params['overlap']
        self.split_by = params['split_by']

        logger.info(
            f"Adaptive chunking: {params['split_by']} strategy, "
            f"chunk_size={params['chunk_size']}, overlap={params['overlap']}, "
            f"estimated {params['estimated_chunks']} chunks"
        )

        return self.chunk_text(text)

    def merge_overlapping_chunks(self, chunks: List[Dict]) -> str:
        """
        Merge chunks back into a single text (removing overlaps).

        Args:
            chunks: List of chunks

        Returns:
            Merged text
        """
        if not chunks:
            return ""

        # Sort chunks by index
        sorted_chunks = sorted(chunks, key=lambda x: x['index'])

        # For character-based chunks, we can use start/end positions
        if 'start_char' in sorted_chunks[0]:
            # This is tricky with overlaps - just concatenate for now
            return ' '.join(chunk['text'] for chunk in sorted_chunks)

        # For sentence/paragraph based, just concatenate
        return ' '.join(chunk['text'] for chunk in sorted_chunks)
