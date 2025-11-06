"""
Индексатор чанков для быстрого поиска релевантных фрагментов
"""
import re
from typing import List, Dict, Set
from dataclasses import dataclass
import numpy as np

from src.text_domain.entities.text_chunk import TextChunk


@dataclass
class ChunkIndex:
    """
    Результат индексации чанков

    Содержит индексы релевантных чанков и метаданные
    """
    chunk_indices: List[int]  # Индексы отобранных чанков
    scores: List[float]  # Скоры релевантности
    coverage: float  # Процент охвата текста (0-1)


class ChunkIndexer:
    """
    Индексатор для быстрого поиска релевантных чанков

    Использует эвристики и ключевые слова для определения
    релевантности чанков без использования LLM
    """

    # Ключевые слова для определения типа контента
    CHARACTER_KEYWORDS = {
        # Русские
        'сказал', 'сказала', 'подумал', 'подумала', 'ответил', 'ответила',
        'спросил', 'спросила', 'произнёс', 'произнесла', 'крикнул', 'крикнула',
        'он', 'она', 'его', 'её', 'ему', 'ей', 'им', 'ей',
        # Английские
        'said', 'asked', 'replied', 'thought', 'whispered', 'shouted',
        'he', 'she', 'his', 'her', 'him', 'them'
    }

    TENSION_KEYWORDS = {
        # Русские
        'опасность', 'угроза', 'страх', 'тревога', 'паника', 'ужас',
        'враг', 'смерть', 'кровь', 'крик', 'бежать', 'спасаться',
        'конфликт', 'битва', 'борьба', 'преследование',
        # Английские
        'danger', 'threat', 'fear', 'panic', 'horror', 'enemy',
        'death', 'blood', 'scream', 'run', 'escape', 'conflict', 'battle'
    }

    EVENT_KEYWORDS = {
        # Действия
        'пошёл', 'пошла', 'побежал', 'побежала', 'прыгнул', 'прыгнула',
        'ударил', 'ударила', 'схватил', 'схватила', 'открыл', 'открыла',
        'бросил', 'бросила', 'нашёл', 'нашла',
        # English
        'went', 'ran', 'jumped', 'hit', 'grabbed', 'opened', 'threw', 'found'
    }

    def build_character_index(self, chunks: List[TextChunk], threshold: float = 0.3) -> ChunkIndex:
        """
        Построить индекс чанков с персонажами

        Args:
            chunks: Список чанков
            threshold: Порог релевантности (0-1)

        Returns:
            ChunkIndex: Индекс персонажных чанков
        """
        relevant_indices = []
        scores = []

        for i, chunk in enumerate(chunks):
            score = self._calculate_character_score(chunk.content)

            if score >= threshold:
                relevant_indices.append(i)
                scores.append(score)

        coverage = len(relevant_indices) / len(chunks) if chunks else 0.0

        return ChunkIndex(
            chunk_indices=relevant_indices,
            scores=scores,
            coverage=coverage
        )

    def build_tension_index(
        self,
        chunks: List[TextChunk],
        threshold: float = 0.5
    ) -> ChunkIndex:
        """
        Построить индекс чанков с высоким напряжением

        Args:
            chunks: Список чанков
            threshold: Порог напряжения (0-1)

        Returns:
            ChunkIndex: Индекс напряжённых чанков
        """
        relevant_indices = []
        scores = []

        for i, chunk in enumerate(chunks):
            score = self._calculate_tension_from_keywords(chunk.content)

            if score >= threshold:
                relevant_indices.append(i)
                scores.append(score)

        coverage = len(relevant_indices) / len(chunks) if chunks else 0.0

        return ChunkIndex(
            chunk_indices=relevant_indices,
            scores=scores,
            coverage=coverage
        )

    def get_chunk_subset(
        self,
        chunks: List[TextChunk],
        index_type: str
    ) -> List[TextChunk]:
        """
        Получить подмножество чанков по типу индекса

        Args:
            chunks: Все чанки
            index_type: Тип индекса (characters, tension)

        Returns:
            List[TextChunk]: Отфильтрованные чанки
        """
        if index_type == 'characters':
            index = self.build_character_index(chunks)
        elif index_type == 'tension':
            index = self.build_tension_index(chunks)
        else:
            return chunks

        return [chunks[i] for i in index.chunk_indices]

    def _calculate_character_score(self, text: str) -> float:
        """
        Вычислить скор наличия персонажей в тексте

        Args:
            text: Текст чанка

        Returns:
            float: Скор 0-1
        """
        text_lower = text.lower()

        # Подсчитываем ключевые слова
        keyword_count = sum(
            text_lower.count(keyword)
            for keyword in self.CHARACTER_KEYWORDS
        )

        # Подсчитываем диалоги (кавычки, тире)
        dialogue_markers = text.count('—') + text.count('"') + text.count('«')

        # Подсчитываем имена собственные (заглавные буквы в середине предложений)
        proper_nouns = len(re.findall(r'(?<=[а-яa-z]\s)[А-ЯA-Z][а-яa-z]+', text))

        # Нормализуем
        text_length = len(text.split())
        if text_length == 0:
            return 0.0

        keyword_density = min(keyword_count / text_length, 1.0)
        dialogue_density = min(dialogue_markers / 10, 1.0)  # Нормализуем к 10 маркерам
        proper_noun_density = min(proper_nouns / text_length, 1.0)

        # Взвешенная сумма
        score = (
            keyword_density * 0.4 +
            dialogue_density * 0.4 +
            proper_noun_density * 0.2
        )

        return min(score, 1.0)

    def _calculate_tension_from_keywords(self, text: str) -> float:
        """
        Вычислить напряжение на основе ключевых слов

        Args:
            text: Текст чанка

        Returns:
            float: Скор напряжения 0-10
        """
        text_lower = text.lower()

        # Подсчитываем ключевые слова напряжения
        tension_count = sum(
            text_lower.count(keyword)
            for keyword in self.TENSION_KEYWORDS
        )

        # Подсчитываем восклицательные знаки
        exclamations = text.count('!')

        # Подсчитываем многоточия (нерешительность, пауза)
        ellipsis = text.count('...')

        # Подсчитываем короткие предложения (быстрый темп)
        sentences = re.split(r'[.!?]+', text)
        short_sentences = sum(1 for s in sentences if 0 < len(s.split()) < 8)

        text_length = len(text.split())
        if text_length == 0:
            return 0.0

        # Нормализуем к шкале 0-10
        tension_density = min(tension_count / text_length * 20, 1.0)
        exclamation_factor = min(exclamations / 5, 1.0)
        short_sent_factor = min(short_sentences / len(sentences), 1.0) if sentences else 0.0

        score = (
            tension_density * 0.5 +
            exclamation_factor * 0.3 +
            short_sent_factor * 0.2
        ) * 10

        return min(score, 10.0)

    def _calculate_event_density(self, text: str) -> float:
        """
        Вычислить плотность событий

        Args:
            text: Текст чанка

        Returns:
            float: Плотность событий 0-1
        """
        text_lower = text.lower()

        # Подсчитываем глаголы действия
        event_count = sum(
            text_lower.count(keyword)
            for keyword in self.EVENT_KEYWORDS
        )

        text_length = len(text.split())
        if text_length == 0:
            return 0.0

        return min(event_count / text_length, 1.0)

    def _calculate_dialogue_ratio(self, text: str) -> float:
        """
        Вычислить соотношение диалогов

        Args:
            text: Текст чанка

        Returns:
            float: Соотношение диалогов 0-1
        """
        # Подсчитываем маркеры диалогов
        dialogue_markers = text.count('—') + text.count('"') + text.count('«') + text.count('"')

        # Нормализуем
        return min(dialogue_markers / 20, 1.0)
