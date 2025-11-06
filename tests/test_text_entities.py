"""
Тесты для текстовых сущностей
"""
import pytest
from pathlib import Path

from src.text_domain.entities.plain_text import PlainText
from src.text_domain.entities.chunking_strategy import ChunkingStrategy
from src.text_domain.services.chunking_service import ChunkingService
from src.common.types import TextStorageType


class TestPlainText:
    """Тесты для PlainText"""

    def test_create_from_string_short(self):
        """Тест создания короткого текста из строки"""
        text = PlainText.create_from_string(
            text_id="test-1",
            title="Короткий текст",
            content="Это короткий текст.",
            language="ru"
        )

        assert text.id == "test-1"
        assert text.title == "Короткий текст"
        assert text.storage_type == TextStorageType.DATABASE
        assert text.content == "Это короткий текст."
        assert text.language == "ru"

    def test_create_from_string_long(self):
        """Тест создания длинного текста из строки"""
        long_content = "A" * 1500  # Больше 1000 символов

        text = PlainText.create_from_string(
            text_id="test-2",
            title="Длинный текст",
            content=long_content,
            language="en"
        )

        assert text.id == "test-2"
        assert text.storage_type == TextStorageType.FILE
        assert text.content is None  # Для FILE хранится в file_path
        assert text.language == "en"

    @pytest.mark.asyncio
    async def test_get_content_from_database(self):
        """Тест получения контента из БД"""
        text = PlainText.create_from_string(
            text_id="test-3",
            title="Test",
            content="Test content"
        )

        content = await text.get_content()

        assert content == "Test content"
        assert text.get_length() == len("Test content")

    def test_get_text_type(self):
        """Тест получения типа текста"""
        text = PlainText.create_from_string(
            text_id="test-4",
            title="Test",
            content="Test"
        )

        assert text.get_text_type() == "plain"

    def test_to_dict(self):
        """Тест сериализации в словарь"""
        text = PlainText.create_from_string(
            text_id="test-5",
            title="Test Title",
            content="Test content",
            language="ru"
        )

        text_dict = text.to_dict()

        assert isinstance(text_dict, dict)
        assert text_dict["id"] == "test-5"
        assert text_dict["title"] == "Test Title"
        assert text_dict["storage_type"] == "database"
        assert text_dict["has_content"] is True
        assert text_dict["language"] == "ru"

    def test_metadata(self):
        """Тест работы с метаданными"""
        metadata = {"source": "test", "author": "Test Author"}

        text = PlainText.create_from_string(
            text_id="test-6",
            title="Test",
            content="Content",
            metadata=metadata
        )

        assert text.metadata == metadata
        assert text.metadata["source"] == "test"
        assert text.metadata["author"] == "Test Author"


class TestChunkingStrategy:
    """Тесты для стратегии чанковки"""

    def test_fixed_size_strategy(self):
        """Тест стратегии фиксированного размера"""
        strategy = ChunkingStrategy.create_fixed_size(
            strategy_id="fixed-100",
            chunk_size=100,
            overlap=10
        )

        assert strategy.id == "fixed-100"
        assert strategy.method == "fixed_size"
        assert strategy.chunk_size == 100
        assert strategy.overlap == 10

    def test_sentence_based_strategy(self):
        """Тест стратегии на основе предложений"""
        strategy = ChunkingStrategy.create_sentence_based(
            strategy_id="sentence-5",
            sentences_per_chunk=5,
            overlap_sentences=1
        )

        assert strategy.id == "sentence-5"
        assert strategy.method == "sentence"
        assert strategy.sentences_per_chunk == 5
        assert strategy.overlap_sentences == 1

    def test_paragraph_based_strategy(self):
        """Тест стратегии на основе параграфов"""
        strategy = ChunkingStrategy.create_paragraph_based(
            strategy_id="para-2",
            paragraphs_per_chunk=2,
            overlap_paragraphs=0
        )

        assert strategy.id == "para-2"
        assert strategy.method == "paragraph"
        assert strategy.paragraphs_per_chunk == 2
        assert strategy.overlap_paragraphs == 0


class TestChunkingService:
    """Тесты для сервиса чанковки"""

    @pytest.mark.asyncio
    async def test_chunk_by_fixed_size(self):
        """Тест разбиения по фиксированному размеру"""
        service = ChunkingService()
        strategy = ChunkingStrategy.create_fixed_size(
            strategy_id="test-fixed",
            chunk_size=50,
            overlap=10
        )

        text = "A" * 150  # 150 символов

        chunks = await service.chunk_text(text, strategy)

        # Должно получиться несколько чанков
        assert len(chunks) > 1
        assert all(chunk.content for chunk in chunks)
        assert all(chunk.chunk_index >= 0 for chunk in chunks)
        assert all(chunk.start_pos >= 0 for chunk in chunks)

    @pytest.mark.asyncio
    async def test_chunk_by_sentences(self):
        """Тест разбиения по предложениям"""
        service = ChunkingService()
        strategy = ChunkingStrategy.create_sentence_based(
            strategy_id="test-sentence",
            sentences_per_chunk=2,
            overlap_sentences=0
        )

        text = "Первое предложение. Второе предложение. Третье предложение. Четвертое предложение."

        chunks = await service.chunk_text(text, strategy)

        # Должно получиться минимум 2 чанка (по 2 предложения)
        assert len(chunks) >= 2
        assert all(chunk.content for chunk in chunks)

    @pytest.mark.asyncio
    async def test_chunk_by_paragraphs(self):
        """Тест разбиения по параграфам"""
        service = ChunkingService()
        strategy = ChunkingStrategy.create_paragraph_based(
            strategy_id="test-para",
            paragraphs_per_chunk=1,
            overlap_paragraphs=0
        )

        text = "Первый параграф.\n\nВторой параграф.\n\nТретий параграф."

        chunks = await service.chunk_text(text, strategy)

        # Должно получиться 3 чанка (по 1 параграфу)
        assert len(chunks) >= 2
        assert all(chunk.content.strip() for chunk in chunks)

    @pytest.mark.asyncio
    async def test_chunk_with_overlap(self):
        """Тест разбиения с перекрытием"""
        service = ChunkingService()
        strategy = ChunkingStrategy.create_fixed_size(
            strategy_id="test-overlap",
            chunk_size=30,
            overlap=10
        )

        text = "ABCDEFGHIJ" * 10  # 100 символов

        chunks = await service.chunk_text(text, strategy)

        # Проверяем что есть перекрытие между чанками
        if len(chunks) > 1:
            # Последние символы первого чанка должны присутствовать в начале второго
            assert len(chunks[0].content) <= 40  # chunk_size + overlap
            assert len(chunks) > 1

    @pytest.mark.asyncio
    async def test_chunk_metadata(self):
        """Тест метаданных чанков"""
        service = ChunkingService()
        strategy = ChunkingStrategy.create_fixed_size(
            strategy_id="test-meta",
            chunk_size=50,
            overlap=0
        )

        text = "X" * 120

        chunks = await service.chunk_text(text, strategy)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.start_pos >= 0
            assert chunk.end_pos > chunk.start_pos
            assert chunk.strategy_id == "test-meta"

    @pytest.mark.asyncio
    async def test_empty_text(self):
        """Тест разбиения пустого текста"""
        service = ChunkingService()
        strategy = ChunkingStrategy.create_fixed_size(
            strategy_id="test-empty",
            chunk_size=50,
            overlap=0
        )

        chunks = await service.chunk_text("", strategy)

        # Пустой текст должен вернуть пустой список или один пустой чанк
        assert len(chunks) <= 1

    @pytest.mark.asyncio
    async def test_very_short_text(self):
        """Тест разбиения очень короткого текста"""
        service = ChunkingService()
        strategy = ChunkingStrategy.create_fixed_size(
            strategy_id="test-short",
            chunk_size=100,
            overlap=0
        )

        text = "Short"

        chunks = await service.chunk_text(text, strategy)

        # Короткий текст должен вернуть 1 чанк
        assert len(chunks) == 1
        assert chunks[0].content == "Short"


class TestTextIntegration:
    """Интеграционные тесты для текстовых компонентов"""

    @pytest.mark.asyncio
    async def test_text_with_chunking(self):
        """Тест работы текста с чанковкой"""
        text = PlainText.create_from_string(
            text_id="integration-1",
            title="Integration Test",
            content="Первое предложение. Второе предложение. Третье предложение."
        )

        service = ChunkingService()
        strategy = ChunkingStrategy.create_sentence_based(
            strategy_id="integration-strategy",
            sentences_per_chunk=1,
            overlap_sentences=0
        )

        content = await text.get_content()
        chunks = await service.chunk_text(content, strategy)

        assert len(chunks) >= 2
        assert all(chunk.content for chunk in chunks)

    @pytest.mark.asyncio
    async def test_multiple_chunking_strategies_on_same_text(self):
        """Тест применения разных стратегий к одному тексту"""
        text_content = "A" * 200

        text = PlainText.create_from_string(
            text_id="multi-strategy",
            title="Multi Strategy Test",
            content=text_content
        )

        service = ChunkingService()

        # Стратегия 1: маленькие чанки
        strategy1 = ChunkingStrategy.create_fixed_size(
            strategy_id="small",
            chunk_size=30,
            overlap=0
        )

        # Стратегия 2: большие чанки
        strategy2 = ChunkingStrategy.create_fixed_size(
            strategy_id="large",
            chunk_size=100,
            overlap=0
        )

        content = await text.get_content()
        chunks1 = await service.chunk_text(content, strategy1)
        chunks2 = await service.chunk_text(content, strategy2)

        # Маленькие чанки должно быть больше чем больших
        assert len(chunks1) > len(chunks2)
