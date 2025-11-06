"""
Тесты для анализаторов текста
"""
import pytest
from typing import Dict, Any

from src.analysis_domain.analyzers.tfidf_analyzer import TfidfAnalyzer
from src.analysis_domain.analyzers.genre_analyzer import GenreAnalyzer
from src.analysis_domain.analyzers.style_analyzer import StyleAnalyzer
from src.analysis_domain.analyzers.emotion_analyzer import EmotionAnalyzer
from src.text_domain.entities.plain_text import PlainText
from src.common.types import AnalysisMode


@pytest.fixture
def sample_text():
    """Создать тестовый текст"""
    return PlainText.create_from_string(
        text_id="test-1",
        title="Тестовый текст",
        content="Это тестовый текст для проверки анализаторов. "
                "Он содержит несколько предложений. "
                "Мы будем использовать его для тестирования различных методов анализа.",
        language="ru"
    )


@pytest.fixture
def long_sample_text():
    """Создать длинный тестовый текст"""
    content = """
    В далеком королевстве жил-был храбрый рыцарь. Он отправился в опасное приключение,
    чтобы спасти принцессу от злого дракона. Путь был долгим и трудным, полным опасностей.
    Рыцарь сражался с темными силами, преодолевал горы и реки.

    Однажды, в темном лесу, он встретил мудрого старца, который дал ему волшебный меч.
    Этот меч обладал огромной силой и мог победить любого врага. Рыцарь был благодарен
    и продолжил свой путь с новой надеждой.

    Наконец, он достиг замка дракона. Битва была жестокой, но благодаря волшебному мечу
    и своей храбрости, рыцарь победил дракона и спас принцессу. Они вернулись в королевство,
    где их встретили как героев.
    """ * 3  # Повторяем для увеличения размера

    return PlainText.create_from_string(
        text_id="test-long",
        title="Длинный тестовый текст",
        content=content,
        language="ru"
    )


class TestTfidfAnalyzer:
    """Тесты для TF-IDF анализатора"""

    @pytest.mark.asyncio
    async def test_analyzer_properties(self):
        """Тест свойств анализатора"""
        analyzer = TfidfAnalyzer()

        assert analyzer.name == "tfidf"
        assert analyzer.display_name == "TF-IDF анализ"
        assert analyzer.description != ""
        assert analyzer.requires_llm is False
        assert analyzer.requires_embeddings is False

    @pytest.mark.asyncio
    async def test_analyze_full_text(self, sample_text):
        """Тест анализа полного текста"""
        analyzer = TfidfAnalyzer()

        result = await analyzer.analyze(
            text=sample_text,
            mode=AnalysisMode.FULL_TEXT
        )

        assert result is not None
        assert result.text_id == "test-1"
        assert result.analyzer_name == "tfidf"
        assert result.mode == "full_text"
        assert result.execution_time_ms > 0

        # Проверяем данные результата
        assert "top_terms" in result.data
        assert "total_terms" in result.data
        assert "avg_score" in result.data
        assert isinstance(result.data["top_terms"], list)

        # Проверяем интерпретацию
        assert result.interpretation is not None
        assert len(result.interpretation) > 0

    @pytest.mark.asyncio
    async def test_top_terms_structure(self, sample_text):
        """Тест структуры топ-терминов"""
        analyzer = TfidfAnalyzer()
        result = await analyzer.analyze(sample_text, AnalysisMode.FULL_TEXT)

        top_terms = result.data["top_terms"]

        if len(top_terms) > 0:
            first_term = top_terms[0]
            assert "term" in first_term
            assert "score" in first_term
            assert "rank" in first_term
            assert isinstance(first_term["score"], float)
            assert first_term["score"] >= 0

    @pytest.mark.asyncio
    async def test_estimated_time(self):
        """Тест оценки времени выполнения"""
        analyzer = TfidfAnalyzer()

        # Короткий текст
        time_short = analyzer.get_estimated_time(1000, AnalysisMode.FULL_TEXT)
        assert time_short > 0

        # Длинный текст
        time_long = analyzer.get_estimated_time(10000, AnalysisMode.FULL_TEXT)
        assert time_long > time_short


class TestGenreAnalyzer:
    """Тесты для анализатора жанра"""

    @pytest.mark.asyncio
    async def test_analyzer_properties(self):
        """Тест свойств анализатора"""
        analyzer = GenreAnalyzer()

        assert analyzer.name == "genre"
        assert analyzer.display_name == "Анализ жанра"
        assert analyzer.description != ""
        assert analyzer.requires_llm is True
        assert analyzer.supports_chunked_mode is False

    @pytest.mark.asyncio
    async def test_analyze_with_mock(self, long_sample_text):
        """Тест анализа с моком (без реального LLM)"""
        analyzer = GenreAnalyzer(llm_service=None)

        result = await analyzer.analyze(
            text=long_sample_text,
            mode=AnalysisMode.FULL_TEXT
        )

        assert result is not None
        assert result.text_id == "test-long"
        assert result.analyzer_name == "genre"
        assert result.execution_time_ms > 0

        # Проверяем структуру данных
        assert "main_genre" in result.data
        assert "sub_genres" in result.data
        assert "confidence" in result.data

        # Проверяем типы
        assert isinstance(result.data["main_genre"], str)
        assert isinstance(result.data["sub_genres"], list)
        assert isinstance(result.data["confidence"], float)
        assert 0 <= result.data["confidence"] <= 1

    @pytest.mark.asyncio
    async def test_interpret_results(self, sample_text):
        """Тест интерпретации результатов"""
        analyzer = GenreAnalyzer()
        result = await analyzer.analyze(sample_text, AnalysisMode.FULL_TEXT)

        interpretation = result.interpretation
        assert interpretation is not None
        assert "Жанр:" in interpretation
        assert len(interpretation) > 20

    @pytest.mark.asyncio
    async def test_estimated_time(self):
        """Тест оценки времени выполнения"""
        analyzer = GenreAnalyzer()

        time_estimate = analyzer.get_estimated_time(5000, AnalysisMode.FULL_TEXT)
        assert time_estimate > 5.0  # LLM анализ должен занимать больше времени


class TestStyleAnalyzer:
    """Тесты для анализатора стиля"""

    @pytest.mark.asyncio
    async def test_analyzer_properties(self):
        """Тест свойств анализатора"""
        try:
            from src.analysis_domain.analyzers.style_analyzer import StyleAnalyzer
            analyzer = StyleAnalyzer()

            assert analyzer.name == "style"
            assert analyzer.requires_llm is True
            assert hasattr(analyzer, 'display_name')
            assert hasattr(analyzer, 'description')
        except ImportError:
            pytest.skip("StyleAnalyzer not implemented yet")

    @pytest.mark.asyncio
    async def test_analyze_with_mock(self, sample_text):
        """Тест анализа стиля с моком"""
        try:
            from src.analysis_domain.analyzers.style_analyzer import StyleAnalyzer
            analyzer = StyleAnalyzer(llm_service=None)

            result = await analyzer.analyze(
                text=sample_text,
                mode=AnalysisMode.FULL_TEXT
            )

            assert result is not None
            assert result.analyzer_name == "style"
            assert result.data is not None
        except ImportError:
            pytest.skip("StyleAnalyzer not implemented yet")


class TestEmotionAnalyzer:
    """Тесты для анализатора эмоций"""

    @pytest.mark.asyncio
    async def test_analyzer_properties(self):
        """Тест свойств анализатора"""
        try:
            from src.analysis_domain.analyzers.emotion_analyzer import EmotionAnalyzer
            analyzer = EmotionAnalyzer()

            assert analyzer.name == "emotion"
            assert analyzer.requires_llm is True
            assert hasattr(analyzer, 'display_name')
            assert hasattr(analyzer, 'description')
        except ImportError:
            pytest.skip("EmotionAnalyzer not implemented yet")

    @pytest.mark.asyncio
    async def test_analyze_with_mock(self, long_sample_text):
        """Тест анализа эмоций с моком"""
        try:
            from src.analysis_domain.analyzers.emotion_analyzer import EmotionAnalyzer
            analyzer = EmotionAnalyzer(llm_service=None)

            result = await analyzer.analyze(
                text=long_sample_text,
                mode=AnalysisMode.FULL_TEXT
            )

            assert result is not None
            assert result.analyzer_name == "emotion"
            assert result.data is not None
        except ImportError:
            pytest.skip("EmotionAnalyzer not implemented yet")


@pytest.mark.asyncio
async def test_multiple_analyzers_on_same_text(sample_text):
    """Тест применения нескольких анализаторов к одному тексту"""
    tfidf = TfidfAnalyzer()
    genre = GenreAnalyzer()

    tfidf_result = await tfidf.analyze(sample_text, AnalysisMode.FULL_TEXT)
    genre_result = await genre.analyze(sample_text, AnalysisMode.FULL_TEXT)

    # Проверяем что результаты разные
    assert tfidf_result.analyzer_name != genre_result.analyzer_name
    assert tfidf_result.data != genre_result.data

    # Но оба валидны
    assert tfidf_result.text_id == sample_text.id
    assert genre_result.text_id == sample_text.id
