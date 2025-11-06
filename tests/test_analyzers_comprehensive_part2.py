"""
Comprehensive tests for analyzers - Part 2 (Tension, Theme, Water)
"""
import pytest
import numpy as np
from unittest.mock import AsyncMock, Mock, MagicMock
from typing import List, Any

from src.text_domain.entities.plain_text import PlainText
from src.text_domain.services.chunking_service import TextChunk
from src.common.types import AnalysisMode
from src.common.exceptions import AnalysisError


# ==================== Fixtures ====================

@pytest.fixture
def sample_text():
    """Create sample text"""
    return PlainText.create_from_string(
        text_id="test-1",
        title="Test Text",
        content="This is a test text. It has multiple sentences. We will use it for testing.",
        language="en"
    )


@pytest.fixture
def tension_text():
    """Create text with tension"""
    content = """
    Битва началась! Враги наступали со всех сторон. Опасность была повсюду.
    Герой сражался отчаянно. Время истекало. Смерть была близка.
    Конфликт достиг своего пика. Напряжение росло с каждой секундой.
    Загадка оставалась неразгаданной. Что же произойдет дальше?
    """ * 5

    return PlainText.create_from_string(
        text_id="tension-test",
        title="Tense Story",
        content=content,
        language="ru"
    )


@pytest.fixture
def mock_chunks():
    """Create mock chunks"""
    chunks = []
    for i in range(10):
        chunk = Mock()
        chunk.index = i
        chunk.content = f"Chunk {i} content with some action and dialogue!"
        chunk.metadata = {
            'position_ratio': i / 10.0,
            'start': i * 100,
            'end': (i + 1) * 100
        }
        chunks.append(chunk)
    return chunks


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings with varying redundancy"""
    np.random.seed(42)
    embeddings = []
    for i in range(10):
        # Create embeddings where some are more similar than others
        if i % 3 == 0:
            # High redundancy - similar to previous
            base = np.random.randn(384) * 0.1 + np.ones(384)
        else:
            # Low redundancy - different
            base = np.random.randn(384)
        base = base / np.linalg.norm(base)
        embeddings.append(base)
    return embeddings


# ==================== TensionAnalyzer Tests ====================

class TestTensionAnalyzerComprehensive:
    """Comprehensive tests for TensionAnalyzer"""

    @pytest.mark.asyncio
    async def test_properties(self):
        """Test all analyzer properties"""
        from src.analysis_domain.analyzers.tension_analyzer import TensionAnalyzer

        analyzer = TensionAnalyzer()

        assert analyzer.name == "tension"
        assert analyzer.display_name == "Анализ напряжения"
        assert len(analyzer.description) > 0
        assert analyzer.requires_llm is True
        assert analyzer.requires_embeddings is False

    @pytest.mark.asyncio
    async def test_initialization_with_params(self):
        """Test initialization with custom parameters"""
        from src.analysis_domain.analyzers.tension_analyzer import TensionAnalyzer

        mock_llm = AsyncMock()
        mock_prompt = Mock()

        analyzer = TensionAnalyzer(
            llm_service=mock_llm,
            prompt_template=mock_prompt,
            tension_threshold=7.0,
            max_peaks=10
        )

        assert analyzer.llm_service == mock_llm
        assert analyzer.prompt_template == mock_prompt
        assert analyzer.tension_threshold == 7.0
        assert analyzer.max_peaks == 10

    @pytest.mark.asyncio
    async def test_analyze_without_chunks(self, sample_text):
        """Test analyze fails without chunks"""
        from src.analysis_domain.analyzers.tension_analyzer import TensionAnalyzer

        analyzer = TensionAnalyzer()

        with pytest.raises(AnalysisError) as exc_info:
            await analyzer.analyze(sample_text, AnalysisMode.FULL_TEXT)

        assert "требует чанки" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_analyze_no_tension_peaks(self, sample_text, mock_chunks):
        """Test analyze with no tension peaks found"""
        from src.analysis_domain.analyzers.tension_analyzer import TensionAnalyzer

        analyzer = TensionAnalyzer()

        # Mock to return low tension everywhere
        analyzer.indexer._calculate_tension_from_keywords = Mock(return_value=0.1)
        analyzer.indexer.build_tension_index = Mock()
        analyzer.indexer.build_tension_index.return_value = Mock(
            chunk_indices=[],
            scores=[],
            coverage=0.0
        )
        analyzer.indexer.get_chunk_subset = Mock(return_value=[])

        result = await analyzer.analyze(
            sample_text,
            AnalysisMode.FULL_TEXT,
            chunks=mock_chunks
        )

        assert result.data['peak_count'] == 0
        assert result.data['peaks'] == []
        assert result.data['timeline'] == []

    @pytest.mark.asyncio
    async def test_analyze_with_tension_peaks(self, tension_text, mock_chunks):
        """Test full tension analysis with peaks"""
        from src.analysis_domain.analyzers.tension_analyzer import TensionAnalyzer

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value='{"source": "danger", "description": "Битва с врагами", "excerpt": "Битва началась!"}'
        )

        analyzer = TensionAnalyzer(llm_service=mock_llm)

        # Mock indexer to find high tension
        analyzer.indexer._calculate_tension_from_keywords = Mock(return_value=0.8)
        analyzer.indexer.build_tension_index = Mock()
        analyzer.indexer.build_tension_index.return_value = Mock(
            chunk_indices=[0, 5, 9],
            scores=[0.8, 0.9, 0.7],
            coverage=0.3
        )
        analyzer.indexer.get_chunk_subset = Mock(return_value=mock_chunks[:3])

        result = await analyzer.analyze(
            tension_text,
            AnalysisMode.FULL_TEXT,
            chunks=mock_chunks
        )

        assert result.text_id == tension_text.id
        assert 'average_tension' in result.data
        assert 'timeline' in result.data
        assert 'peak_count' in result.data

    @pytest.mark.asyncio
    async def test_format_tension_prompt(self):
        """Test tension prompt formatting"""
        from src.analysis_domain.analyzers.tension_analyzer import TensionAnalyzer

        analyzer = TensionAnalyzer()
        prompt = analyzer._format_tension_prompt("Test chunk with danger and conflict")

        assert "Test chunk with danger and conflict" in prompt
        assert "JSON" in prompt
        assert "source" in prompt

    @pytest.mark.asyncio
    async def test_parse_tension_response_valid(self):
        """Test parsing valid tension response"""
        from src.analysis_domain.analyzers.tension_analyzer import TensionAnalyzer

        analyzer = TensionAnalyzer()
        response = '{"source": "conflict", "description": "Test", "excerpt": "Quote"}'

        parsed = analyzer._parse_tension_response(response)

        assert parsed['source'] == "conflict"
        assert parsed['description'] == "Test"

    @pytest.mark.asyncio
    async def test_parse_tension_response_invalid(self):
        """Test parsing invalid response"""
        from src.analysis_domain.analyzers.tension_analyzer import TensionAnalyzer

        analyzer = TensionAnalyzer()
        response = "Not valid JSON"

        parsed = analyzer._parse_tension_response(response)

        assert parsed == {}

    @pytest.mark.asyncio
    async def test_interpret_results_low_tension(self):
        """Test interpretation for low tension"""
        from src.analysis_domain.analyzers.tension_analyzer import TensionAnalyzer
        from src.analysis_domain.entities.analysis_result import AnalysisResult

        analyzer = TensionAnalyzer()
        result = AnalysisResult(
            text_id="test",
            analyzer_name="TensionAnalyzer",
            data={
                'average_tension': 2.5,
                'timeline': [],
                'peak_count': 0
            },
            execution_time_ms=100,
            mode="full_text"
        )

        interpretation = analyzer.interpret_results(result)

        assert "низкое" in interpretation.lower()

    @pytest.mark.asyncio
    async def test_interpret_results_high_tension(self):
        """Test interpretation for high tension"""
        from src.analysis_domain.analyzers.tension_analyzer import TensionAnalyzer
        from src.analysis_domain.entities.analysis_result import AnalysisResult

        analyzer = TensionAnalyzer()
        result = AnalysisResult(
            text_id="test",
            analyzer_name="TensionAnalyzer",
            data={
                'average_tension': 8.5,
                'timeline': [
                    {
                        'position': 0.5,
                        'score': 9.0,
                        'source': 'danger',
                        'description': 'Epic battle scene'
                    }
                ],
                'peak_count': 1
            },
            execution_time_ms=100,
            mode="full_text"
        )

        interpretation = analyzer.interpret_results(result)

        assert "высокое" in interpretation.lower()
        assert "Epic battle scene" in interpretation

    @pytest.mark.asyncio
    async def test_analyze_peaks_with_llm_error_handling(self, mock_chunks):
        """Test LLM peak analysis handles errors"""
        from src.analysis_domain.analyzers.tension_analyzer import TensionAnalyzer

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM Error"))

        analyzer = TensionAnalyzer(llm_service=mock_llm, max_peaks=2)

        mock_index = Mock()
        mock_index.chunk_indices = [0, 1]
        mock_index.scores = [0.8, 0.9]

        result = await analyzer._analyze_peaks_with_llm(mock_chunks[:2], mock_index)

        # Should handle errors and continue
        assert isinstance(result, list)


# ==================== ThemeAnalyzer Tests ====================

class TestThemeAnalyzerComprehensive:
    """Comprehensive tests for ThemeAnalyzer"""

    @pytest.mark.asyncio
    async def test_properties(self):
        """Test all analyzer properties"""
        from src.analysis_domain.analyzers.theme_analyzer import ThemeAnalyzer

        analyzer = ThemeAnalyzer()

        assert analyzer.name == "theme"
        assert analyzer.display_name == "Анализ тем"
        assert len(analyzer.description) > 0
        assert analyzer.requires_llm is True
        assert analyzer.requires_embeddings is False

    @pytest.mark.asyncio
    async def test_initialization_with_params(self):
        """Test initialization with parameters"""
        from src.analysis_domain.analyzers.theme_analyzer import ThemeAnalyzer

        mock_llm = AsyncMock()
        mock_prompt = Mock()

        analyzer = ThemeAnalyzer(
            llm_service=mock_llm,
            prompt_template=mock_prompt,
            max_themes=3,
            sample_chunks=5
        )

        assert analyzer.llm_service == mock_llm
        assert analyzer.max_themes == 3
        assert analyzer.sample_chunks == 5

    @pytest.mark.asyncio
    async def test_analyze_without_chunks(self, sample_text):
        """Test analyze fails without chunks"""
        from src.analysis_domain.analyzers.theme_analyzer import ThemeAnalyzer

        analyzer = ThemeAnalyzer()

        with pytest.raises(AnalysisError):
            await analyzer.analyze(sample_text, AnalysisMode.FULL_TEXT)

    @pytest.mark.asyncio
    async def test_analyze_with_chunks(self, sample_text, mock_chunks):
        """Test full theme analysis"""
        from src.analysis_domain.analyzers.theme_analyzer import ThemeAnalyzer

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value='{"theme": "дружба", "description": "История о дружбе", "example": "Они были друзьями"}'
        )

        analyzer = ThemeAnalyzer(llm_service=mock_llm, sample_chunks=3)

        result = await analyzer.analyze(
            sample_text,
            AnalysisMode.FULL_TEXT,
            chunks=mock_chunks
        )

        assert result.text_id == sample_text.id
        assert 'themes' in result.data
        assert 'total_themes' in result.data
        assert result.data['chunks_analyzed'] <= 3

    @pytest.mark.asyncio
    async def test_format_theme_prompt(self):
        """Test theme prompt formatting"""
        from src.analysis_domain.analyzers.theme_analyzer import ThemeAnalyzer

        analyzer = ThemeAnalyzer()
        prompt = analyzer._format_theme_prompt("Test text about love and friendship")

        assert "Test text about love and friendship" in prompt
        assert "JSON" in prompt

    @pytest.mark.asyncio
    async def test_parse_theme_response_valid(self):
        """Test parsing valid theme response"""
        from src.analysis_domain.analyzers.theme_analyzer import ThemeAnalyzer

        analyzer = ThemeAnalyzer()
        response = '{"theme": "любовь", "description": "Test", "example": "Quote"}'

        parsed = analyzer._parse_theme_response(response)

        assert parsed['theme'] == "любовь"

    @pytest.mark.asyncio
    async def test_parse_theme_response_invalid(self):
        """Test parsing invalid response"""
        from src.analysis_domain.analyzers.theme_analyzer import ThemeAnalyzer

        analyzer = ThemeAnalyzer()
        response = "Not valid JSON"

        parsed = analyzer._parse_theme_response(response)

        assert parsed == {}

    @pytest.mark.asyncio
    async def test_aggregate_themes_empty(self):
        """Test aggregating empty themes"""
        from src.analysis_domain.analyzers.theme_analyzer import ThemeAnalyzer

        analyzer = ThemeAnalyzer()
        result = analyzer._aggregate_themes([])

        assert result == []

    @pytest.mark.asyncio
    async def test_aggregate_themes_filters_invalid(self):
        """Test aggregating filters invalid themes"""
        from src.analysis_domain.analyzers.theme_analyzer import ThemeAnalyzer

        analyzer = ThemeAnalyzer()
        mentions = [
            {'theme': '', 'description': 'test'},
            {'theme': 'другое', 'description': 'test'},
        ]

        result = analyzer._aggregate_themes(mentions)

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_aggregate_themes_combines_duplicates(self):
        """Test aggregating combines duplicate themes"""
        from src.analysis_domain.analyzers.theme_analyzer import ThemeAnalyzer

        analyzer = ThemeAnalyzer()
        mentions = [
            {
                'theme': 'любовь',
                'description': 'Описание 1',
                'example': 'Пример 1',
                'position': 0.1
            },
            {
                'theme': 'Любовь',
                'description': 'Описание 2',
                'example': 'Пример 2',
                'position': 0.5
            },
            {
                'theme': 'дружба',
                'description': 'Описание дружбы',
                'example': 'Пример',
                'position': 0.8
            }
        ]

        result = analyzer._aggregate_themes(mentions)

        # Should combine "любовь" and "Любовь"
        assert len(result) == 2
        love_theme = next(t for t in result if t['theme'].lower() == 'любовь')
        assert love_theme['frequency'] == 2

    @pytest.mark.asyncio
    async def test_aggregate_themes_limits_count(self):
        """Test aggregating limits theme count"""
        from src.analysis_domain.analyzers.theme_analyzer import ThemeAnalyzer

        analyzer = ThemeAnalyzer(max_themes=3)
        mentions = [
            {'theme': f'theme_{i}', 'description': 'test', 'example': 'test', 'position': 0.1}
            for i in range(10)
        ]

        result = analyzer._aggregate_themes(mentions)

        assert len(result) <= 3

    @pytest.mark.asyncio
    async def test_aggregate_themes_sorts_by_frequency(self):
        """Test aggregating sorts by frequency"""
        from src.analysis_domain.analyzers.theme_analyzer import ThemeAnalyzer

        analyzer = ThemeAnalyzer()
        mentions = [
            {'theme': 'популярная', 'description': 'test', 'position': 0.1},
            {'theme': 'популярная', 'description': 'test', 'position': 0.3},
            {'theme': 'популярная', 'description': 'test', 'position': 0.5},
            {'theme': 'редкая', 'description': 'test', 'position': 0.7},
        ]

        result = analyzer._aggregate_themes(mentions)

        assert result[0]['theme'].lower() == 'популярная'
        assert result[0]['frequency'] == 3

    @pytest.mark.asyncio
    async def test_interpret_results_no_themes(self):
        """Test interpretation with no themes"""
        from src.analysis_domain.analyzers.theme_analyzer import ThemeAnalyzer
        from src.analysis_domain.entities.analysis_result import AnalysisResult

        analyzer = ThemeAnalyzer()
        result = AnalysisResult(
            text_id="test",
            analyzer_name="ThemeAnalyzer",
            data={'themes': [], 'total_themes': 0},
            execution_time_ms=100,
            mode="full_text"
        )

        interpretation = analyzer.interpret_results(result)

        assert "не обнаружены" in interpretation.lower()

    @pytest.mark.asyncio
    async def test_interpret_results_with_themes(self):
        """Test interpretation with themes"""
        from src.analysis_domain.analyzers.theme_analyzer import ThemeAnalyzer
        from src.analysis_domain.entities.analysis_result import AnalysisResult

        analyzer = ThemeAnalyzer()
        result = AnalysisResult(
            text_id="test",
            analyzer_name="ThemeAnalyzer",
            data={
                'themes': [
                    {
                        'theme': 'Любовь',
                        'frequency': 3,
                        'descriptions': ['Романтическая история'],
                        'examples': ['Он любил её']
                    },
                    {
                        'theme': 'Дружба',
                        'frequency': 2,
                        'descriptions': ['Верные друзья'],
                        'examples': ['Они были друзьями']
                    }
                ],
                'total_themes': 2
            },
            execution_time_ms=100,
            mode="full_text"
        )

        interpretation = analyzer.interpret_results(result)

        assert "Любовь" in interpretation
        assert "Дружба" in interpretation
        assert "3 раз" in interpretation


# ==================== WaterAnalyzer Tests ====================

class TestWaterAnalyzerComprehensive:
    """Comprehensive tests for WaterAnalyzer"""

    @pytest.mark.asyncio
    async def test_properties(self):
        """Test all analyzer properties"""
        from src.analysis_domain.analyzers.water_analyzer import WaterAnalyzer

        analyzer = WaterAnalyzer()

        assert analyzer.name == "water"
        assert analyzer.display_name == "Анализ воды в тексте"
        assert len(analyzer.description) > 0
        assert analyzer.requires_llm is False
        assert analyzer.requires_embeddings is False

    @pytest.mark.asyncio
    async def test_analyze_without_chunks(self, sample_text):
        """Test analyze fails without chunks"""
        from src.analysis_domain.analyzers.water_analyzer import WaterAnalyzer

        analyzer = WaterAnalyzer()

        with pytest.raises(AnalysisError):
            await analyzer.analyze(sample_text, AnalysisMode.FULL_TEXT)

    @pytest.mark.asyncio
    async def test_analyze_with_embeddings(self, sample_text, mock_chunks, mock_embeddings):
        """Test analysis with embeddings"""
        from src.analysis_domain.analyzers.water_analyzer import WaterAnalyzer

        analyzer = WaterAnalyzer()

        result = await analyzer.analyze(
            sample_text,
            AnalysisMode.FULL_TEXT,
            chunks=mock_chunks,
            embeddings=mock_embeddings
        )

        assert result.text_id == sample_text.id
        assert 'water_percentage' in result.data
        assert 'info_density' in result.data
        assert 'rating' in result.data
        assert 0 <= result.data['water_percentage'] <= 100

    @pytest.mark.asyncio
    async def test_analyze_without_embeddings_fallback(self, sample_text, mock_chunks):
        """Test analysis falls back to heuristics without embeddings"""
        from src.analysis_domain.analyzers.water_analyzer import WaterAnalyzer

        analyzer = WaterAnalyzer()

        result = await analyzer.analyze(
            sample_text,
            AnalysisMode.FULL_TEXT,
            chunks=mock_chunks,
            embeddings=[]
        )

        assert 'water_percentage' in result.data
        assert 'method' in result.data
        assert result.data['method'] == 'heuristic'

    @pytest.mark.asyncio
    async def test_analyze_with_embeddings_mismatch(self, sample_text, mock_chunks):
        """Test handles embedding count mismatch"""
        from src.analysis_domain.analyzers.water_analyzer import WaterAnalyzer

        analyzer = WaterAnalyzer()

        # Wrong number of embeddings
        embeddings = [np.random.randn(384) for _ in range(5)]

        result = await analyzer.analyze(
            sample_text,
            AnalysisMode.FULL_TEXT,
            chunks=mock_chunks,
            embeddings=embeddings
        )

        # Should fallback to heuristics
        assert result.data.get('method') == 'heuristic'

    @pytest.mark.asyncio
    async def test_analyze_with_heuristics(self, mock_chunks):
        """Test heuristic analysis"""
        from src.analysis_domain.analyzers.water_analyzer import WaterAnalyzer

        analyzer = WaterAnalyzer()
        result = analyzer._analyze_with_heuristics(mock_chunks)

        assert 'water_percentage' in result
        assert 'info_density' in result
        assert 'rating' in result
        assert 'method' in result

    @pytest.mark.asyncio
    async def test_analyze_with_heuristics_empty_chunks(self):
        """Test heuristic analysis with empty chunks"""
        from src.analysis_domain.analyzers.water_analyzer import WaterAnalyzer

        analyzer = WaterAnalyzer()

        # Create chunks with no content
        chunks = []
        for i in range(5):
            chunk = Mock()
            chunk.content = ""
            chunks.append(chunk)

        result = analyzer._analyze_with_heuristics(chunks)

        assert result['water_percentage'] == 50.0  # Default

    @pytest.mark.asyncio
    async def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        from src.analysis_domain.analyzers.water_analyzer import WaterAnalyzer

        analyzer = WaterAnalyzer()

        # Identical vectors
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.0, 2.0, 3.0])
        sim1 = analyzer._cosine_similarity(vec1, vec2)
        assert abs(sim1 - 1.0) < 0.001

        # Orthogonal vectors
        vec3 = np.array([1.0, 0.0, 0.0])
        vec4 = np.array([0.0, 1.0, 0.0])
        sim2 = analyzer._cosine_similarity(vec3, vec4)
        assert abs(sim2) < 0.001

    @pytest.mark.asyncio
    async def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector"""
        from src.analysis_domain.analyzers.water_analyzer import WaterAnalyzer

        analyzer = WaterAnalyzer()

        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])

        sim = analyzer._cosine_similarity(vec1, vec2)

        assert sim == 0.0

    @pytest.mark.asyncio
    async def test_interpret_results_concise(self):
        """Test interpretation for concise text"""
        from src.analysis_domain.analyzers.water_analyzer import WaterAnalyzer
        from src.analysis_domain.entities.analysis_result import AnalysisResult

        analyzer = WaterAnalyzer()
        result = AnalysisResult(
            text_id="test",
            analyzer_name="WaterAnalyzer",
            data={
                'water_percentage': 15.0,
                'info_density': 0.85,
                'rating_ru': 'лаконичный',
                'rating_emoji': '✨'
            },
            execution_time_ms=100,
            mode="full_text"
        )

        interpretation = analyzer.interpret_results(result)

        assert "лаконичный" in interpretation.lower()
        assert "высокая информационная плотность" in interpretation.lower()

    @pytest.mark.asyncio
    async def test_interpret_results_verbose(self):
        """Test interpretation for verbose text"""
        from src.analysis_domain.analyzers.water_analyzer import WaterAnalyzer
        from src.analysis_domain.entities.analysis_result import AnalysisResult

        analyzer = WaterAnalyzer()
        result = AnalysisResult(
            text_id="test",
            analyzer_name="WaterAnalyzer",
            data={
                'water_percentage': 75.0,
                'info_density': 0.25,
                'rating_ru': 'многословный',
                'rating_emoji': '💧'
            },
            execution_time_ms=100,
            mode="full_text"
        )

        interpretation = analyzer.interpret_results(result)

        assert "много" in interpretation.lower()
        assert "редактура" in interpretation.lower()

    @pytest.mark.asyncio
    async def test_water_rating_boundaries(self):
        """Test water percentage rating boundaries"""
        from src.analysis_domain.analyzers.water_analyzer import WaterAnalyzer

        analyzer = WaterAnalyzer()

        # Create chunks for heuristic testing
        chunk = Mock()
        chunk.content = "test " * 100
        chunks = [chunk]

        result = analyzer._analyze_with_heuristics(chunks)

        rating = result['rating']
        water = result['water_percentage']

        if water < 20:
            assert rating == "concise"
        elif water < 50:
            assert rating == "balanced"
        else:
            assert rating == "verbose"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/analysis_domain/analyzers"])
