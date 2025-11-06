"""
Comprehensive tests for all analyzers to achieve 100% coverage
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
def russian_text():
    """Create Russian text"""
    content = """
    Рыцарь мчался по темному лесу. Враги преследовали его по пятам.
    Внезапно он заметил старую пещеру. Сердце билось быстро.

    В пещере было темно и сыро. Он достал факел и осмотрелся.
    Иван встретил Марию в парке. "Привет!" - сказал он.
    "Как дела?" - спросила она. Они давно не виделись.
    """ * 3

    return PlainText.create_from_string(
        text_id="ru-test",
        title="Russian Adventure",
        content=content,
        language="ru"
    )


@pytest.fixture
def watery_text():
    """Create text with lots of 'water' (repetitive content)"""
    content = """
    Как известно, в наше время, вообще говоря, можно сказать, что,
    по сути дела, на самом деле, если разобраться, то есть,
    другими словами, иными словами, так сказать, в некотором смысле,
    как бы, в общем-то, в принципе, фактически, практически...
    """ * 20

    return PlainText.create_from_string(
        text_id="watery",
        title="Watery Text",
        content=content,
        language="ru"
    )


@pytest.fixture
def mock_chunks():
    """Create mock chunks for testing"""
    chunks = []
    for i in range(10):
        chunk = Mock()
        chunk.index = i
        chunk.content = f"This is chunk {i}. It contains some text. Characters are talking. There's action happening!"
        chunk.metadata = {
            'position_ratio': i / 10.0,
            'start': i * 100,
            'end': (i + 1) * 100
        }
        chunks.append(chunk)
    return chunks


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings"""
    # Create 10 embeddings with varying similarity
    np.random.seed(42)
    embeddings = []
    for i in range(10):
        # Create embeddings with some variation
        base = np.random.randn(384)
        # Add some variance
        base = base / np.linalg.norm(base)
        embeddings.append(base)
    return embeddings


@pytest.fixture
def mock_llm_service():
    """Create mock LLM service"""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value='{"test": "response"}')
    return llm


# ==================== CharacterAnalyzer Tests ====================

class TestCharacterAnalyzerComprehensive:
    """Comprehensive tests for CharacterAnalyzer"""

    @pytest.mark.asyncio
    async def test_properties(self):
        """Test all analyzer properties"""
        from src.analysis_domain.analyzers.character_analyzer import CharacterAnalyzer

        analyzer = CharacterAnalyzer()

        assert analyzer.name == "character"
        assert analyzer.display_name == "Анализ персонажей"
        assert len(analyzer.description) > 0
        assert analyzer.requires_llm is True
        assert analyzer.requires_embeddings is False
        assert analyzer.max_chunks_to_analyze == 30

    @pytest.mark.asyncio
    async def test_initialization_with_params(self):
        """Test initialization with custom parameters"""
        from src.analysis_domain.analyzers.character_analyzer import CharacterAnalyzer

        mock_llm = AsyncMock()
        mock_prompt = Mock()

        analyzer = CharacterAnalyzer(
            llm_service=mock_llm,
            prompt_template=mock_prompt,
            max_chunks_to_analyze=20
        )

        assert analyzer.llm_service == mock_llm
        assert analyzer.prompt_template == mock_prompt
        assert analyzer.max_chunks_to_analyze == 20

    @pytest.mark.asyncio
    async def test_analyze_without_chunks(self, sample_text):
        """Test analyze fails without chunks"""
        from src.analysis_domain.analyzers.character_analyzer import CharacterAnalyzer

        analyzer = CharacterAnalyzer()

        with pytest.raises(AnalysisError) as exc_info:
            await analyzer.analyze(sample_text, AnalysisMode.FULL_TEXT)

        assert "требует чанки" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_analyze_with_empty_character_chunks(self, sample_text, mock_chunks):
        """Test analyze with chunks but no character-related content"""
        from src.analysis_domain.analyzers.character_analyzer import CharacterAnalyzer

        analyzer = CharacterAnalyzer()

        # Mock indexer to return no character chunks
        analyzer.indexer.build_character_index = Mock()
        analyzer.indexer.build_character_index.return_value = Mock(
            chunk_indices=[],
            coverage=0.0
        )
        analyzer.indexer.get_chunk_subset = Mock(return_value=[])

        result = await analyzer.analyze(
            sample_text,
            AnalysisMode.FULL_TEXT,
            chunks=mock_chunks
        )

        assert result.data['characters'] == []
        assert result.data['total_characters'] == 0
        assert result.data['chunks_analyzed'] == 0

    @pytest.mark.asyncio
    async def test_analyze_with_characters(self, russian_text, mock_chunks, mock_llm_service):
        """Test full character analysis with LLM"""
        from src.analysis_domain.analyzers.character_analyzer import CharacterAnalyzer

        # Setup LLM to return character data
        mock_llm_service.generate = AsyncMock(
            return_value='{"name": "Иван", "role": "main", "traits": [{"trait": "храбрый", "evidence": "мчался по лесу"}]}'
        )

        analyzer = CharacterAnalyzer(llm_service=mock_llm_service)

        # Mock indexer
        analyzer.indexer.build_character_index = Mock()
        analyzer.indexer.build_character_index.return_value = Mock(
            chunk_indices=[0, 1, 2],
            coverage=0.3
        )
        analyzer.indexer.get_chunk_subset = Mock(return_value=mock_chunks[:3])

        result = await analyzer.analyze(
            russian_text,
            AnalysisMode.FULL_TEXT,
            chunks=mock_chunks
        )

        assert result.text_id == russian_text.id
        assert result.analyzer_name == "character"
        assert 'characters' in result.data
        assert result.data['total_characters'] >= 0

    @pytest.mark.asyncio
    async def test_parse_character_response_valid_json(self):
        """Test parsing valid character JSON response"""
        from src.analysis_domain.analyzers.character_analyzer import CharacterAnalyzer

        analyzer = CharacterAnalyzer()
        response = '{"name": "Иван", "role": "main", "traits": []}'

        parsed = analyzer._parse_character_response(response)

        assert parsed['name'] == "Иван"
        assert parsed['role'] == "main"

    @pytest.mark.asyncio
    async def test_parse_character_response_invalid_json(self):
        """Test parsing invalid JSON returns empty dict"""
        from src.analysis_domain.analyzers.character_analyzer import CharacterAnalyzer

        analyzer = CharacterAnalyzer()
        response = "This is not valid JSON"

        parsed = analyzer._parse_character_response(response)

        assert parsed == {}

    @pytest.mark.asyncio
    async def test_parse_character_response_nested_json(self):
        """Test parsing nested JSON structures"""
        from src.analysis_domain.analyzers.character_analyzer import CharacterAnalyzer

        analyzer = CharacterAnalyzer()
        response = 'Some text before {"name": "Test", "role": "secondary", "traits": [{"trait": "smart", "evidence": "solved puzzle"}]} some text after'

        parsed = analyzer._parse_character_response(response)

        assert parsed['name'] == "Test"
        assert len(parsed['traits']) == 1

    @pytest.mark.asyncio
    async def test_aggregate_characters_empty(self):
        """Test aggregating empty character list"""
        from src.analysis_domain.analyzers.character_analyzer import CharacterAnalyzer

        analyzer = CharacterAnalyzer()
        result = analyzer._aggregate_characters([])

        assert result == []

    @pytest.mark.asyncio
    async def test_aggregate_characters_filters_invalid(self):
        """Test aggregating filters out invalid names"""
        from src.analysis_domain.analyzers.character_analyzer import CharacterAnalyzer

        analyzer = CharacterAnalyzer()
        mentions = [
            {'name': 'Имя Персонажа', 'role': 'main'},
            {'name': 'Неизвестный', 'role': 'secondary'},
            {'name': '', 'role': 'episodic'},
            {'name': 'Unknown', 'role': 'main'},
        ]

        result = analyzer._aggregate_characters(mentions)

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_aggregate_characters_combines_duplicates(self):
        """Test aggregating combines duplicate characters"""
        from src.analysis_domain.analyzers.character_analyzer import CharacterAnalyzer

        analyzer = CharacterAnalyzer()
        mentions = [
            {
                'name': 'Иван',
                'role': 'secondary',
                'traits': [{'trait': 'храбрый', 'evidence': 'test1'}],
                'position': 0.1,
                'chunk_index': 0
            },
            {
                'name': 'Иван',
                'role': 'main',
                'traits': [{'trait': 'умный', 'evidence': 'test2'}],
                'position': 0.5,
                'chunk_index': 5
            },
            {
                'name': 'Мария',
                'role': 'episodic',
                'traits': [],
                'position': 0.8,
                'chunk_index': 8
            }
        ]

        result = analyzer._aggregate_characters(mentions)

        # Should combine Иван into one character
        assert len(result) == 2
        ivan = next(c for c in result if c['name'] == 'Иван')
        assert ivan['role'] == 'main'  # Upgraded from secondary
        assert len(ivan['appearances']) == 2
        assert len(ivan['traits']) > 0

    @pytest.mark.asyncio
    async def test_aggregate_characters_limits_traits(self):
        """Test aggregating limits traits to top 5"""
        from src.analysis_domain.analyzers.character_analyzer import CharacterAnalyzer

        analyzer = CharacterAnalyzer()
        traits = [{'trait': f'trait_{i}', 'evidence': f'evidence_{i}'} for i in range(10)]
        mentions = [
            {
                'name': 'Тест',
                'role': 'main',
                'traits': traits,
                'position': 0.5,
                'chunk_index': 0
            }
        ]

        result = analyzer._aggregate_characters(mentions)

        assert len(result[0]['traits']) <= 5

    @pytest.mark.asyncio
    async def test_aggregate_characters_creates_timeline(self):
        """Test aggregating creates development timeline for multiple appearances"""
        from src.analysis_domain.analyzers.character_analyzer import CharacterAnalyzer

        analyzer = CharacterAnalyzer()
        mentions = [
            {'name': 'Герой', 'role': 'main', 'position': 0.1, 'chunk_index': 1},
            {'name': 'Герой', 'role': 'main', 'position': 0.9, 'chunk_index': 9},
        ]

        result = analyzer._aggregate_characters(mentions)

        assert len(result[0]['development_timeline']) == 2
        assert result[0]['development_timeline'][0]['description'] == "Первое появление"
        assert result[0]['development_timeline'][1]['description'] == "Последнее появление"

    @pytest.mark.asyncio
    async def test_interpret_results_no_characters(self):
        """Test interpretation with no characters"""
        from src.analysis_domain.analyzers.character_analyzer import CharacterAnalyzer
        from src.analysis_domain.entities.analysis_result import AnalysisResult

        analyzer = CharacterAnalyzer()
        result = AnalysisResult(
            text_id="test",
            analyzer_name="character",
            data={'characters': [], 'total_characters': 0, 'coverage': 0.0},
            execution_time_ms=100,
            mode="full_text"
        )

        interpretation = analyzer.interpret_results(result)

        assert "не обнаружены" in interpretation.lower()

    @pytest.mark.asyncio
    async def test_interpret_results_with_characters(self):
        """Test interpretation with characters"""
        from src.analysis_domain.analyzers.character_analyzer import CharacterAnalyzer
        from src.analysis_domain.entities.analysis_result import AnalysisResult

        analyzer = CharacterAnalyzer()
        result = AnalysisResult(
            text_id="test",
            analyzer_name="CharacterAnalyzer",
            data={
                'characters': [
                    {
                        'name': 'Главный Герой',
                        'role': 'main',
                        'traits': [
                            {'trait': 'храбрый', 'evidence': 'test1'},
                            {'trait': 'умный', 'evidence': 'test2'}
                        ],
                        'appearances': [{'position': 0.1, 'chunk_index': 0}]
                    },
                    {
                        'name': 'Злодей',
                        'role': 'secondary',
                        'traits': [],
                        'appearances': [{'position': 0.5, 'chunk_index': 5}]
                    }
                ],
                'total_characters': 2,
                'coverage': 0.3
            },
            execution_time_ms=100,
            mode="full_text"
        )

        interpretation = analyzer.interpret_results(result)

        assert "Главный Герой" in interpretation
        assert "храбрый" in interpretation
        assert "умный" in interpretation
        assert "Злодей" in interpretation

    @pytest.mark.asyncio
    async def test_format_character_prompt(self):
        """Test character prompt formatting"""
        from src.analysis_domain.analyzers.character_analyzer import CharacterAnalyzer

        analyzer = CharacterAnalyzer()
        prompt = analyzer._format_character_prompt("Test chunk text", "Context text")

        assert "Test chunk text" in prompt
        assert "Context text" in prompt
        assert "JSON" in prompt

    @pytest.mark.asyncio
    async def test_format_character_prompt_no_context(self):
        """Test character prompt formatting without context"""
        from src.analysis_domain.analyzers.character_analyzer import CharacterAnalyzer

        analyzer = CharacterAnalyzer()
        prompt = analyzer._format_character_prompt("Test chunk text")

        assert "Test chunk text" in prompt
        assert "Нет контекста" in prompt

    @pytest.mark.asyncio
    async def test_analyze_chunks_with_llm_error_handling(self, mock_chunks):
        """Test LLM analysis handles errors gracefully"""
        from src.analysis_domain.analyzers.character_analyzer import CharacterAnalyzer

        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM Error"))

        analyzer = CharacterAnalyzer(llm_service=mock_llm, max_chunks_to_analyze=2)

        result = await analyzer._analyze_chunks_with_llm(mock_chunks[:2])

        # Should return empty list on errors
        assert result == []


# ==================== PaceAnalyzer Tests ====================

class TestPaceAnalyzerComprehensive:
    """Comprehensive tests for PaceAnalyzer"""

    @pytest.mark.asyncio
    async def test_properties(self):
        """Test all analyzer properties"""
        from src.analysis_domain.analyzers.pace_analyzer import PaceAnalyzer

        analyzer = PaceAnalyzer()

        assert analyzer.name == "pace"
        assert analyzer.display_name == "Анализ темпа повествования"
        assert len(analyzer.description) > 0
        assert analyzer.requires_llm is False
        assert analyzer.requires_embeddings is False

    @pytest.mark.asyncio
    async def test_analyze_without_chunks(self, sample_text):
        """Test analyze fails without chunks"""
        from src.analysis_domain.analyzers.pace_analyzer import PaceAnalyzer

        analyzer = PaceAnalyzer()

        with pytest.raises(AnalysisError):
            await analyzer.analyze(sample_text, AnalysisMode.FULL_TEXT)

    @pytest.mark.asyncio
    async def test_analyze_with_chunks(self, sample_text, mock_chunks):
        """Test full pace analysis"""
        from src.analysis_domain.analyzers.pace_analyzer import PaceAnalyzer

        analyzer = PaceAnalyzer()

        # Mock the indexer methods
        analyzer.indexer._calculate_event_density = Mock(return_value=0.5)
        analyzer.indexer._calculate_dialogue_ratio = Mock(return_value=0.3)

        result = await analyzer.analyze(
            sample_text,
            AnalysisMode.FULL_TEXT,
            chunks=mock_chunks
        )

        assert result.text_id == sample_text.id
        assert 'overall_pace' in result.data
        assert 'pace_score' in result.data
        assert 'timeline' in result.data
        assert 'statistics' in result.data

    @pytest.mark.asyncio
    async def test_calculate_pace_score(self):
        """Test pace score calculation"""
        from src.analysis_domain.analyzers.pace_analyzer import PaceAnalyzer

        analyzer = PaceAnalyzer()

        # The formula is: score = (event_density * 7.0 + dialogue_ratio * 3.0) * 10, capped at 10.0

        # High event density, low dialogue: (1.0 * 7 + 0 * 3) * 10 = 70, capped at 10
        score1 = analyzer._calculate_pace_score(1.0, 0.0)
        assert score1 == 10.0

        # Low event density, high dialogue: (0 * 7 + 1.0 * 3) * 10 = 30, capped at 10
        score2 = analyzer._calculate_pace_score(0.0, 1.0)
        assert score2 == 10.0

        # Balanced: (0.5 * 7 + 0.5 * 3) * 10 = 50, capped at 10
        score3 = analyzer._calculate_pace_score(0.5, 0.5)
        assert score3 == 10.0

        # Low values: (0.1 * 7 + 0.1 * 3) * 10 = 10
        score4 = analyzer._calculate_pace_score(0.1, 0.1)
        assert abs(score4 - 10.0) < 0.01

    @pytest.mark.asyncio
    async def test_median_empty(self):
        """Test median with empty list"""
        from src.analysis_domain.analyzers.pace_analyzer import PaceAnalyzer

        analyzer = PaceAnalyzer()
        result = analyzer._median([])

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_median_odd_count(self):
        """Test median with odd number of elements"""
        from src.analysis_domain.analyzers.pace_analyzer import PaceAnalyzer

        analyzer = PaceAnalyzer()
        result = analyzer._median([1.0, 2.0, 3.0, 4.0, 5.0])

        assert result == 3.0

    @pytest.mark.asyncio
    async def test_median_even_count(self):
        """Test median with even number of elements"""
        from src.analysis_domain.analyzers.pace_analyzer import PaceAnalyzer

        analyzer = PaceAnalyzer()
        result = analyzer._median([1.0, 2.0, 3.0, 4.0])

        assert result == 2.5

    @pytest.mark.asyncio
    async def test_create_timeline_empty(self):
        """Test timeline creation with empty list"""
        from src.analysis_domain.analyzers.pace_analyzer import PaceAnalyzer

        analyzer = PaceAnalyzer()
        result = analyzer._create_timeline([])

        assert result == []

    @pytest.mark.asyncio
    async def test_create_timeline(self):
        """Test timeline creation"""
        from src.analysis_domain.analyzers.pace_analyzer import PaceAnalyzer

        analyzer = PaceAnalyzer()
        pace_scores = [
            {'position': 0.1, 'score': 5.0},
            {'position': 0.3, 'score': 7.0},
            {'position': 0.5, 'score': 6.0},
            {'position': 0.7, 'score': 8.0},
            {'position': 0.9, 'score': 4.0},
        ]

        result = analyzer._create_timeline(pace_scores, sample_rate=0.5)

        assert len(result) > 0
        assert all('position' in item and 'pace' in item for item in result)

    @pytest.mark.asyncio
    async def test_calculate_statistics_empty(self):
        """Test statistics calculation with empty list"""
        from src.analysis_domain.analyzers.pace_analyzer import PaceAnalyzer

        analyzer = PaceAnalyzer()
        result = analyzer._calculate_statistics([])

        assert result == {}

    @pytest.mark.asyncio
    async def test_calculate_statistics(self):
        """Test statistics calculation"""
        from src.analysis_domain.analyzers.pace_analyzer import PaceAnalyzer

        analyzer = PaceAnalyzer()
        pace_scores = [
            {'score': 5.0, 'event_density': 0.5, 'dialogue_ratio': 0.3},
            {'score': 7.0, 'event_density': 0.7, 'dialogue_ratio': 0.4},
            {'score': 6.0, 'event_density': 0.6, 'dialogue_ratio': 0.35},
        ]

        result = analyzer._calculate_statistics(pace_scores)

        assert 'min_pace' in result
        assert 'max_pace' in result
        assert 'avg_pace' in result
        assert 'median_pace' in result
        assert 'pace_variance' in result
        assert result['min_pace'] == 5.0
        assert result['max_pace'] == 7.0

    @pytest.mark.asyncio
    async def test_variance_empty(self):
        """Test variance with empty list"""
        from src.analysis_domain.analyzers.pace_analyzer import PaceAnalyzer

        analyzer = PaceAnalyzer()
        result = analyzer._variance([])

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_variance(self):
        """Test variance calculation"""
        from src.analysis_domain.analyzers.pace_analyzer import PaceAnalyzer

        analyzer = PaceAnalyzer()
        result = analyzer._variance([1.0, 2.0, 3.0, 4.0, 5.0])

        assert result == 2.0  # Variance of [1,2,3,4,5] is 2.0

    @pytest.mark.asyncio
    async def test_interpret_results_slow_pace(self):
        """Test interpretation for slow pace"""
        from src.analysis_domain.analyzers.pace_analyzer import PaceAnalyzer
        from src.analysis_domain.entities.analysis_result import AnalysisResult

        analyzer = PaceAnalyzer()
        result = AnalysisResult(
            text_id="test",
            analyzer_name="PaceAnalyzer",
            data={
                'overall_pace': 'slow',
                'pace_score': 3.0,
                'pace_emoji': '🐌',
                'pace_ru': 'медленный',
                'statistics': {'min_pace': 2.0, 'max_pace': 4.0}
            },
            execution_time_ms=100,
            mode="full_text"
        )

        interpretation = analyzer.interpret_results(result)

        assert "медленный" in interpretation.lower()
        assert "Медленный темп" in interpretation

    @pytest.mark.asyncio
    async def test_interpret_results_fast_pace(self):
        """Test interpretation for fast pace"""
        from src.analysis_domain.analyzers.pace_analyzer import PaceAnalyzer
        from src.analysis_domain.entities.analysis_result import AnalysisResult

        analyzer = PaceAnalyzer()
        result = AnalysisResult(
            text_id="test",
            analyzer_name="PaceAnalyzer",
            data={
                'overall_pace': 'fast',
                'pace_score': 8.0,
                'pace_emoji': '🏃',
                'pace_ru': 'быстрый',
                'statistics': {'min_pace': 7.0, 'max_pace': 9.0}
            },
            execution_time_ms=100,
            mode="full_text"
        )

        interpretation = analyzer.interpret_results(result)

        assert "быстрый" in interpretation.lower()
        assert "Быстрый темп" in interpretation


# ==================== Run coverage test ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/analysis_domain/analyzers"])
