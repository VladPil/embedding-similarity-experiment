"""
Тесты для сессии анализа
"""
import pytest
from datetime import datetime

from src.analysis_domain.entities.analysis_session import AnalysisSession
from src.analysis_domain.entities.analysis_result import AnalysisResult
from src.analysis_domain.analyzers.tfidf_analyzer import TfidfAnalyzer
from src.analysis_domain.analyzers.genre_analyzer import GenreAnalyzer
from src.text_domain.entities.plain_text import PlainText
from src.common.types import SessionStatus, AnalysisMode
from src.common.exceptions import ValidationError


@pytest.fixture
def sample_session():
    """Создать тестовую сессию"""
    return AnalysisSession(
        id="session-test-1",
        name="Тестовая сессия",
        status=SessionStatus.DRAFT
    )


@pytest.fixture
def sample_text_1():
    """Первый тестовый текст"""
    return PlainText.create_from_string(
        text_id="text-1",
        title="Текст 1",
        content="Это первый тестовый текст для анализа.",
        language="ru"
    )


@pytest.fixture
def sample_text_2():
    """Второй тестовый текст"""
    return PlainText.create_from_string(
        text_id="text-2",
        title="Текст 2",
        content="Это второй тестовый текст для сравнения.",
        language="ru"
    )


class TestAnalysisSessionBasics:
    """Базовые тесты для сессии анализа"""

    def test_session_creation(self):
        """Тест создания сессии"""
        session = AnalysisSession(
            id="test-session",
            name="Test Session"
        )

        assert session.id == "test-session"
        assert session.name == "Test Session"
        assert session.status == SessionStatus.DRAFT
        assert session.progress == 0
        assert len(session.texts) == 0
        assert len(session.analyzers) == 0
        assert session.comparator is None

    def test_session_properties(self, sample_session):
        """Тест свойств сессии"""
        assert sample_session.created_at is not None
        assert isinstance(sample_session.created_at, datetime)
        assert sample_session.mode == AnalysisMode.FULL_TEXT
        assert sample_session.use_faiss_search is False


class TestSessionTextManagement:
    """Тесты управления текстами в сессии"""

    def test_add_text(self, sample_session, sample_text_1):
        """Тест добавления текста"""
        sample_session.add_text(sample_text_1)

        assert len(sample_session.texts) == 1
        assert sample_session.texts[0].id == "text-1"

    def test_add_multiple_texts(self, sample_session, sample_text_1, sample_text_2):
        """Тест добавления нескольких текстов"""
        sample_session.add_text(sample_text_1)
        sample_session.add_text(sample_text_2)

        assert len(sample_session.texts) == 2
        assert sample_session.texts[0].id == "text-1"
        assert sample_session.texts[1].id == "text-2"

    def test_add_duplicate_text(self, sample_session, sample_text_1):
        """Тест добавления дубликата текста"""
        sample_session.add_text(sample_text_1)

        with pytest.raises(ValidationError) as exc_info:
            sample_session.add_text(sample_text_1)

        assert "already in the session" in str(exc_info.value)

    def test_add_text_limit(self, sample_session):
        """Тест лимита текстов (максимум 5)"""
        # Добавляем 5 текстов
        for i in range(5):
            text = PlainText.create_from_string(
                text_id=f"text-{i}",
                title=f"Text {i}",
                content=f"Content {i}"
            )
            sample_session.add_text(text)

        assert len(sample_session.texts) == 5

        # Попытка добавить 6-й текст
        text_6 = PlainText.create_from_string(
            text_id="text-6",
            title="Text 6",
            content="Content 6"
        )

        with pytest.raises(ValidationError) as exc_info:
            sample_session.add_text(text_6)

        assert "Maximum 5 texts" in str(exc_info.value)

    def test_remove_text(self, sample_session, sample_text_1, sample_text_2):
        """Тест удаления текста"""
        sample_session.add_text(sample_text_1)
        sample_session.add_text(sample_text_2)

        removed = sample_session.remove_text("text-1")

        assert removed is True
        assert len(sample_session.texts) == 1
        assert sample_session.texts[0].id == "text-2"

    def test_remove_nonexistent_text(self, sample_session, sample_text_1):
        """Тест удаления несуществующего текста"""
        sample_session.add_text(sample_text_1)

        removed = sample_session.remove_text("nonexistent")

        assert removed is False
        assert len(sample_session.texts) == 1

    def test_get_text_by_id(self, sample_session, sample_text_1, sample_text_2):
        """Тест получения текста по ID"""
        sample_session.add_text(sample_text_1)
        sample_session.add_text(sample_text_2)

        text = sample_session.get_text_by_id("text-1")

        assert text is not None
        assert text.id == "text-1"

        nonexistent = sample_session.get_text_by_id("nonexistent")
        assert nonexistent is None


class TestSessionAnalyzerManagement:
    """Тесты управления анализаторами в сессии"""

    def test_add_analyzer(self, sample_session):
        """Тест добавления анализатора"""
        analyzer = TfidfAnalyzer()
        sample_session.add_analyzer(analyzer)

        assert len(sample_session.analyzers) == 1
        assert sample_session.analyzers[0].name == "tfidf"

    def test_add_multiple_analyzers(self, sample_session):
        """Тест добавления нескольких анализаторов"""
        tfidf = TfidfAnalyzer()
        genre = GenreAnalyzer()

        sample_session.add_analyzer(tfidf)
        sample_session.add_analyzer(genre)

        assert len(sample_session.analyzers) == 2
        assert sample_session.analyzers[0].name == "tfidf"
        assert sample_session.analyzers[1].name == "genre"

    def test_add_duplicate_analyzer(self, sample_session):
        """Тест добавления дубликата анализатора"""
        analyzer = TfidfAnalyzer()
        sample_session.add_analyzer(analyzer)

        with pytest.raises(ValidationError) as exc_info:
            sample_session.add_analyzer(analyzer)

        assert "already in the session" in str(exc_info.value)

    def test_remove_analyzer(self, sample_session):
        """Тест удаления анализатора"""
        tfidf = TfidfAnalyzer()
        genre = GenreAnalyzer()

        sample_session.add_analyzer(tfidf)
        sample_session.add_analyzer(genre)

        removed = sample_session.remove_analyzer("tfidf")

        assert removed is True
        assert len(sample_session.analyzers) == 1
        assert sample_session.analyzers[0].name == "genre"

    def test_get_analyzer_by_name(self, sample_session):
        """Тест получения анализатора по имени"""
        tfidf = TfidfAnalyzer()
        sample_session.add_analyzer(tfidf)

        analyzer = sample_session.get_analyzer_by_name("tfidf")

        assert analyzer is not None
        assert analyzer.name == "tfidf"

        nonexistent = sample_session.get_analyzer_by_name("nonexistent")
        assert nonexistent is None


class TestSessionResultsManagement:
    """Тесты управления результатами в сессии"""

    def test_set_and_get_result(self, sample_session, sample_text_1):
        """Тест установки и получения результата"""
        result = AnalysisResult(
            text_id="text-1",
            analyzer_name="tfidf",
            data={"test": "data"},
            execution_time_ms=100.0,
            mode="full_text"
        )

        sample_session.set_result("text-1", "tfidf", result)

        retrieved = sample_session.get_result("text-1", "tfidf")

        assert retrieved is not None
        assert retrieved.text_id == "text-1"
        assert retrieved.analyzer_name == "tfidf"
        assert retrieved.data == {"test": "data"}

    def test_get_nonexistent_result(self, sample_session):
        """Тест получения несуществующего результата"""
        result = sample_session.get_result("text-1", "tfidf")
        assert result is None


class TestSessionValidation:
    """Тесты валидации сессии"""

    def test_validate_empty_session(self, sample_session):
        """Тест валидации пустой сессии"""
        with pytest.raises(ValidationError) as exc_info:
            sample_session.validate()

        assert "at least one text" in str(exc_info.value)

    def test_validate_session_without_analyzers(self, sample_session, sample_text_1):
        """Тест валидации сессии без анализаторов"""
        sample_session.add_text(sample_text_1)

        with pytest.raises(ValidationError) as exc_info:
            sample_session.validate()

        assert "at least one analyzer" in str(exc_info.value)

    def test_validate_valid_single_text_session(self, sample_session, sample_text_1):
        """Тест валидации валидной сессии с одним текстом"""
        sample_session.add_text(sample_text_1)
        sample_session.add_analyzer(TfidfAnalyzer())

        is_valid = sample_session.validate()
        assert is_valid is True

    def test_validate_multiple_texts_without_comparator(
        self, sample_session, sample_text_1, sample_text_2
    ):
        """Тест валидации сессии с несколькими текстами без компаратора"""
        sample_session.add_text(sample_text_1)
        sample_session.add_text(sample_text_2)
        sample_session.add_analyzer(TfidfAnalyzer())

        with pytest.raises(ValidationError) as exc_info:
            sample_session.validate()

        assert "must have a comparator" in str(exc_info.value)


class TestSessionStatusManagement:
    """Тесты управления статусом сессии"""

    def test_is_completed(self, sample_session):
        """Тест проверки завершения сессии"""
        assert sample_session.is_completed() is False

        sample_session.status = SessionStatus.COMPLETED
        assert sample_session.is_completed() is True

    def test_is_running(self, sample_session):
        """Тест проверки выполнения сессии"""
        assert sample_session.is_running() is False

        sample_session.status = SessionStatus.RUNNING
        assert sample_session.is_running() is True

    def test_has_error(self, sample_session):
        """Тест проверки ошибки в сессии"""
        assert sample_session.has_error() is False

        sample_session.status = SessionStatus.FAILED
        assert sample_session.has_error() is True


class TestSessionEstimations:
    """Тесты оценок времени выполнения"""

    def test_estimate_total_time_single_text(self, sample_session, sample_text_1):
        """Тест оценки времени для одного текста"""
        sample_session.add_text(sample_text_1)
        sample_session.add_analyzer(TfidfAnalyzer())

        estimated_time = sample_session.estimate_total_time()

        assert estimated_time > 0
        assert isinstance(estimated_time, float)

    def test_estimate_total_time_multiple_analyzers(
        self, sample_session, sample_text_1
    ):
        """Тест оценки времени для нескольких анализаторов"""
        sample_session.add_text(sample_text_1)
        sample_session.add_analyzer(TfidfAnalyzer())
        sample_session.add_analyzer(GenreAnalyzer())

        estimated_time = sample_session.estimate_total_time()

        # Время должно быть больше чем для одного анализатора
        assert estimated_time > 5.0  # GenreAnalyzer требует больше времени


class TestSessionSerialization:
    """Тесты сериализации сессии"""

    def test_to_dict(self, sample_session, sample_text_1):
        """Тест сериализации сессии в словарь"""
        sample_session.add_text(sample_text_1)
        sample_session.add_analyzer(TfidfAnalyzer())

        session_dict = sample_session.to_dict()

        assert isinstance(session_dict, dict)
        assert session_dict["id"] == "session-test-1"
        assert session_dict["name"] == "Тестовая сессия"
        assert session_dict["status"] == "draft"
        assert session_dict["text_count"] == 1
        assert session_dict["analyzer_count"] == 1
        assert "text-1" in session_dict["text_ids"]
        assert "tfidf" in session_dict["analyzer_names"]

    def test_str_representation(self, sample_session, sample_text_1):
        """Тест строкового представления сессии"""
        sample_session.add_text(sample_text_1)

        str_repr = str(sample_session)

        assert "session-test-1" in str_repr
        assert "Тестовая сессия" in str_repr
        assert "draft" in str_repr
