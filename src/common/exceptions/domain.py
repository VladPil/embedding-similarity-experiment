"""
Доменные исключения
"""
from .base import AppException


# Текстовый домен
class TextNotFoundError(AppException):
    """Текст не найден"""
    def __init__(self, text_id: str):
        super().__init__(
            message=f"Text with id '{text_id}' not found",
            code="TEXT_NOT_FOUND",
            details={"text_id": text_id}
        )


class TextProcessingError(AppException):
    """Ошибка при обработке текста"""
    pass


class FB2ParseError(AppException):
    """Ошибка при парсинге FB2"""
    pass


class ChunkingError(AppException):
    """Ошибка при разбиении на чанки"""
    pass


# Домен анализа
class AnalysisError(AppException):
    """Базовая ошибка анализа"""
    pass


class SessionNotFoundError(AppException):
    """Сессия не найдена"""
    def __init__(self, session_id: str):
        super().__init__(
            message=f"Analysis session with id '{session_id}' not found",
            code="SESSION_NOT_FOUND",
            details={"session_id": session_id}
        )


class SessionValidationError(AppException):
    """Ошибка валидации сессии"""
    pass


class AnalyzerNotFoundError(AppException):
    """Анализатор не найден"""
    def __init__(self, analyzer_name: str):
        super().__init__(
            message=f"Analyzer '{analyzer_name}' not found",
            code="ANALYZER_NOT_FOUND",
            details={"analyzer_name": analyzer_name}
        )


class ComparatorNotFoundError(AppException):
    """Компаратор не найден"""
    def __init__(self, comparator_name: str):
        super().__init__(
            message=f"Comparator '{comparator_name}' not found",
            code="COMPARATOR_NOT_FOUND",
            details={"comparator_name": comparator_name}
        )


class PromptNotFoundError(AppException):
    """Промпт не найден"""
    def __init__(self, prompt_id: str):
        super().__init__(
            message=f"Prompt template with id '{prompt_id}' not found",
            code="PROMPT_NOT_FOUND",
            details={"prompt_id": prompt_id}
        )


# Управление моделями
class ModelError(AppException):
    """Базовая ошибка моделей"""
    pass


class ModelNotFoundError(ModelError):
    """Модель не найдена"""
    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' not found",
            code="MODEL_NOT_FOUND",
            details={"model_name": model_name}
        )


class ModelLoadError(ModelError):
    """Ошибка загрузки модели"""
    pass


class ModelDownloadError(ModelError):
    """Ошибка скачивания модели"""
    pass


class ModelInferenceError(ModelError):
    """Ошибка инференса модели"""
    pass


class GPUMemoryError(ModelError):
    """Недостаточно памяти GPU"""
    pass


# Векторное хранилище
class VectorStoreError(AppException):
    """Базовая ошибка векторного хранилища"""
    pass


class IndexNotFoundError(VectorStoreError):
    """Индекс не найден"""
    def __init__(self, index_id: str):
        super().__init__(
            message=f"FAISS index with id '{index_id}' not found",
            code="INDEX_NOT_FOUND",
            details={"index_id": index_id}
        )


class IndexBuildError(VectorStoreError):
    """Ошибка построения индекса"""
    pass


class SearchError(VectorStoreError):
    """Ошибка поиска"""
    pass


# Экспорт
class ExportError(AppException):
    """Ошибка экспорта"""
    pass


class UnsupportedFormatError(ExportError):
    """Неподдерживаемый формат экспорта"""
    def __init__(self, format_name: str):
        super().__init__(
            message=f"Export format '{format_name}' is not supported",
            code="UNSUPPORTED_FORMAT",
            details={"format": format_name}
        )


# Валидация
class ValidationError(AppException):
    """Ошибка валидации"""
    pass
