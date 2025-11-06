"""
>4C;L 8A:;NG5=89
"""
from .base import AppException

# ><5==K5 8A:;NG5=8O
from .domain import (
    # "5:AB>2K9 4><5=
    TextNotFoundError,
    TextProcessingError,
    FB2ParseError,
    ChunkingError,
    # ><5= 0=0;870
    AnalysisError,
    SessionNotFoundError,
    SessionValidationError,
    AnalyzerNotFoundError,
    ComparatorNotFoundError,
    PromptNotFoundError,
    # #?@02;5=85 <>45;O<8
    ModelError,
    ModelNotFoundError,
    ModelLoadError,
    ModelDownloadError,
    ModelInferenceError,
    GPUMemoryError,
    # 5:B>@=>5 E@0=8;8I5
    VectorStoreError,
    IndexNotFoundError,
    IndexBuildError,
    SearchError,
    # -:A?>@B
    ExportError,
    UnsupportedFormatError,
    # 0;840F8O
    ValidationError,
)

# =D@0AB@C:BC@=K5 8A:;NG5=8O
from .infrastructure import (
    DatabaseError,
    DatabaseConnectionError,
    DatabaseQueryError,
    DatabaseOperationError,
    CacheError,
    CacheConnectionError,
    CacheOperationError,
    QueueError,
    TaskSubmissionError,
    TaskExecutionError,
    TaskTimeoutError,
    FileSystemError,
    FileNotFoundError,
    FileReadError,
    FileWriteError,
    SettingsError,
    SettingsNotFoundError,
)

__all__ = [
    # Base
    "AppException",
    # Domain
    "TextNotFoundError",
    "TextProcessingError",
    "FB2ParseError",
    "ChunkingError",
    "AnalysisError",
    "SessionNotFoundError",
    "SessionValidationError",
    "AnalyzerNotFoundError",
    "ComparatorNotFoundError",
    "PromptNotFoundError",
    "ModelError",
    "ModelNotFoundError",
    "ModelLoadError",
    "ModelDownloadError",
    "ModelInferenceError",
    "GPUMemoryError",
    "VectorStoreError",
    "IndexNotFoundError",
    "IndexBuildError",
    "SearchError",
    "ExportError",
    "UnsupportedFormatError",
    "ValidationError",
    # Infrastructure
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseQueryError",
    "DatabaseOperationError",
    "CacheError",
    "CacheConnectionError",
    "CacheOperationError",
    "QueueError",
    "TaskSubmissionError",
    "TaskExecutionError",
    "TaskTimeoutError",
    "FileSystemError",
    "FileNotFoundError",
    "FileReadError",
    "FileWriteError",
    "SettingsError",
    "SettingsNotFoundError",
]
