"""
Custom exceptions for the application.
"""


class AppException(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class TextNotFoundError(AppException):
    """Raised when a text is not found."""
    pass


class TextProcessingError(AppException):
    """Raised when text processing fails."""
    pass


class ModelLoadError(AppException):
    """Raised when model loading fails."""
    pass


class AnalysisError(AppException):
    """Raised when analysis fails."""
    pass


class CacheError(AppException):
    """Raised when cache operations fail."""
    pass


class TaskError(AppException):
    """Raised when task operations fail."""
    pass


class ValidationError(AppException):
    """Raised when validation fails."""
    pass
