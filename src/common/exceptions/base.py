"""
Базовые исключения приложения
"""
from typing import Optional, Any, Dict


class AppException(Exception):
    """Базовое исключение для всего приложения"""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, code={self.code!r})"

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация для API ответов"""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }
