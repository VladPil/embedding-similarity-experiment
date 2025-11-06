"""
Инфраструктурные исключения
"""
from .base import AppException


# База данных
class DatabaseError(AppException):
    """Ошибка базы данных"""
    pass


class DatabaseConnectionError(DatabaseError):
    """Ошибка подключения к базе данных"""
    pass


class DatabaseQueryError(DatabaseError):
    """Ошибка выполнения запроса"""
    pass


# Кэш
class CacheError(AppException):
    """Ошибка кэша"""
    pass


class CacheConnectionError(CacheError):
    """Ошибка подключения к кэшу"""
    pass


class CacheOperationError(CacheError):
    """Ошибка операции с кэшем"""
    pass


# Очередь задач
class QueueError(AppException):
    """Ошибка очереди"""
    pass


class TaskSubmissionError(QueueError):
    """Ошибка отправки задачи в очередь"""
    pass


class TaskExecutionError(QueueError):
    """Ошибка выполнения задачи"""
    pass


class TaskTimeoutError(QueueError):
    """Таймаут выполнения задачи"""
    def __init__(self, task_id: str, timeout: int):
        super().__init__(
            message=f"Task '{task_id}' timed out after {timeout} seconds",
            code="TASK_TIMEOUT",
            details={"task_id": task_id, "timeout": timeout}
        )


# Файловая система
class FileSystemError(AppException):
    """Ошибка файловой системы"""
    pass


class FileNotFoundError(FileSystemError):
    """Файл не найден"""
    def __init__(self, file_path: str):
        super().__init__(
            message=f"File '{file_path}' not found",
            code="FILE_NOT_FOUND",
            details={"file_path": file_path}
        )


class FileReadError(FileSystemError):
    """Ошибка чтения файла"""
    pass


class FileWriteError(FileSystemError):
    """Ошибка записи файла"""
    pass


# Настройки
class SettingsError(AppException):
    """Ошибка настроек"""
    pass


class SettingsNotFoundError(SettingsError):
    """Настройки не найдены"""
    pass
