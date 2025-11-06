"""
Утилиты для валидации данных
"""
import re
from typing import Any, List, Optional
from pathlib import Path

from ..exceptions import ValidationError


def validate_not_empty(value: Any, field_name: str) -> None:
    """
    Проверить что значение не пустое

    Args:
        value: Значение для проверки
        field_name: Имя поля для сообщения об ошибке

    Raises:
        ValidationError: Если значение пустое
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValidationError(
            message=f"Field '{field_name}' cannot be empty",
            details={"field": field_name}
        )


def validate_string_length(
    value: str,
    field_name: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
) -> None:
    """
    Проверить длину строки

    Args:
        value: Строка для проверки
        field_name: Имя поля
        min_length: Минимальная длина
        max_length: Максимальная длина

    Raises:
        ValidationError: Если длина не соответствует
    """
    length = len(value)

    if min_length is not None and length < min_length:
        raise ValidationError(
            message=f"Field '{field_name}' must be at least {min_length} characters",
            details={"field": field_name, "min_length": min_length, "actual": length}
        )

    if max_length is not None and length > max_length:
        raise ValidationError(
            message=f"Field '{field_name}' must be at most {max_length} characters",
            details={"field": field_name, "max_length": max_length, "actual": length}
        )


def validate_list_length(
    value: List[Any],
    field_name: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
) -> None:
    """
    Проверить длину списка

    Args:
        value: Список для проверки
        field_name: Имя поля
        min_length: Минимальная длина
        max_length: Максимальная длина

    Raises:
        ValidationError: Если длина не соответствует
    """
    length = len(value)

    if min_length is not None and length < min_length:
        raise ValidationError(
            message=f"Field '{field_name}' must have at least {min_length} items",
            details={"field": field_name, "min_length": min_length, "actual": length}
        )

    if max_length is not None and length > max_length:
        raise ValidationError(
            message=f"Field '{field_name}' must have at most {max_length} items",
            details={"field": field_name, "max_length": max_length, "actual": length}
        )


def validate_range(
    value: float,
    field_name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> None:
    """
    Проверить что значение в диапазоне

    Args:
        value: Значение для проверки
        field_name: Имя поля
        min_value: Минимальное значение
        max_value: Максимальное значение

    Raises:
        ValidationError: Если значение вне диапазона
    """
    if min_value is not None and value < min_value:
        raise ValidationError(
            message=f"Field '{field_name}' must be at least {min_value}",
            details={"field": field_name, "min_value": min_value, "actual": value}
        )

    if max_value is not None and value > max_value:
        raise ValidationError(
            message=f"Field '{field_name}' must be at most {max_value}",
            details={"field": field_name, "max_value": max_value, "actual": value}
        )


def validate_email(email: str) -> None:
    """
    Проверить корректность email

    Args:
        email: Email для проверки

    Raises:
        ValidationError: Если email некорректный
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValidationError(
            message=f"Invalid email format: {email}",
            details={"email": email}
        )


def validate_file_path(file_path: str, must_exist: bool = False) -> None:
    """
    Проверить корректность пути к файлу

    Args:
        file_path: Путь к файлу
        must_exist: Должен ли файл существовать

    Raises:
        ValidationError: Если путь некорректный
    """
    path = Path(file_path)

    if must_exist and not path.exists():
        raise ValidationError(
            message=f"File does not exist: {file_path}",
            details={"file_path": file_path}
        )

    # Проверка на попытку выхода за пределы разрешённых директорий
    try:
        path.resolve()
    except (OSError, RuntimeError) as e:
        raise ValidationError(
            message=f"Invalid file path: {file_path}",
            details={"file_path": file_path, "error": str(e)}
        )


def validate_file_extension(file_path: str, allowed_extensions: List[str]) -> None:
    """
    Проверить расширение файла

    Args:
        file_path: Путь к файлу
        allowed_extensions: Список разрешённых расширений (без точки)

    Raises:
        ValidationError: Если расширение не разрешено
    """
    path = Path(file_path)
    extension = path.suffix.lstrip('.').lower()

    if extension not in [ext.lower() for ext in allowed_extensions]:
        raise ValidationError(
            message=f"File extension '{extension}' is not allowed",
            details={
                "file_path": file_path,
                "extension": extension,
                "allowed": allowed_extensions
            }
        )


def validate_id_format(id_value: str, field_name: str = "id") -> None:
    """
    Проверить формат ID (должен быть буквенно-цифровым)

    Args:
        id_value: ID для проверки
        field_name: Имя поля

    Raises:
        ValidationError: Если формат некорректный
    """
    if not id_value:
        raise ValidationError(
            message=f"Field '{field_name}' cannot be empty",
            details={"field": field_name}
        )

    # Разрешаем буквы, цифры, дефисы и подчёркивания
    pattern = r'^[a-zA-Z0-9_-]+$'
    if not re.match(pattern, id_value):
        raise ValidationError(
            message=f"Field '{field_name}' has invalid format. Only alphanumeric, dash and underscore allowed",
            details={"field": field_name, "value": id_value}
        )
