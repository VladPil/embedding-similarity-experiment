"""
Утилиты для работы с датой и временем
"""
from datetime import datetime, timezone
from typing import Optional


def now_utc() -> datetime:
    """
    Получить текущее время в UTC

    Returns:
        Текущее время в UTC
    """
    return datetime.now(timezone.utc)


def to_utc(dt: datetime) -> datetime:
    """
    Конвертировать datetime в UTC

    Args:
        dt: Datetime объект

    Returns:
        Datetime в UTC
    """
    if dt.tzinfo is None:
        # Если timezone не указан, считаем что это UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_datetime(dt: Optional[datetime], fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Форматировать datetime в строку

    Args:
        dt: Datetime объект
        fmt: Формат строки

    Returns:
        Отформатированная строка или "N/A" если dt is None
    """
    if dt is None:
        return "N/A"
    return dt.strftime(fmt)


def timestamp_to_datetime(timestamp: float) -> datetime:
    """
    Конвертировать Unix timestamp в datetime

    Args:
        timestamp: Unix timestamp

    Returns:
        Datetime объект в UTC
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def datetime_to_timestamp(dt: datetime) -> float:
    """
    Конвертировать datetime в Unix timestamp

    Args:
        dt: Datetime объект

    Returns:
        Unix timestamp
    """
    return dt.timestamp()


def calculate_duration(start: datetime, end: datetime) -> float:
    """
    Вычислить продолжительность между двумя датами в секундах

    Args:
        start: Начальная дата
        end: Конечная дата

    Returns:
        Продолжительность в секундах
    """
    return (end - start).total_seconds()


def format_duration(seconds: float) -> str:
    """
    Форматировать длительность в человекочитаемый вид

    Args:
        seconds: Количество секунд

    Returns:
        Строка вида "1h 23m 45s" или "45.2s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"

    hours = minutes // 60
    minutes = minutes % 60

    return f"{hours}h {minutes}m {remaining_seconds}s"
