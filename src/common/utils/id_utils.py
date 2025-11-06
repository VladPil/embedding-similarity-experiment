"""
Утилиты для генерации ID
"""
import uuid
import hashlib
from datetime import datetime


def generate_id(prefix: str = "") -> str:
    """
    Генерировать уникальный ID

    Args:
        prefix: Префикс для ID

    Returns:
        str: Уникальный ID
    """
    unique_str = f"{datetime.utcnow().isoformat()}{uuid.uuid4()}"
    hash_obj = hashlib.sha256(unique_str.encode())
    hash_id = hash_obj.hexdigest()[:16]

    if prefix:
        return f"{prefix}_{hash_id}"
    return hash_id


def generate_short_id() -> str:
    """
    Генерировать короткий уникальный ID (8 символов)

    Returns:
        str: Короткий ID
    """
    return uuid.uuid4().hex[:8]


def generate_uuid() -> str:
    """
    Генерировать UUID

    Returns:
        str: UUID строка
    """
    return str(uuid.uuid4())
