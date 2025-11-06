"""
Репозитории для работы с базой данных
"""
from .text_repository import TextRepository
from .session_repository import SessionRepository
from .model_config_repository import ModelConfigRepository
from .prompt_repository import PromptTemplateRepository

__all__ = [
    "TextRepository",
    "SessionRepository",
    "ModelConfigRepository",
    "PromptTemplateRepository",
]
