"""
Примитивные типы и type aliases
"""
from typing import NewType, Dict, Any


# ID типы для различных сущностей
TextID = NewType('TextID', str)
SessionID = NewType('SessionID', str)
AnalyzerID = NewType('AnalyzerID', str)
ComparatorID = NewType('ComparatorID', str)
ModelID = NewType('ModelID', str)
IndexID = NewType('IndexID', str)
TaskID = NewType('TaskID', str)
PromptID = NewType('PromptID', str)
UserID = NewType('UserID', str)

# Часто используемые типы
JSON = Dict[str, Any]
Metadata = Dict[str, Any]
