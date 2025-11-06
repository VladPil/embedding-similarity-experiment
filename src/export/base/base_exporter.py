"""
Базовый класс для экспортёров
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

from src.analysis_domain.entities.analysis_session import AnalysisSession
from src.analysis_domain.entities.comparison_matrix import ComparisonMatrix


class BaseExporter(ABC):
    """
    Базовый класс для всех экспортёров

    Определяет общий интерфейс для экспорта результатов анализа
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Инициализация экспортёра

        Args:
            output_dir: Директория для сохранения файлов
        """
        self.output_dir = output_dir or "./exports"
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        """Создать директорию для экспорта если не существует"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def export_session(
        self,
        session: AnalysisSession,
        file_path: Optional[str] = None
    ) -> str:
        """
        Экспортировать сессию анализа

        Args:
            session: Сессия анализа
            file_path: Путь к файлу (если None - генерируется автоматически)

        Returns:
            str: Путь к созданному файлу
        """
        pass

    @abstractmethod
    async def export_comparison_matrix(
        self,
        matrix: ComparisonMatrix,
        file_path: Optional[str] = None
    ) -> str:
        """
        Экспортировать матрицу сравнения

        Args:
            matrix: Матрица сравнения
            file_path: Путь к файлу

        Returns:
            str: Путь к созданному файлу
        """
        pass

    def _generate_filename(
        self,
        session_name: str,
        extension: str,
        timestamp: bool = True
    ) -> str:
        """
        Сгенерировать имя файла

        Args:
            session_name: Название сессии
            extension: Расширение файла
            timestamp: Добавить timestamp

        Returns:
            str: Полный путь к файлу
        """
        # Очищаем имя сессии от недопустимых символов
        safe_name = "".join(
            c if c.isalnum() or c in "._- " else "_"
            for c in session_name
        )

        # Добавляем timestamp если нужно
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_name}_{ts}.{extension}"
        else:
            filename = f"{safe_name}.{extension}"

        return str(Path(self.output_dir) / filename)

    def _prepare_session_data(self, session: AnalysisSession) -> Dict[str, Any]:
        """
        Подготовить данные сессии для экспорта

        Args:
            session: Сессия анализа

        Returns:
            Dict: Данные для экспорта
        """
        return {
            "session_id": session.id,
            "session_name": session.name,
            "status": session.status.value,
            "mode": session.mode.value,
            "created_at": session.created_at.isoformat(),
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "texts": [
                {
                    "id": text.id,
                    "title": text.title,
                    "length": len(await text.get_content()) if hasattr(text, 'get_content') else 0,
                }
                for text in session.texts
            ],
            "analyzers": [
                {
                    "type": analyzer.__class__.__name__,
                    "requires_llm": analyzer.requires_llm,
                }
                for analyzer in session.analyzers
            ],
            "results": {
                text_id: result.to_dict()
                for text_id, result in session.results.items()
            },
            "comparison_matrix": session.comparison_matrix.to_dict() if session.comparison_matrix else None,
        }
