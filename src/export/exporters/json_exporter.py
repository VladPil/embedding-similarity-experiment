"""
Экспорт в JSON формат
"""
import json
from typing import Optional
from loguru import logger

from ..base.base_exporter import BaseExporter
from src.analysis_domain.entities.analysis_session import AnalysisSession
from src.analysis_domain.entities.comparison_matrix import ComparisonMatrix


class JSONExporter(BaseExporter):
    """
    Экспортёр в JSON формат

    Сохраняет полную структуру данных в читаемом JSON
    """

    def __init__(self, output_dir: Optional[str] = None, indent: int = 2):
        """
        Инициализация JSON экспортёра

        Args:
            output_dir: Директория для сохранения
            indent: Отступ для форматирования JSON
        """
        super().__init__(output_dir)
        self.indent = indent

    async def export_session(
        self,
        session: AnalysisSession,
        file_path: Optional[str] = None
    ) -> str:
        """
        Экспортировать сессию в JSON

        Args:
            session: Сессия анализа
            file_path: Путь к файлу

        Returns:
            str: Путь к созданному файлу
        """
        # Генерируем путь если не задан
        if not file_path:
            file_path = self._generate_filename(session.name, "json")

        # Подготавливаем данные
        data = self._prepare_session_data(session)

        # Добавляем метаинформацию
        export_data = {
            "export_format": "json",
            "export_version": "1.0",
            "data": data
        }

        # Сохраняем
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=self.indent, ensure_ascii=False)

        logger.info(f"📄 JSON экспорт завершён: {file_path}")

        return file_path

    async def export_comparison_matrix(
        self,
        matrix: ComparisonMatrix,
        file_path: Optional[str] = None
    ) -> str:
        """
        Экспортировать матрицу сравнения в JSON

        Args:
            matrix: Матрица сравнения
            file_path: Путь к файлу

        Returns:
            str: Путь к созданному файлу
        """
        # Генерируем путь если не задан
        if not file_path:
            file_path = self._generate_filename("comparison_matrix", "json")

        # Подготавливаем данные
        export_data = {
            "export_format": "json",
            "export_version": "1.0",
            "data": matrix.to_dict()
        }

        # Сохраняем
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=self.indent, ensure_ascii=False)

        logger.info(f"📄 JSON экспорт матрицы завершён: {file_path}")

        return file_path

    async def export_results_batch(
        self,
        sessions: list[AnalysisSession],
        file_path: Optional[str] = None
    ) -> str:
        """
        Экспортировать несколько сессий в один JSON файл

        Args:
            sessions: Список сессий
            file_path: Путь к файлу

        Returns:
            str: Путь к созданному файлу
        """
        # Генерируем путь если не задан
        if not file_path:
            file_path = self._generate_filename("batch_export", "json")

        # Подготавливаем данные для всех сессий
        sessions_data = []
        for session in sessions:
            data = self._prepare_session_data(session)
            sessions_data.append(data)

        # Формируем итоговый файл
        export_data = {
            "export_format": "json",
            "export_version": "1.0",
            "total_sessions": len(sessions),
            "sessions": sessions_data
        }

        # Сохраняем
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=self.indent, ensure_ascii=False)

        logger.info(f"📄 JSON batch экспорт завершён: {len(sessions)} сессий → {file_path}")

        return file_path
