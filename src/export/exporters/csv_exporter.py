"""
Экспорт в CSV формат
"""
import csv
from typing import Optional, List, Dict, Any
from loguru import logger
import numpy as np

from ..base.base_exporter import BaseExporter
from src.analysis_domain.entities.analysis_session import AnalysisSession
from src.analysis_domain.entities.comparison_matrix import ComparisonMatrix


class CSVExporter(BaseExporter):
    """
    Экспортёр в CSV формат

    Особенно удобен для матриц сравнения
    """

    def __init__(self, output_dir: Optional[str] = None, delimiter: str = ","):
        """
        Инициализация CSV экспортёра

        Args:
            output_dir: Директория для сохранения
            delimiter: Разделитель (по умолчанию запятая)
        """
        super().__init__(output_dir)
        self.delimiter = delimiter

    async def export_session(
        self,
        session: AnalysisSession,
        file_path: Optional[str] = None
    ) -> str:
        """
        Экспортировать сессию в CSV

        Args:
            session: Сессия анализа
            file_path: Путь к файлу

        Returns:
            str: Путь к созданному файлу
        """
        # Генерируем путь если не задан
        if not file_path:
            file_path = self._generate_filename(session.name, "csv")

        # Подготавливаем данные в табличном формате
        rows = []

        # Заголовок
        rows.append([
            "Session ID", session.id,
            "Name", session.name,
            "Status", session.status.value,
            "Mode", session.mode.value
        ])
        rows.append([])  # Пустая строка

        # Информация о текстах
        rows.append(["Text ID", "Title", "Length"])
        for text in session.texts:
            rows.append([
                text.id,
                text.title,
                len(await text.get_content()) if hasattr(text, 'get_content') else 0
            ])
        rows.append([])

        # Результаты анализа
        if session.results:
            rows.append(["Text ID", "Analyzer", "Result Type", "Summary"])
            for text_id, result in session.results.items():
                rows.append([
                    text_id,
                    result.analyzer_type,
                    result.result_type.value if hasattr(result, 'result_type') else "N/A",
                    str(result.data)[:100] + "..." if len(str(result.data)) > 100 else str(result.data)
                ])
            rows.append([])

        # Матрица сравнения (если есть)
        if session.comparison_matrix:
            await self._add_matrix_to_rows(session.comparison_matrix, rows)

        # Сохраняем
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=self.delimiter)
            writer.writerows(rows)

        logger.info(f"📊 CSV экспорт завершён: {file_path}")

        return file_path

    async def export_comparison_matrix(
        self,
        matrix: ComparisonMatrix,
        file_path: Optional[str] = None
    ) -> str:
        """
        Экспортировать матрицу сравнения в CSV

        Args:
            matrix: Матрица сравнения
            file_path: Путь к файлу

        Returns:
            str: Путь к созданному файлу
        """
        # Генерируем путь если не задан
        if not file_path:
            file_path = self._generate_filename("comparison_matrix", "csv")

        rows = []

        # Добавляем матрицу
        await self._add_matrix_to_rows(matrix, rows)

        # Сохраняем
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=self.delimiter)
            writer.writerows(rows)

        logger.info(f"📊 CSV экспорт матрицы завершён: {file_path}")

        return file_path

    async def _add_matrix_to_rows(
        self,
        matrix: ComparisonMatrix,
        rows: List[List[Any]]
    ) -> None:
        """
        Добавить матрицу сравнения в список строк

        Args:
            matrix: Матрица сравнения
            rows: Список строк для добавления
        """
        rows.append(["Comparison Matrix"])
        rows.append([])

        # Заголовок матрицы
        text_ids = matrix.text_ids
        header = [""] + text_ids  # Первая колонка пустая для ID строк
        rows.append(header)

        # Строки матрицы
        similarity_matrix = matrix.similarity_matrix

        for i, row_id in enumerate(text_ids):
            row = [row_id]  # ID строки
            for j in range(len(text_ids)):
                value = similarity_matrix[i, j]
                # Форматируем до 4 знаков после запятой
                row.append(f"{value:.4f}")
            rows.append(row)

        rows.append([])

        # Статистика
        rows.append(["Statistics"])
        rows.append(["Metric", "Value"])
        rows.append(["Average Similarity", f"{matrix.get_average_similarity():.4f}"])

        # Самые похожие пары
        most_similar = matrix.get_most_similar_pairs(top_k=5)
        rows.append([])
        rows.append(["Most Similar Pairs (Top 5)"])
        rows.append(["Text 1", "Text 2", "Similarity"])

        for pair in most_similar:
            rows.append([
                pair["text1_id"],
                pair["text2_id"],
                f"{pair['similarity']:.4f}"
            ])

    async def export_results_table(
        self,
        sessions: List[AnalysisSession],
        file_path: Optional[str] = None
    ) -> str:
        """
        Экспортировать результаты нескольких сессий в таблицу

        Args:
            sessions: Список сессий
            file_path: Путь к файлу

        Returns:
            str: Путь к созданному файлу
        """
        # Генерируем путь если не задан
        if not file_path:
            file_path = self._generate_filename("results_table", "csv")

        rows = []

        # Заголовок
        rows.append([
            "Session ID",
            "Session Name",
            "Status",
            "Mode",
            "Texts Count",
            "Analyzers Count",
            "Average Similarity",
            "Created At",
            "Completed At"
        ])

        # Данные
        for session in sessions:
            avg_sim = "N/A"
            if session.comparison_matrix:
                avg_sim = f"{session.comparison_matrix.get_average_similarity():.4f}"

            rows.append([
                session.id,
                session.name,
                session.status.value,
                session.mode.value,
                len(session.texts),
                len(session.analyzers),
                avg_sim,
                session.created_at.isoformat(),
                session.completed_at.isoformat() if session.completed_at else "N/A"
            ])

        # Сохраняем
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=self.delimiter)
            writer.writerows(rows)

        logger.info(f"📊 CSV таблица результатов экспортирована: {file_path}")

        return file_path
