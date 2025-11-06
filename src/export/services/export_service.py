"""
Сервис для экспорта результатов анализа
"""
from typing import Optional, List
from pathlib import Path
from loguru import logger
import zipfile

from ..exporters.json_exporter import JSONExporter
from ..exporters.csv_exporter import CSVExporter
from ..exporters.pdf_exporter import PDFExporter
from src.analysis_domain.entities.analysis_session import AnalysisSession
from src.analysis_domain.entities.comparison_matrix import ComparisonMatrix
from src.common.exceptions import ExportError


class ExportService:
    """
    Сервис для экспорта результатов анализа в различные форматы

    Поддерживает экспорт в JSON, CSV, PDF, Markdown
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Инициализация сервиса экспорта

        Args:
            output_dir: Директория для сохранения файлов
        """
        self.output_dir = output_dir or "./data/exports"

        # Инициализируем экспортёры
        self.json_exporter = JSONExporter(output_dir=self.output_dir)
        self.csv_exporter = CSVExporter(output_dir=self.output_dir)
        self.pdf_exporter = PDFExporter(output_dir=self.output_dir)

    async def export_session(
        self,
        session: AnalysisSession,
        format: str = "json",
        file_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Экспортировать сессию в указанный формат

        Args:
            session: Сессия анализа
            format: Формат экспорта (json, csv, pdf, markdown)
            file_path: Путь к файлу (опционально)
            **kwargs: Дополнительные параметры

        Returns:
            str: Путь к созданному файлу

        Raises:
            ExportError: Если формат не поддерживается или ошибка экспорта
        """
        try:
            if format == "json":
                return await self.json_exporter.export_session(session, file_path)
            elif format == "csv":
                return await self.csv_exporter.export_session(session, file_path)
            elif format == "pdf":
                return await self.pdf_exporter.export_session(session, file_path)
            elif format == "markdown":
                return await self.export_session_markdown(session, file_path)
            else:
                raise ExportError(
                    message=f"Неподдерживаемый формат экспорта: {format}",
                    details={"supported_formats": ["json", "csv", "pdf", "markdown"]}
                )
        except Exception as e:
            logger.error(f"Ошибка экспорта сессии в {format}: {e}")
            raise ExportError(
                message=f"Не удалось экспортировать сессию в {format}",
                details={"error": str(e), "session_id": session.id}
            )

    async def export_comparison_matrix(
        self,
        matrix: ComparisonMatrix,
        format: str = "json",
        file_path: Optional[str] = None
    ) -> str:
        """
        Экспортировать матрицу сравнения

        Args:
            matrix: Матрица сравнения
            format: Формат экспорта
            file_path: Путь к файлу

        Returns:
            str: Путь к созданному файлу
        """
        try:
            if format == "json":
                return await self.json_exporter.export_comparison_matrix(matrix, file_path)
            elif format == "csv":
                return await self.csv_exporter.export_comparison_matrix(matrix, file_path)
            elif format == "pdf":
                return await self.pdf_exporter.export_comparison_matrix(matrix, file_path)
            else:
                raise ExportError(
                    message=f"Неподдерживаемый формат: {format}",
                    details={"supported_formats": ["json", "csv", "pdf"]}
                )
        except Exception as e:
            logger.error(f"Ошибка экспорта матрицы в {format}: {e}")
            raise ExportError(
                message=f"Не удалось экспортировать матрицу в {format}",
                details={"error": str(e)}
            )

    async def export_session_markdown(
        self,
        session: AnalysisSession,
        file_path: Optional[str] = None
    ) -> str:
        """
        Экспортировать сессию в Markdown формат

        Args:
            session: Сессия анализа
            file_path: Путь к файлу

        Returns:
            str: Путь к созданному файлу
        """
        # Генерируем путь если не задан
        if not file_path:
            from datetime import datetime
            safe_name = "".join(
                c if c.isalnum() or c in "._- " else "_"
                for c in session.name
            )
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_name}_{ts}.md"
            file_path = str(Path(self.output_dir) / filename)

        # Генерируем Markdown контент
        lines = []

        # Заголовок
        lines.append(f"# Отчёт по анализу: {session.name}\n")
        lines.append(f"**ID сессии:** `{session.id}`\n")
        lines.append(f"**Статус:** {session.status.value}\n")
        lines.append(f"**Режим:** {session.mode.value}\n")
        lines.append(f"**Создано:** {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if session.completed_at:
            lines.append(f"**Завершено:** {session.completed_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("\n---\n\n")

        # Тексты
        lines.append("## Тексты\n\n")
        for idx, text in enumerate(session.texts, 1):
            content_len = len(await text.get_content()) if hasattr(text, 'get_content') else 0
            lines.append(f"{idx}. **{text.title}**\n")
            lines.append(f"   - ID: `{text.id}`\n")
            lines.append(f"   - Длина: {content_len:,} символов\n\n")

        # Анализаторы
        if session.analyzers:
            lines.append("## Анализаторы\n\n")
            for analyzer in session.analyzers:
                lines.append(f"- {analyzer.__class__.__name__}")
                if analyzer.requires_llm:
                    lines.append(" *(требует LLM)*")
                lines.append("\n")
            lines.append("\n")

        # Результаты анализа
        if session.results:
            lines.append("## Результаты анализа\n\n")
            for text_id, result in session.results.items():
                lines.append(f"### Текст: {text_id[:8]}...\n\n")
                lines.append(f"**Анализатор:** {result.analyzer_type}\n\n")
                if hasattr(result, 'data') and result.data:
                    lines.append("**Данные:**\n\n")
                    lines.append(f"```json\n{result.data}\n```\n\n")

        # Матрица сравнения
        if session.comparison_matrix:
            lines.append("## Матрица сравнения\n\n")
            matrix = session.comparison_matrix
            avg_sim = matrix.get_average_similarity()
            lines.append(f"**Средняя схожесть:** {avg_sim:.4f}\n\n")

            # Топ похожих пар
            most_similar = matrix.get_most_similar_pairs(top_k=5)
            if most_similar:
                lines.append("### Топ-5 наиболее похожих пар\n\n")
                lines.append("| № | Текст 1 | Текст 2 | Схожесть |\n")
                lines.append("|---|---------|---------|----------|\n")
                for idx, pair in enumerate(most_similar, 1):
                    lines.append(
                        f"| {idx} | {pair['text1_id'][:15]}... | "
                        f"{pair['text2_id'][:15]}... | {pair['similarity']:.4f} |\n"
                    )
                lines.append("\n")

        # Сохраняем файл
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        logger.info(f"📝 Markdown экспорт завершён: {file_path}")
        return file_path

    async def batch_export(
        self,
        sessions: List[AnalysisSession],
        format: str = "json",
        create_archive: bool = True
    ) -> str:
        """
        Массовый экспорт нескольких сессий

        Args:
            sessions: Список сессий для экспорта
            format: Формат экспорта
            create_archive: Создать ZIP архив с файлами

        Returns:
            str: Путь к архиву или директории с файлами
        """
        if not sessions:
            raise ExportError(
                message="Список сессий пуст",
                details={}
            )

        exported_files = []

        # Экспортируем каждую сессию
        for session in sessions:
            try:
                file_path = await self.export_session(session, format=format)
                exported_files.append(file_path)
            except Exception as e:
                logger.warning(f"Не удалось экспортировать сессию {session.id}: {e}")

        if not exported_files:
            raise ExportError(
                message="Не удалось экспортировать ни одной сессии",
                details={"total_sessions": len(sessions)}
            )

        # Создаём архив если нужно
        if create_archive:
            from datetime import datetime
            archive_name = f"batch_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            archive_path = str(Path(self.output_dir) / archive_name)

            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in exported_files:
                    zipf.write(file_path, Path(file_path).name)

            logger.info(
                f"📦 Создан архив с {len(exported_files)} файлами: {archive_path}"
            )
            return archive_path
        else:
            logger.info(f"✅ Экспортировано {len(exported_files)} файлов")
            return self.output_dir

    def get_available_templates(self) -> List[dict]:
        """
        Получить список доступных шаблонов экспорта

        Returns:
            List[dict]: Список шаблонов
        """
        return [
            {
                "id": "default",
                "name": "Стандартный отчёт",
                "formats": ["json", "csv", "pdf", "markdown"],
                "description": "Полный отчёт со всеми результатами анализа"
            },
            {
                "id": "detailed_pdf",
                "name": "Детальный PDF с графиками",
                "formats": ["pdf"],
                "description": "PDF отчёт с визуализациями и подробной статистикой"
            },
            {
                "id": "summary",
                "name": "Краткая сводка",
                "formats": ["csv", "markdown"],
                "description": "Компактный отчёт с основными метриками"
            },
            {
                "id": "comparison_only",
                "name": "Только матрица сравнения",
                "formats": ["json", "csv", "pdf"],
                "description": "Экспорт только результатов сравнения текстов"
            }
        ]

    def get_supported_formats(self) -> List[str]:
        """
        Получить список поддерживаемых форматов

        Returns:
            List[str]: Список форматов
        """
        return ["json", "csv", "pdf", "markdown"]
