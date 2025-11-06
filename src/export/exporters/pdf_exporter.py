"""
Экспорт в PDF формат с графиками
"""
from typing import Optional, List
from loguru import logger
import io

from ..base.base_exporter import BaseExporter
from src.analysis_domain.entities.analysis_session import AnalysisSession
from src.analysis_domain.entities.comparison_matrix import ComparisonMatrix


class PDFExporter(BaseExporter):
    """
    Экспортёр в PDF формат с визуализацией

    Создаёт красивые отчёты с графиками и таблицами
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Инициализация PDF экспортёра

        Args:
            output_dir: Директория для сохранения
        """
        super().__init__(output_dir)

        # Проверяем наличие необходимых библиотек
        try:
            from reportlab.lib.pagesizes import A4, letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
            from reportlab.lib import colors
            import matplotlib
            matplotlib.use('Agg')  # Non-GUI backend
            import matplotlib.pyplot as plt
            import seaborn as sns

            self._has_dependencies = True
        except ImportError as e:
            logger.warning(f"PDF экспорт недоступен, отсутствуют зависимости: {e}")
            self._has_dependencies = False

    async def export_session(
        self,
        session: AnalysisSession,
        file_path: Optional[str] = None
    ) -> str:
        """
        Экспортировать сессию в PDF

        Args:
            session: Сессия анализа
            file_path: Путь к файлу

        Returns:
            str: Путь к созданному файлу
        """
        if not self._has_dependencies:
            raise RuntimeError("PDF экспорт недоступен. Установите: pip install reportlab matplotlib seaborn")

        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
        from reportlab.lib import colors

        # Генерируем путь если не задан
        if not file_path:
            file_path = self._generate_filename(session.name, "pdf")

        # Создаём документ
        doc = SimpleDocTemplate(file_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        # Заголовок
        title = Paragraph(f"<b>Отчёт по анализу: {session.name}</b>", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 0.2 * inch))

        # Информация о сессии
        session_info = [
            ["ID сессии:", session.id],
            ["Статус:", session.status.value],
            ["Режим:", session.mode.value],
            ["Создано:", session.created_at.strftime("%Y-%m-%d %H:%M:%S")],
            ["Завершено:", session.completed_at.strftime("%Y-%m-%d %H:%M:%S") if session.completed_at else "N/A"],
        ]

        info_table = Table(session_info, colWidths=[2 * inch, 4 * inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(info_table)
        story.append(Spacer(1, 0.3 * inch))

        # Информация о текстах
        story.append(Paragraph("<b>Тексты</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1 * inch))

        texts_data = [["№", "ID", "Название", "Длина"]]
        for idx, text in enumerate(session.texts, 1):
            content_len = len(await text.get_content()) if hasattr(text, 'get_content') else 0
            texts_data.append([
                str(idx),
                text.id[:8] + "...",
                text.title[:40] + "..." if len(text.title) > 40 else text.title,
                f"{content_len:,} символов"
            ])

        texts_table = Table(texts_data, colWidths=[0.5 * inch, 1.5 * inch, 3 * inch, 1.5 * inch])
        texts_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))

        story.append(texts_table)
        story.append(Spacer(1, 0.3 * inch))

        # Матрица сравнения (если есть)
        if session.comparison_matrix:
            story.append(PageBreak())
            story.append(Paragraph("<b>Матрица сравнения</b>", styles['Heading2']))
            story.append(Spacer(1, 0.1 * inch))

            # Создаём heatmap
            heatmap_image = self._create_heatmap(session.comparison_matrix)
            if heatmap_image:
                story.append(heatmap_image)
                story.append(Spacer(1, 0.2 * inch))

            # Статистика
            avg_sim = session.comparison_matrix.get_average_similarity()
            stats_text = Paragraph(
                f"<b>Средняя схожесть:</b> {avg_sim:.4f}",
                styles['Normal']
            )
            story.append(stats_text)
            story.append(Spacer(1, 0.2 * inch))

            # Самые похожие пары
            most_similar = session.comparison_matrix.get_most_similar_pairs(top_k=5)
            if most_similar:
                story.append(Paragraph("<b>Топ-5 наиболее похожих пар</b>", styles['Heading3']))
                story.append(Spacer(1, 0.1 * inch))

                pairs_data = [["№", "Текст 1", "Текст 2", "Схожесть"]]
                for idx, pair in enumerate(most_similar, 1):
                    pairs_data.append([
                        str(idx),
                        pair["text1_id"][:15] + "...",
                        pair["text2_id"][:15] + "...",
                        f"{pair['similarity']:.4f}"
                    ])

                pairs_table = Table(pairs_data, colWidths=[0.5 * inch, 2 * inch, 2 * inch, 1.5 * inch])
                pairs_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))

                story.append(pairs_table)

        # Генерируем PDF
        doc.build(story)

        logger.info(f"📑 PDF экспорт завершён: {file_path}")

        return file_path

    async def export_comparison_matrix(
        self,
        matrix: ComparisonMatrix,
        file_path: Optional[str] = None
    ) -> str:
        """
        Экспортировать матрицу сравнения в PDF

        Args:
            matrix: Матрица сравнения
            file_path: Путь к файлу

        Returns:
            str: Путь к созданному файлу
        """
        if not self._has_dependencies:
            raise RuntimeError("PDF экспорт недоступен")

        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

        # Генерируем путь
        if not file_path:
            file_path = self._generate_filename("comparison_matrix", "pdf")

        # Создаём документ
        doc = SimpleDocTemplate(file_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()

        # Заголовок
        title = Paragraph("<b>Матрица сравнения</b>", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 0.2 * inch))

        # Heatmap
        heatmap_image = self._create_heatmap(matrix)
        if heatmap_image:
            story.append(heatmap_image)

        # Генерируем
        doc.build(story)

        logger.info(f"📑 PDF экспорт матрицы завершён: {file_path}")

        return file_path

    def _create_heatmap(self, matrix: ComparisonMatrix):
        """
        Создать heatmap из матрицы сравнения

        Args:
            matrix: Матрица сравнения

        Returns:
            Image: Изображение для ReportLab
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from reportlab.platypus import Image
            import numpy as np

            # Создаём фигуру
            fig, ax = plt.subplots(figsize=(8, 6))

            # Рисуем heatmap
            sns.heatmap(
                matrix.similarity_matrix,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                xticklabels=[tid[:8] + "..." for tid in matrix.text_ids],
                yticklabels=[tid[:8] + "..." for tid in matrix.text_ids],
                ax=ax,
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Similarity'}
            )

            ax.set_title('Матрица схожести текстов')
            plt.tight_layout()

            # Сохраняем в BytesIO
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)

            plt.close(fig)

            # Конвертируем в ReportLab Image
            from reportlab.lib.units import inch
            img = Image(img_buffer, width=6 * inch, height=4.5 * inch)

            return img

        except Exception as e:
            logger.error(f"Ошибка создания heatmap: {e}")
            return None
