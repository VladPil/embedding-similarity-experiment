"""
Репозиторий для работы с сессиями анализа
"""
from typing import Optional, List
from datetime import datetime
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from loguru import logger

from ..models import (
    AnalysisSessionModel,
    SessionTextModel,
    SessionAnalyzerModel,
    AnalysisResultModel,
    ComparisonMatrixModel
)
from src.common.exceptions import DatabaseOperationError
from src.common.utils import generate_id, now_utc


class SessionRepository:
    """Репозиторий для работы с сессиями анализа"""

    def __init__(self, session: AsyncSession):
        """
        Args:
            session: Сессия SQLAlchemy
        """
        self.session = session

    async def create(
        self,
        name: str,
        text_ids: List[str],
        analyzer_types: List[str],
        mode: str = "full_text",
        **kwargs
    ) -> AnalysisSessionModel:
        """
        Создать новую сессию анализа

        Args:
            name: Название сессии
            text_ids: ID текстов для анализа
            analyzer_types: Типы анализаторов
            mode: Режим анализа (full_text/chunked)
            **kwargs: Дополнительные параметры

        Returns:
            Созданная сессия
        """
        try:
            session_obj = AnalysisSessionModel(
                id=generate_id("session"),
                name=name,
                status="draft",
                mode=mode,
                progress=0,
                **kwargs
            )

            # Добавляем связи с текстами
            for position, text_id in enumerate(text_ids):
                session_text = SessionTextModel(
                    session_id=session_obj.id,
                    text_id=text_id,
                    position=position
                )
                self.session.add(session_text)

            # Добавляем связи с анализаторами
            for position, analyzer_name in enumerate(analyzer_types):
                session_analyzer = SessionAnalyzerModel(
                    session_id=session_obj.id,
                    analyzer_name=analyzer_name,
                    position=position
                )
                self.session.add(session_analyzer)

            self.session.add(session_obj)
            await self.session.commit()
            await self.session.refresh(session_obj)

            logger.info(f"Создана сессия: {session_obj.id} - {name}")
            return session_obj

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка создания сессии: {e}")
            raise DatabaseOperationError(
                message=f"Failed to create session: {e}",
                details={"name": name}
            )

    async def get_by_id(
        self,
        session_id: str,
        load_relations: bool = False
    ) -> Optional[AnalysisSessionModel]:
        """
        Получить сессию по ID

        Args:
            session_id: ID сессии
            load_relations: Загрузить связанные объекты

        Returns:
            Сессия или None
        """
        try:
            stmt = select(AnalysisSessionModel).where(
                AnalysisSessionModel.id == session_id
            )

            if load_relations:
                stmt = stmt.options(
                    selectinload(AnalysisSessionModel.texts),
                    selectinload(AnalysisSessionModel.analyzers),
                    selectinload(AnalysisSessionModel.analysis_results),
                    selectinload(AnalysisSessionModel.comparison_matrix)
                )

            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()

        except Exception as e:
            logger.error(f"Ошибка получения сессии {session_id}: {e}")
            return None

    async def list(
        self,
        offset: int = 0,
        limit: int = 50,
        status: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> tuple[List[AnalysisSessionModel], int]:
        """
        Получить список сессий с пагинацией

        Args:
            offset: Смещение
            limit: Количество
            status: Фильтр по статусу
            user_id: Фильтр по пользователю

        Returns:
            (Список сессий, общее количество)
        """
        try:
            # Фильтры
            where_clause = []
            if status:
                where_clause.append(AnalysisSessionModel.status == status)
            if user_id:
                where_clause.append(AnalysisSessionModel.user_id == user_id)

            # Общее количество
            count_stmt = select(func.count(AnalysisSessionModel.id))
            if where_clause:
                count_stmt = count_stmt.where(and_(*where_clause))

            count_result = await self.session.execute(count_stmt)
            total = count_result.scalar()

            # Сессии с пагинацией
            stmt = select(AnalysisSessionModel)
            if where_clause:
                stmt = stmt.where(and_(*where_clause))

            stmt = stmt.order_by(AnalysisSessionModel.created_at.desc())
            stmt = stmt.offset(offset).limit(limit)

            result = await self.session.execute(stmt)
            sessions = result.scalars().all()

            return list(sessions), total

        except Exception as e:
            logger.error(f"Ошибка получения списка сессий: {e}")
            return [], 0

    async def update_status(
        self,
        session_id: str,
        status: str,
        progress: Optional[int] = None,
        progress_message: Optional[str] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        Обновить статус сессии

        Args:
            session_id: ID сессии
            status: Новый статус
            progress: Прогресс (0-100)
            progress_message: Сообщение о прогрессе
            error: Сообщение об ошибке

        Returns:
            True если обновлён
        """
        try:
            session_obj = await self.get_by_id(session_id)
            if not session_obj:
                return False

            session_obj.status = status
            if progress is not None:
                session_obj.progress = progress
            if progress_message is not None:
                session_obj.progress_message = progress_message
            if error is not None:
                session_obj.error = error

            # Обновляем временные метки
            if status == "queued" and not session_obj.queued_at:
                session_obj.queued_at = now_utc()
            elif status == "running" and not session_obj.started_at:
                session_obj.started_at = now_utc()
            elif status in ["completed", "failed", "cancelled"]:
                session_obj.completed_at = now_utc()

            await self.session.commit()
            logger.debug(f"Обновлён статус сессии {session_id}: {status}")
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка обновления статуса сессии: {e}")
            return False

    async def save_result(
        self,
        session_id: str,
        text_id: str,
        analyzer_name: str,
        result_data: dict,
        interpretation: Optional[str] = None,
        execution_time_ms: Optional[float] = None
    ) -> bool:
        """
        Сохранить результат анализа

        Args:
            session_id: ID сессии
            text_id: ID текста
            analyzer_name: Название анализатора
            result_data: Данные результата
            interpretation: Интерпретация
            execution_time_ms: Время выполнения

        Returns:
            True если сохранён
        """
        try:
            result = AnalysisResultModel(
                id=generate_id("result"),
                session_id=session_id,
                text_id=text_id,
                analyzer_name=analyzer_name,
                result_data=result_data,
                interpretation=interpretation,
                execution_time_ms=execution_time_ms
            )

            self.session.add(result)
            await self.session.commit()

            logger.debug(f"Сохранён результат анализа: {analyzer_name} для текста {text_id}")
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка сохранения результата: {e}")
            return False

    async def save_comparison_matrix(
        self,
        session_id: str,
        matrix_data: dict,
        aggregated_scores: Optional[dict] = None
    ) -> bool:
        """
        Сохранить матрицу сравнений

        Args:
            session_id: ID сессии
            matrix_data: Данные матрицы
            aggregated_scores: Агрегированные скоры

        Returns:
            True если сохранена
        """
        try:
            matrix = ComparisonMatrixModel(
                id=generate_id("matrix"),
                session_id=session_id,
                matrix_data=matrix_data,
                aggregated_scores=aggregated_scores
            )

            self.session.add(matrix)
            await self.session.commit()

            logger.debug(f"Сохранена матрица сравнений для сессии {session_id}")
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка сохранения матрицы: {e}")
            return False

    async def delete(self, session_id: str) -> bool:
        """
        Удалить сессию

        Args:
            session_id: ID сессии

        Returns:
            True если удалена
        """
        try:
            session_obj = await self.get_by_id(session_id)
            if not session_obj:
                return False

            await self.session.delete(session_obj)
            await self.session.commit()

            logger.info(f"Удалена сессия: {session_id}")
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Ошибка удаления сессии {session_id}: {e}")
            return False

    async def get_text_ids(self, session_id: str) -> List[str]:
        """
        Получить ID текстов в сессии

        Args:
            session_id: ID сессии

        Returns:
            Список ID текстов в правильном порядке
        """
        try:
            stmt = select(SessionTextModel).where(
                SessionTextModel.session_id == session_id
            ).order_by(SessionTextModel.position)

            result = await self.session.execute(stmt)
            session_texts = result.scalars().all()

            return [st.text_id for st in session_texts]

        except Exception as e:
            logger.error(f"Ошибка получения текстов сессии: {e}")
            return []

    async def get_analyzer_names(self, session_id: str) -> List[str]:
        """
        Получить названия анализаторов в сессии

        Args:
            session_id: ID сессии

        Returns:
            Список названий анализаторов в правильном порядке
        """
        try:
            stmt = select(SessionAnalyzerModel).where(
                SessionAnalyzerModel.session_id == session_id
            ).order_by(SessionAnalyzerModel.position)

            result = await self.session.execute(stmt)
            session_analyzers = result.scalars().all()

            return [sa.analyzer_name for sa in session_analyzers]

        except Exception as e:
            logger.error(f"Ошибка получения анализаторов сессии: {e}")
            return []

    async def get_results(
        self,
        session_id: str,
        text_id: Optional[str] = None
    ) -> List[AnalysisResultModel]:
        """
        Получить результаты анализа сессии

        Args:
            session_id: ID сессии
            text_id: Фильтр по тексту (опционально)

        Returns:
            Список результатов
        """
        try:
            stmt = select(AnalysisResultModel).where(
                AnalysisResultModel.session_id == session_id
            )

            if text_id:
                stmt = stmt.where(AnalysisResultModel.text_id == text_id)

            stmt = stmt.order_by(AnalysisResultModel.created_at)

            result = await self.session.execute(stmt)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Ошибка получения результатов: {e}")
            return []
