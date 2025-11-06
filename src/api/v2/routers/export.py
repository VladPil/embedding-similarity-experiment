"""
Роутер для экспорта результатов анализа
"""
from fastapi import APIRouter, HTTPException, Query, Response, Depends
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
import json

from src.infrastructure.database.connection import get_db
from src.infrastructure.database.repositories import SessionRepository
from src.export.services.export_service import ExportService
from ..dependencies import get_export_service
from src.common.exceptions import ExportError, SessionNotFoundError

router = APIRouter(prefix="/export", tags=["export"])


@router.get("/sessions/{session_id}/json")
async def export_session_json(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    export_service: ExportService = Depends(get_export_service)
):
    """
    Экспортировать сессию в JSON

    - **session_id**: ID сессии

    Возвращает полные результаты анализа в формате JSON
    """
    try:
        logger.info(f"Экспорт сессии {session_id} в JSON")

        # Получаем сессию из БД
        session_repo = SessionRepository(db)
        session_entity = await session_repo.get_by_id(session_id)

        if not session_entity:
            raise HTTPException(status_code=404, detail=f"Сессия {session_id} не найдена")

        # Экспортируем через сервис
        file_path = await export_service.export_session(session_entity, format="json")

        # Возвращаем файл
        return FileResponse(
            path=file_path,
            media_type="application/json",
            filename=f"session_{session_id}.json"
        )

    except HTTPException:
        raise
    except ExportError as e:
        logger.error(f"Ошибка экспорта в JSON: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка экспорта в JSON: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/csv")
async def export_session_csv(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    export_service: ExportService = Depends(get_export_service)
):
    """
    Экспортировать сессию в CSV

    - **session_id**: ID сессии

    Возвращает результаты анализа в формате CSV
    """
    try:
        logger.info(f"Экспорт сессии {session_id} в CSV")

        # Получаем сессию из БД
        session_repo = SessionRepository(db)
        session_entity = await session_repo.get_by_id(session_id)

        if not session_entity:
            raise HTTPException(status_code=404, detail=f"Сессия {session_id} не найдена")

        # Экспортируем через сервис
        file_path = await export_service.export_session(session_entity, format="csv")

        # Возвращаем файл
        return FileResponse(
            path=file_path,
            media_type="text/csv",
            filename=f"session_{session_id}.csv"
        )

    except HTTPException:
        raise
    except ExportError as e:
        logger.error(f"Ошибка экспорта в CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка экспорта в CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/pdf")
async def export_session_pdf(
    session_id: str,
    include_charts: bool = Query(True, description="Включить графики"),
    db: AsyncSession = Depends(get_db),
    export_service: ExportService = Depends(get_export_service)
):
    """
    Экспортировать сессию в PDF

    - **session_id**: ID сессии
    - **include_charts**: Включить графики и визуализации

    Возвращает красивый PDF отчёт с результатами
    """
    try:
        logger.info(f"Экспорт сессии {session_id} в PDF, charts={include_charts}")

        # Получаем сессию из БД
        session_repo = SessionRepository(db)
        session_entity = await session_repo.get_by_id(session_id)

        if not session_entity:
            raise HTTPException(status_code=404, detail=f"Сессия {session_id} не найдена")

        # Экспортируем через сервис
        file_path = await export_service.export_session(session_entity, format="pdf")

        # Возвращаем файл
        return FileResponse(
            path=file_path,
            media_type="application/pdf",
            filename=f"session_{session_id}.pdf"
        )

    except HTTPException:
        raise
    except ExportError as e:
        logger.error(f"Ошибка экспорта в PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка экспорта в PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/markdown")
async def export_session_markdown(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    export_service: ExportService = Depends(get_export_service)
):
    """
    Экспортировать сессию в Markdown

    - **session_id**: ID сессии

    Возвращает результаты в формате Markdown для документации
    """
    try:
        logger.info(f"Экспорт сессии {session_id} в Markdown")

        # Получаем сессию из БД
        session_repo = SessionRepository(db)
        session_entity = await session_repo.get_by_id(session_id)

        if not session_entity:
            raise HTTPException(status_code=404, detail=f"Сессия {session_id} не найдена")

        # Экспортируем через сервис
        file_path = await export_service.export_session(session_entity, format="markdown")

        # Возвращаем файл
        return FileResponse(
            path=file_path,
            media_type="text/markdown",
            filename=f"session_{session_id}.md"
        )

    except HTTPException:
        raise
    except ExportError as e:
        logger.error(f"Ошибка экспорта в Markdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка экспорта в Markdown: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparisons/{comparison_id}/json")
async def export_comparison_json(
    comparison_id: str,
    db: AsyncSession = Depends(get_db),
    export_service: ExportService = Depends(get_export_service)
):
    """
    Экспортировать результаты сравнения в JSON

    - **comparison_id**: ID задачи сравнения (session_id)

    Возвращает результаты сравнения текстов в формате JSON
    """
    try:
        logger.info(f"Экспорт сравнения {comparison_id} в JSON")

        # Получаем сессию из БД (comparison_id это session_id)
        session_repo = SessionRepository(db)
        session_entity = await session_repo.get_by_id(comparison_id)

        if not session_entity:
            raise HTTPException(status_code=404, detail=f"Сравнение {comparison_id} не найдено")

        if not session_entity.comparison_matrix:
            raise HTTPException(
                status_code=404,
                detail=f"Матрица сравнения для сессии {comparison_id} не найдена"
            )

        # Экспортируем матрицу
        file_path = await export_service.export_comparison_matrix(
            session_entity.comparison_matrix,
            format="json"
        )

        return FileResponse(
            path=file_path,
            media_type="application/json",
            filename=f"comparison_{comparison_id}.json"
        )

    except HTTPException:
        raise
    except ExportError as e:
        logger.error(f"Ошибка экспорта сравнения: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка экспорта сравнения: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def list_export_templates(
    export_service: ExportService = Depends(get_export_service)
):
    """
    Получить список доступных шаблонов экспорта

    Возвращает доступные шаблоны для PDF и Markdown экспорта
    """
    try:
        templates = export_service.get_available_templates()
        return {"templates": templates}

    except Exception as e:
        logger.error(f"Ошибка получения шаблонов: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def batch_export(
    session_ids: List[str],
    format: str = Query(..., description="Формат: json/csv/pdf/markdown"),
    db: AsyncSession = Depends(get_db),
    export_service: ExportService = Depends(get_export_service)
):
    """
    Массовый экспорт нескольких сессий

    - **session_ids**: Список ID сессий
    - **format**: Формат экспорта

    Создаёт архив с экспортированными сессиями
    """
    try:
        logger.info(f"Массовый экспорт {len(session_ids)} сессий в формат {format}")

        if format not in ["json", "csv", "pdf", "markdown"]:
            raise HTTPException(
                status_code=400,
                detail="Поддерживаемые форматы: json, csv, pdf, markdown"
            )

        # Получаем все сессии из БД
        session_repo = SessionRepository(db)
        sessions = []

        for session_id in session_ids:
            session_entity = await session_repo.get_by_id(session_id)
            if session_entity:
                sessions.append(session_entity)
            else:
                logger.warning(f"Сессия {session_id} не найдена, пропускаем")

        if not sessions:
            raise HTTPException(
                status_code=404,
                detail="Ни одна из указанных сессий не найдена"
            )

        # Создаём архив через сервис
        archive_path = await export_service.batch_export(
            sessions=sessions,
            format=format,
            create_archive=True
        )

        # Возвращаем архив
        return FileResponse(
            path=archive_path,
            media_type="application/zip",
            filename=f"batch_export_{format}.zip"
        )

    except HTTPException:
        raise
    except ExportError as e:
        logger.error(f"Ошибка массового экспорта: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка массового экспорта: {e}")
        raise HTTPException(status_code=500, detail=str(e))
