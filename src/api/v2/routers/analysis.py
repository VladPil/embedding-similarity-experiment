"""
Роутер для анализа текстов
"""
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from typing import List
from loguru import logger
import asyncio
import json

from ..schemas.analysis_schemas import (
    AnalysisSessionCreateRequest,
    AnalysisSessionResponse,
    AnalysisSessionDetailResponse,
    AnalysisSessionListResponse,
    SessionRunRequest
)
from src.infrastructure.database.connection import get_db
from src.infrastructure.database.repositories import SessionRepository, TextRepository
from src.analysis_domain.analysis_service import AnalysisService
from sqlalchemy.ext.asyncio import AsyncSession
from ..dependencies import (
    get_analysis_service,
    get_progress_broadcaster,
    get_llm_service,
    get_embedding_service
)
from src.infrastructure.queue.progress_broadcaster import ProgressBroadcaster

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/sessions", response_model=AnalysisSessionResponse, status_code=201)
async def create_session(
    request: AnalysisSessionCreateRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Создать новую сессию анализа

    - **name**: Название сессии
    - **text_ids**: ID текстов для анализа (1-5 штук)
    - **analyzer_types**: Типы анализаторов (GenreAnalyzer, StyleAnalyzer, и т.д.)
    - **mode**: full_text или chunked
    - **comparator_type**: Опционально, для сравнения текстов
    """
    try:
        logger.info(f"Создание сессии: {request.name}, тексты: {len(request.text_ids)}")

        # Валидация
        if len(request.text_ids) > 5:
            raise HTTPException(
                status_code=400,
                detail="Максимум 5 текстов в одной сессии"
            )

        if len(request.text_ids) == 0:
            raise HTTPException(
                status_code=400,
                detail="Нужен хотя бы один текст"
            )

        if len(request.analyzer_types) == 0:
            raise HTTPException(
                status_code=400,
                detail="Нужен хотя бы один анализатор"
            )

        # Проверяем что тексты существуют
        text_repo = TextRepository(db)
        for text_id in request.text_ids:
            text = await text_repo.get_by_id(text_id)
            if not text:
                raise HTTPException(
                    status_code=404,
                    detail=f"Текст {text_id} не найден"
                )

        # Создаём сессию через репозиторий
        session_repo = SessionRepository(db)
        session = await session_repo.create(
            name=request.name,
            text_ids=request.text_ids,
            analyzer_types=request.analyzer_types,
            mode=request.mode,
            chunked_comparison_strategy=request.comparator_type
        )

        return AnalysisSessionResponse(
            id=session.id,
            name=session.name,
            status=session.status,
            mode=session.mode,
            text_ids=request.text_ids,
            analyzer_types=request.analyzer_types,
            created_at=session.created_at,
            started_at=None,
            completed_at=None,
            progress=0.0,
            error_message=None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка создания сессии: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=AnalysisSessionListResponse)
async def list_sessions(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    status: str = Query(None, description="Фильтр по статусу"),
    db: AsyncSession = Depends(get_db)
):
    """
    Получить список сессий

    - **offset**: Смещение для пагинации
    - **limit**: Количество элементов
    - **status**: Фильтр по статусу (draft/queued/running/completed/failed)
    """
    try:
        session_repo = SessionRepository(db)
        sessions, total = await session_repo.list(
            offset=offset,
            limit=limit,
            status=status
        )

        session_responses = []
        for session in sessions:
            text_ids = await session_repo.get_text_ids(session.id)
            analyzer_names = await session_repo.get_analyzer_names(session.id)

            session_responses.append(
                AnalysisSessionResponse(
                    id=session.id,
                    name=session.name,
                    status=session.status,
                    mode=session.mode,
                    text_ids=text_ids,
                    analyzer_types=analyzer_names,
                    created_at=session.created_at,
                    started_at=session.started_at,
                    completed_at=session.completed_at,
                    progress=float(session.progress) / 100.0,
                    error_message=session.error
                )
            )

        return AnalysisSessionListResponse(
            sessions=session_responses,
            total=total,
            offset=offset,
            limit=limit
        )

    except Exception as e:
        logger.error(f"Ошибка получения списка сессий: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=AnalysisSessionDetailResponse)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Получить детальную информацию о сессии

    - **session_id**: ID сессии

    Возвращает сессию со всеми результатами анализа
    """
    try:
        logger.info(f"Получение сессии: {session_id}")

        session_repo = SessionRepository(db)
        session = await session_repo.get_by_id(session_id, load_relations=True)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Получаем тексты и анализаторы
        text_ids = await session_repo.get_text_ids(session_id)
        analyzer_names = await session_repo.get_analyzer_names(session_id)

        # Получаем результаты
        results = await session_repo.get_results(session_id)

        # Группируем результаты по text_id
        results_by_text = {}
        for r in results:
            text_id = r.text_id
            if text_id not in results_by_text:
                results_by_text[text_id] = []

            results_by_text[text_id].append({
                "text_id": r.text_id,
                "analyzer_type": r.analyzer_name,
                "mode": "full_text",  # TODO: получать из данных
                "data": r.result_data,
                "interpretation": r.interpretation,
                "created_at": r.created_at
            })

        # Вычисляем время выполнения
        execution_time_seconds = None
        if session.started_at and session.completed_at:
            delta = session.completed_at - session.started_at
            execution_time_seconds = delta.total_seconds()

        return AnalysisSessionDetailResponse(
            id=session.id,
            name=session.name,
            status=session.status,
            mode=session.mode,
            text_ids=text_ids,
            analyzer_types=analyzer_names,
            created_at=session.created_at,
            started_at=session.started_at,
            completed_at=session.completed_at,
            progress=float(session.progress) / 100.0,
            error_message=session.error,
            results=results_by_text,
            comparison_matrix=None,  # TODO: Реализовать сравнения
            execution_time_seconds=execution_time_seconds
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения сессии: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_analysis_task(
    session_id: str,
    analysis_service: AnalysisService
):
    """Фоновая задача для запуска анализа"""
    try:
        # Запускаем анализ
        await analysis_service.run_session(session_id)

    except Exception as e:
        logger.error(f"Ошибка в фоновой задаче анализа: {e}")


@router.post("/sessions/{session_id}/run")
async def run_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    async_mode: bool = Query(True),
    db: AsyncSession = Depends(get_db),
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Запустить сессию анализа

    - **session_id**: ID сессии
    - **async_mode**: Асинхронный режим (через очередь задач)

    Если async_mode=True, задача добавляется в очередь и возвращается сразу.
    Если async_mode=False, выполняется синхронно (может быть долго).
    """
    try:
        logger.info(f"Запуск сессии {session_id}, async={async_mode}")

        session_repo = SessionRepository(db)
        session = await session_repo.get_by_id(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if session.status not in ["draft", "failed"]:
            raise HTTPException(
                status_code=400,
                detail=f"Нельзя запустить сессию в статусе {session.status}"
            )

        if async_mode:
            # Помечаем как в очереди
            await session_repo.update_status(session_id, "queued")

            # Добавляем фоновую задачу
            background_tasks.add_task(run_analysis_task, session_id, analysis_service)

            return {
                "status": "queued",
                "message": "Сессия добавлена в очередь",
                "session_id": session_id
            }
        else:
            # Синхронное выполнение
            await analysis_service.run_session(session_id)

            return {
                "status": "completed",
                "message": "Сессия выполнена",
                "session_id": session_id
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка запуска сессии: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Удалить сессию

    - **session_id**: ID сессии
    """
    try:
        logger.info(f"Удаление сессии: {session_id}")

        session_repo = SessionRepository(db)
        deleted = await session_repo.delete(session_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")

        return None

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка удаления сессии: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/cancel")
async def cancel_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    analysis_service: AnalysisService = Depends(get_analysis_service)
):
    """
    Отменить выполнение сессии

    - **session_id**: ID сессии

    Работает только для сессий в статусе queued или running
    """
    try:
        logger.info(f"Отмена сессии: {session_id}")

        session_repo = SessionRepository(db)
        session = await session_repo.get_by_id(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if session.status not in ["queued", "running"]:
            raise HTTPException(
                status_code=400,
                detail=f"Нельзя отменить сессию в статусе {session.status}"
            )

        # Отменяем через сервис
        await analysis_service.cancel_session(session_id)

        return {
            "status": "cancelled",
            "message": "Сессия отменена",
            "session_id": session_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка отмены сессии: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/sessions/{session_id}/progress")
async def session_progress_ws(
    websocket: WebSocket,
    session_id: str,
    progress_broadcaster: ProgressBroadcaster = Depends(get_progress_broadcaster)
):
    """
    WebSocket для отслеживания прогресса сессии в реальном времени

    - **session_id**: ID сессии

    Отправляет обновления прогресса в реальном времени
    """
    await websocket.accept()

    try:
        logger.info(f"WebSocket подключён для сессии: {session_id}")

        # Подписываемся на обновления
        await progress_broadcaster.subscribe(session_id, websocket)

        # Ждём пока соединение активно
        while True:
            try:
                # Ждём сообщения от клиента (для проверки соединения)
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Таймаут - отправляем ping
                await websocket.send_json({"type": "ping"})
            except Exception:
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket отключён для сессии: {session_id}")
    except Exception as e:
        logger.error(f"Ошибка WebSocket: {e}")
    finally:
        # Отписываемся
        await progress_broadcaster.unsubscribe(session_id, websocket)
        try:
            await websocket.close()
        except:
            pass


@router.get("/sessions/{session_id}/export/{format}")
async def export_session(session_id: str, format: str):
    """
    Экспортировать результаты сессии

    - **session_id**: ID сессии
    - **format**: Формат экспорта (json/csv/pdf)

    Возвращает файл с результатами
    """
    try:
        logger.info(f"Экспорт сессии {session_id} в формат {format}")

        if format not in ["json", "csv", "pdf"]:
            raise HTTPException(
                status_code=400,
                detail="Поддерживаемые форматы: json, csv, pdf"
            )

        # Перенаправляем на export роутер
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"/export/sessions/{session_id}/{format}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка экспорта: {e}")
        raise HTTPException(status_code=500, detail=str(e))
