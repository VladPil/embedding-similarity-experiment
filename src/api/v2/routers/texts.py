"""
Роутер для работы с текстами
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List
from loguru import logger

from ..schemas.text_schemas import (
    TextCreateRequest,
    FB2CreateRequest,
    TextResponse,
    TextListResponse,
    ChunkingRequest,
    ChunkingResponse
)
from src.infrastructure.database.connection import get_db
from src.infrastructure.database.repositories import TextRepository
from src.text_domain.services.chunking_service import ChunkingService
from src.text_domain.entities.chunking_strategy import ChunkingStrategy
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/texts", tags=["texts"])


@router.post("/", response_model=TextResponse, status_code=201)
async def create_text(
    request: TextCreateRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Создать новый текст

    - **title**: Название текста
    - **content**: Содержимое (для коротких текстов)
    - **file_path**: Путь к файлу (для длинных текстов)
    - **storage_type**: database или file
    """
    try:
        logger.info(f"Создание текста: {request.title}")

        # Валидация
        if request.storage_type == "database" and not request.content:
            raise HTTPException(
                status_code=400,
                detail="Для storage_type=database нужно указать content"
            )
        if request.storage_type == "file" and not request.file_path:
            raise HTTPException(
                status_code=400,
                detail="Для storage_type=file нужно указать file_path"
            )

        # Создаём через репозиторий
        repo = TextRepository(db)
        text = await repo.create(
            title=request.title,
            text_type="plain",
            storage_type=request.storage_type,
            content=request.content,
            file_path=request.file_path,
            metadata=request.metadata or {}
        )

        return TextResponse(
            id=text.id,
            title=text.title,
            storage_type=text.storage_type,
            content_length=text.length,
            metadata=text.text_metadata or {},
            created_at=text.created_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка создания текста: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fb2", response_model=TextResponse, status_code=201)
async def create_fb2_book(
    request: FB2CreateRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Создать FB2 книгу

    - **title**: Название
    - **file_path**: Путь к FB2 файлу
    - **parse_metadata**: Парсить метаданные из FB2
    """
    try:
        logger.info(f"Создание FB2 книги: {request.title}")

        if not request.file_path:
            raise HTTPException(
                status_code=400,
                detail="Нужно указать file_path для FB2"
            )

        # Создаём FB2 текст
        repo = TextRepository(db)

        # Если нужно парсить метаданные, делаем это здесь
        # Пока простая версия без парсинга
        metadata = {}
        if request.parse_metadata:
            # TODO: Парсинг FB2 метаданных (автор, жанр, год и т.д.)
            logger.info("Парсинг FB2 метаданных пока не реализован")

        text = await repo.create(
            title=request.title,
            text_type="fb2",
            storage_type="file",
            file_path=request.file_path,
            metadata=metadata
        )

        return TextResponse(
            id=text.id,
            title=text.title,
            storage_type=text.storage_type,
            content_length=text.length,
            metadata=text.text_metadata or {},
            created_at=text.created_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка создания FB2: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=TextListResponse)
async def list_texts(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """
    Получить список текстов

    - **offset**: Смещение для пагинации
    - **limit**: Количество элементов (макс 100)
    """
    try:
        repo = TextRepository(db)
        texts, total = await repo.list(offset=offset, limit=limit)

        text_responses = [
            TextResponse(
                id=text.id,
                title=text.title,
                storage_type=text.storage_type,
                content_length=text.length,
                metadata=text.text_metadata or {},
                created_at=text.created_at.isoformat()
            )
            for text in texts
        ]

        return TextListResponse(
            texts=text_responses,
            total=total,
            offset=offset,
            limit=limit
        )

    except Exception as e:
        logger.error(f"Ошибка получения списка текстов: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{text_id}", response_model=TextResponse)
async def get_text(
    text_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Получить текст по ID

    - **text_id**: ID текста
    """
    try:
        logger.info(f"Получение текста: {text_id}")

        repo = TextRepository(db)
        text = await repo.get_by_id(text_id)

        if not text:
            raise HTTPException(status_code=404, detail="Text not found")

        return TextResponse(
            id=text.id,
            title=text.title,
            storage_type=text.storage_type,
            content_length=text.length,
            metadata=text.text_metadata or {},
            created_at=text.created_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения текста: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{text_id}", status_code=204)
async def delete_text(
    text_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Удалить текст

    - **text_id**: ID текста
    """
    try:
        logger.info(f"Удаление текста: {text_id}")

        repo = TextRepository(db)
        deleted = await repo.delete(text_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Text not found")

        return None

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка удаления текста: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chunk", response_model=ChunkingResponse)
async def chunk_text(
    request: ChunkingRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Разбить текст на чанки

    - **text_id**: ID текста
    - **chunk_size**: Размер чанка (100-5000 символов)
    - **overlap**: Перекрытие (0-500 символов)
    - **boundary_type**: Тип границы (sentence/paragraph/none)
    """
    try:
        logger.info(f"Чанкинг текста {request.text_id}: size={request.chunk_size}")

        # Получаем текст
        repo = TextRepository(db)
        content = await repo.get_content(request.text_id)

        if not content:
            raise HTTPException(status_code=404, detail="Text not found")

        # Создаём стратегию чанкинга
        if request.boundary_type == "sentence":
            strategy = ChunkingStrategy.create_sentence_based(
                strategy_id=f"api-sentence-{request.text_id}",
                sentences_per_chunk=max(1, request.chunk_size // 100),
                overlap_sentences=max(0, request.overlap // 100)
            )
        elif request.boundary_type == "paragraph":
            strategy = ChunkingStrategy.create_paragraph_based(
                strategy_id=f"api-paragraph-{request.text_id}",
                paragraphs_per_chunk=max(1, request.chunk_size // 200),
                overlap_paragraphs=max(0, request.overlap // 200)
            )
        else:
            strategy = ChunkingStrategy.create_fixed_size(
                strategy_id=f"api-fixed-{request.text_id}",
                chunk_size=request.chunk_size,
                overlap=request.overlap
            )

        # Чанкуем
        chunking_service = ChunkingService()
        chunks = await chunking_service.chunk_text(content, strategy)

        # Форматируем ответ
        chunk_dicts = [
            {
                "index": chunk.chunk_index,
                "content": chunk.content,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos,
                "metadata": {}  # Добавляем пустой словарь метаданных
            }
            for chunk in chunks
        ]

        return ChunkingResponse(
            text_id=request.text_id,
            chunks=chunk_dicts,
            total_chunks=len(chunks),
            strategy_used=request.boundary_type
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка чанкинга: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{text_id}/content")
async def get_text_content(
    text_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Получить содержимое текста

    - **text_id**: ID текста

    Возвращает полный текст в виде строки
    """
    try:
        logger.info(f"Получение содержимого текста: {text_id}")

        repo = TextRepository(db)
        content = await repo.get_content(text_id)

        if content is None:
            raise HTTPException(status_code=404, detail="Text not found")

        return {"text_id": text_id, "content": content}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения содержимого: {e}")
        raise HTTPException(status_code=500, detail=str(e))
