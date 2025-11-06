"""
Text management API endpoints.
Uses the new service layer architecture.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from loguru import logger

from server.db.database import database
from server.services.text_service import TextService
from server.schemas.models import (
    TextUploadRequest,
    TextInfo,
    TextListResponse,
    ErrorResponse
)
from server.core.texts.fb2_parser import FB2Parser

router = APIRouter(prefix="/texts", tags=["texts"])


# Dependency injection
def get_db() -> Session:
    """Get database session."""
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_text_service(db: Session = Depends(get_db)) -> TextService:
    """Get text service instance."""
    return TextService(db)


@router.get("", response_model=TextListResponse)
async def list_texts(
    service: TextService = Depends(get_text_service),
    limit: int = 100,
    offset: int = 0
):
    """
    List all stored texts.

    Returns:
        List of text metadata
    """
    try:
        texts = await service.list_texts(limit=limit, offset=offset)

        # Convert to TextInfo objects
        text_infos = []
        for text in texts:
            text_infos.append(TextInfo(
                id=text['id'],
                title=text['title'],
                lines=text['lines'],
                length=text['length']
            ))

        return TextListResponse(texts=text_infos)

    except Exception as e:
        logger.error(f"Failed to list texts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=TextInfo)
async def upload_text(
    request: TextUploadRequest,
    service: TextService = Depends(get_text_service)
):
    """
    Upload text via JSON.

    Args:
        request: Text upload request with title and content

    Returns:
        Created text info
    """
    try:
        # Create text using service
        text = await service.create_text(
            title=request.title,
            content=request.text
        )

        return TextInfo(
            id=text.id,
            title=text.title,
            lines=text.lines,
            length=text.length
        )

    except Exception as e:
        logger.error(f"Failed to upload text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-fb2", response_model=TextInfo)
async def upload_fb2(
    file: UploadFile = File(...),
    title: Optional[str] = None,
    service: TextService = Depends(get_text_service)
):
    """
    Upload FB2 file.

    Args:
        file: FB2 file upload
        title: Optional title override

    Returns:
        Created text info
    """
    try:
        # Read file content
        content = await file.read()

        # Parse FB2
        parser = FB2Parser()
        text_content = parser.parse_bytes(content)

        # Extract title
        if not title:
            title = file.filename.replace('.fb2', '') if file.filename else "Untitled"

        # Create text using service
        text = await service.create_text(
            title=title,
            content=text_content
        )

        logger.info(f"Uploaded FB2: {title} ({text.length} chars)")

        return TextInfo(
            id=text.id,
            title=text.title,
            lines=text.lines,
            length=text.length
        )

    except Exception as e:
        logger.error(f"Failed to upload FB2: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse FB2: {str(e)}")


@router.get("/{text_id}", response_model=dict)
async def get_text(
    text_id: str,
    service: TextService = Depends(get_text_service)
):
    """
    Get text content by ID.

    Args:
        text_id: Text identifier

    Returns:
        Text content and metadata
    """
    try:
        # Get text from service
        text = service.get_text(text_id)

        if not text:
            raise HTTPException(status_code=404, detail="Text not found")

        # Get content
        content = await service.get_text_content(text_id)

        return {
            "success": True,
            "text_id": text.id,
            "title": text.title,
            "text": content,
            "lines": text.lines,
            "length": text.length
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get text {text_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{text_id}", response_model=TextInfo)
async def update_text(
    text_id: str,
    request: TextUploadRequest,
    service: TextService = Depends(get_text_service)
):
    """
    Update text by ID.

    Args:
        text_id: Text identifier
        request: Text update request with title and content

    Returns:
        Updated text info
    """
    try:
        # Update text using service
        text = await service.update_text(
            text_id=text_id,
            title=request.title,
            content=request.text
        )

        if not text:
            raise HTTPException(status_code=404, detail="Text not found")

        return TextInfo(
            id=text.id,
            title=text.title,
            lines=text.lines,
            length=text.length
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update text {text_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{text_id}")
async def delete_text(
    text_id: str,
    service: TextService = Depends(get_text_service)
):
    """
    Delete text by ID.
    This will also delete all related embeddings and cache entries.

    Args:
        text_id: Text identifier

    Returns:
        Success message
    """
    try:
        deleted = await service.delete_text(text_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Text not found")

        return {"success": True, "message": f"Text {text_id} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete text {text_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))