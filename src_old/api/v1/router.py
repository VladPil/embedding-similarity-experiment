"""
Main v1 API router.
Aggregates all v1 endpoints.
"""

from fastapi import APIRouter

from .texts import router as texts_router
from .embeddings import router as embeddings_router
from .analysis_refactored import router as analysis_router
from .tasks import router as tasks_router
from .multi_analysis import router as multi_analysis_router
from .cache import router as cache_router
from .book_analysis import router as book_analysis_router
from .websockets import router as websockets_router

# Create v1 router
api_v1_router = APIRouter()

# Include all sub-routers
api_v1_router.include_router(texts_router)
api_v1_router.include_router(embeddings_router)
api_v1_router.include_router(analysis_router)
api_v1_router.include_router(tasks_router)
api_v1_router.include_router(multi_analysis_router)
api_v1_router.include_router(cache_router, prefix="/cache", tags=["cache"])
api_v1_router.include_router(book_analysis_router, prefix="/book", tags=["book-analysis"])
api_v1_router.include_router(websockets_router, tags=["websockets"])
