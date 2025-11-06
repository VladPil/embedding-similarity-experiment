"""
FastAPI application factory.
Creates and configures the FastAPI application instance.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from loguru import logger

from server.config import settings, BASE_DIR
from server.system.cache import init_redis, close_redis


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Development mode: {settings.development_mode}")

    # Initialize Redis
    try:
        await init_redis()
        logger.info("✓ Redis initialized")
    except Exception as e:
        logger.error(f"✗ Redis initialization failed: {e}")
        logger.warning("Continuing without Redis caching")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    await close_redis()
    logger.info("✓ Application shutdown complete")


def create_app(mount_static: bool = True) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        mount_static: Whether to mount static files directory

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        debug=settings.debug,
        lifespan=lifespan,
        default_response_class=JSONResponse,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # Mount static files
    if mount_static:
        static_path = BASE_DIR / "frontend" / "dist"
        if static_path.exists():
            app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")
            logger.info(f"✓ Static files mounted: {static_path}")

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "app": settings.app_name,
            "version": settings.version,
        }

    return app
