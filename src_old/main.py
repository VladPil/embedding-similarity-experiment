"""
Main application entry point.
Initializes FastAPI app with all routers and middleware.
"""

import uvloop
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from fastapi import Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from loguru import logger

from server.config import settings
from server.system import create_app
from server.api.v1.router import api_v1_router
from server.exceptions import AppException
from server.system.queue.app import start_broker, stop_broker, get_broker
from server.system.queue.consumers import router as consumer_router
from server.system.queue.broadcaster import start_broadcaster, stop_broadcaster


# Configure loguru
if settings.loguru_enabled:
    logger.remove()  # Remove default handler

    # Console handler
    logger.add(
        sys.stderr,
        level=settings.loguru_console_level,
        format=settings.loguru_console_format,
        colorize=settings.loguru_colorize,
    )

    # File handler
    if settings.loguru_write_log_file:
        logger.add(
            settings.loguru_path,
            level=settings.loguru_level,
            rotation=settings.loguru_rotation,
            retention=settings.loguru_retention,
            compression=settings.loguru_compression,
            serialize=settings.loguru_serialize,
        )


# Create FastAPI app
app = create_app(mount_static=False)  # We'll mount static files manually after API routes


# FastStream lifecycle events
@app.on_event("startup")
async def startup_event():
    """Start FastStream broker and broadcaster on application startup."""
    logger.info("Starting FastStream broker...")
    await start_broker()

    # Include consumer router
    broker = get_broker()
    broker.include_router(consumer_router)

    logger.info("FastStream broker started successfully")

    # Start broadcaster
    logger.info("Starting task progress broadcaster...")
    await start_broadcaster()
    logger.info("Broadcaster started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop FastStream broker and broadcaster on application shutdown."""
    logger.info("Stopping broadcaster...")
    await stop_broadcaster()
    logger.info("Broadcaster stopped")

    logger.info("Stopping FastStream broker...")
    await stop_broker()
    logger.info("FastStream broker stopped")


# Exception handlers
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """Handle application exceptions."""
    logger.error(f"Application error: {exc.message} | Details: {exc.details}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": exc.message,
            "details": exc.details
        }
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Validation error",
            "details": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions."""
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "details": str(exc) if settings.debug else None
        }
    )


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# Include API routers
app.include_router(
    router=api_v1_router,
    prefix=settings.api_prefix + settings.api_v1_prefix,
    tags=["api"],
)


# Mount frontend AFTER API routes
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from server.config import BASE_DIR

frontend_path = BASE_DIR / "frontend" / "dist"
if frontend_path.exists():
    # Mount static files (JS, CSS, etc.)
    app.mount("/assets", StaticFiles(directory=str(frontend_path / "assets")), name="assets")

    # Catch-all route for SPA - serve index.html for all non-API routes
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str, request: Request):
        """Serve SPA index.html for all non-API routes."""
        # Don't intercept API routes - return 404 for GET requests to API
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")

        # If file exists in dist, serve it
        file_path = frontend_path / full_path
        if file_path.is_file():
            return FileResponse(file_path)

        # Otherwise serve index.html (for SPA routing)
        return FileResponse(frontend_path / "index.html")

    logger.info(f"✓ Frontend mounted from: {frontend_path}")
else:
    logger.warning(f"Frontend directory not found: {frontend_path}")


def get_app():
    """Get FastAPI app instance (for gunicorn/uvicorn workers)."""
    return app


def main():
    """Run application with uvicorn."""
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    logger.info(f"Starting server on {settings.run_host}:{settings.run_port}")

    uvicorn.run(
        "server.main:app",
        host=settings.run_host,
        port=settings.run_port,
        reload=settings.development_mode,
        log_level="info",
    )


if __name__ == "__main__":
    main()
