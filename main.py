"""
Точка входа для запуска FastAPI приложения
"""
import uvicorn
from loguru import logger


def main():
    """
    Запуск FastAPI сервера с uvicorn
    """
    logger.info("Запуск Embedding Similarity Experiment API...")

    uvicorn.run(
        "src.api.v2.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload при изменениях кода (для разработки)
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
