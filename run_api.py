#!/usr/bin/env python3
"""
Скрипт для запуска API v2
"""
import uvicorn
from loguru import logger

if __name__ == "__main__":
    logger.info("Запуск Embedding Similarity Experiment API v2")

    uvicorn.run(
        "src.api.v2.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload при изменениях кода
        log_level="info",
        access_log=True,
    )
