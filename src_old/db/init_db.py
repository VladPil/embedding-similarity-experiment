#!/usr/bin/env python3
"""
Initialize database - create tables and initial data.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from server.db.database import database
from server.db.models import Base
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database():
    """Initialize database with tables."""
    try:
        # Test connection
        with database.engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info(f"Database connection successful: {result.scalar()}")

        # Create all tables
        logger.info("Creating database tables...")
        database.create_tables()

        logger.info("Database initialization completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


if __name__ == "__main__":
    success = init_database()
    sys.exit(0 if success else 1)