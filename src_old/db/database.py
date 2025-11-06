"""
Database connection and session management.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from contextlib import contextmanager
from typing import Generator
import logging

from server.config import settings
from server.db.models import Base

logger = logging.getLogger(__name__)


class Database:
    """Database connection manager with proper session handling."""

    def __init__(self):
        """Initialize database connection."""
        self.engine = None
        self.SessionLocal = None
        self._initialize()

    def _initialize(self):
        """Initialize database engine and session factory."""
        # Build PostgreSQL URL
        database_url = (
            f"postgresql://{settings.database_user}:{settings.database_password}"
            f"@{settings.database_host}:{settings.database_port}/{settings.database_name}"
        )

        # Create engine with connection pooling
        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,  # Test connections before using
            pool_size=5,
            max_overflow=10,
            echo=settings.debug,  # SQL logging in debug mode
        )

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        logger.info(f"Database initialized: {settings.database_name}")

    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")

    def drop_tables(self):
        """Drop all tables in the database."""
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Database tables dropped")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get database session with automatic cleanup.

        Usage:
            with db.get_session() as session:
                # Use session
                session.query(Text).all()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_db(self) -> Generator[Session, None, None]:
        """
        Dependency for FastAPI to get database session.

        Usage in FastAPI:
            @app.get("/items")
            def get_items(db: Session = Depends(database.get_db)):
                return db.query(Item).all()
        """
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()

    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


# Global database instance (Singleton)
database = Database()


# Dependency for FastAPI
def get_db() -> Generator[Session, None, None]:
    """Get database session for FastAPI dependency injection."""
    yield from database.get_db()