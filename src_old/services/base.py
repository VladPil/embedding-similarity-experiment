"""
Base service class with dependency injection.
Implements Service Layer pattern for business logic.
"""

from abc import ABC, abstractmethod
from typing import Optional
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """
    Base service class.
    All services should inherit from this class.
    """

    def __init__(self, db: Session):
        """
        Initialize service with database session.

        Args:
            db: Database session (injected dependency)
        """
        self.db = db
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def initialize(self):
        """Initialize service (called once on startup)."""
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleanup service resources (called on shutdown)."""
        pass

    def log_info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def log_error(self, message: str, exception: Optional[Exception] = None):
        """Log error message."""
        if exception:
            self.logger.error(f"{message}: {str(exception)}", exc_info=True)
        else:
            self.logger.error(message)

    def log_debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)