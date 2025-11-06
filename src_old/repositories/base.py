"""
Base repository with common CRUD operations.
Implements Repository Pattern for database abstraction.
"""

from typing import Generic, TypeVar, Type, Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from abc import ABC, abstractmethod
import logging

from server.db.models import Base

logger = logging.getLogger(__name__)

ModelType = TypeVar("ModelType", bound=Base)


class IRepository(ABC, Generic[ModelType]):
    """Repository interface (for Dependency Inversion Principle)."""

    @abstractmethod
    async def get(self, id: Any) -> Optional[ModelType]:
        """Get entity by ID."""
        pass

    @abstractmethod
    async def get_all(self, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """Get all entities with pagination."""
        pass

    @abstractmethod
    def create(self, **kwargs) -> ModelType:
        """Create new entity."""
        pass

    @abstractmethod
    def update(self, id: Any, **kwargs) -> Optional[ModelType]:
        """Update existing entity."""
        pass

    @abstractmethod
    def delete(self, id: Any) -> bool:
        """Delete entity by ID."""
        pass


class BaseRepository(IRepository[ModelType]):
    """
    Base repository implementation with common CRUD operations.
    Uses Generic typing for model flexibility.
    """

    def __init__(self, model: Type[ModelType], db: Session):
        """
        Initialize repository.

        Args:
            model: SQLAlchemy model class
            db: Database session
        """
        self.model = model
        self.db = db

    def get(self, id: Any) -> Optional[ModelType]:
        """
        Get entity by ID.

        Args:
            id: Entity ID

        Returns:
            Entity or None if not found
        """
        try:
            return self.db.query(self.model).filter(self.model.id == id).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting {self.model.__name__} with id {id}: {e}")
            return None

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ModelType]:
        """
        Get all entities with optional filtering and pagination.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Optional filter conditions

        Returns:
            List of entities
        """
        try:
            query = self.db.query(self.model)

            # Apply filters if provided
            if filters:
                for key, value in filters.items():
                    if hasattr(self.model, key):
                        query = query.filter(getattr(self.model, key) == value)

            return query.offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting all {self.model.__name__}: {e}")
            return []

    def create(self, **kwargs) -> ModelType:
        """
        Create new entity.

        Args:
            **kwargs: Entity attributes

        Returns:
            Created entity

        Raises:
            SQLAlchemyError: If creation fails
        """
        try:
            entity = self.model(**kwargs)
            self.db.add(entity)
            self.db.commit()
            self.db.refresh(entity)
            logger.info(f"Created {self.model.__name__}: {entity}")
            return entity
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating {self.model.__name__}: {e}")
            raise

    def update(self, id: Any, **kwargs) -> Optional[ModelType]:
        """
        Update existing entity.

        Args:
            id: Entity ID
            **kwargs: Updated attributes

        Returns:
            Updated entity or None if not found
        """
        try:
            entity = self.get(id)
            if not entity:
                return None

            for key, value in kwargs.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)

            self.db.commit()
            self.db.refresh(entity)
            logger.info(f"Updated {self.model.__name__}: {entity}")
            return entity
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating {self.model.__name__} with id {id}: {e}")
            return None

    def delete(self, id: Any) -> bool:
        """
        Delete entity by ID.

        Args:
            id: Entity ID

        Returns:
            True if deleted, False if not found
        """
        try:
            entity = self.get(id)
            if not entity:
                return False

            self.db.delete(entity)
            self.db.commit()
            logger.info(f"Deleted {self.model.__name__} with id {id}")
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error deleting {self.model.__name__} with id {id}: {e}")
            return False

    def exists(self, id: Any) -> bool:
        """
        Check if entity exists.

        Args:
            id: Entity ID

        Returns:
            True if exists, False otherwise
        """
        try:
            return self.db.query(
                self.db.query(self.model).filter(self.model.id == id).exists()
            ).scalar()
        except SQLAlchemyError:
            return False

    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count entities with optional filtering.

        Args:
            filters: Optional filter conditions

        Returns:
            Number of entities
        """
        try:
            query = self.db.query(self.model)

            if filters:
                for key, value in filters.items():
                    if hasattr(self.model, key):
                        query = query.filter(getattr(self.model, key) == value)

            return query.count()
        except SQLAlchemyError as e:
            logger.error(f"Error counting {self.model.__name__}: {e}")
            return 0