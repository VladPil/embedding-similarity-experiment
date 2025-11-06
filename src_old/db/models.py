"""
Database models for the embedding similarity experiment.
Using SQLAlchemy ORM with PostgreSQL.
"""

from sqlalchemy import (
    Column, String, Integer, DateTime, Boolean,
    Float, JSON, ForeignKey, Index, UniqueConstraint,
    Enum as SQLEnum, Text as SQLText
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum
from typing import Optional

Base = declarative_base()


class TextStorageType(enum.Enum):
    """Storage type for texts."""
    DATABASE = "database"  # Short texts < 1000 chars
    FILE = "file"  # Long texts stored in files


class TaskStatus(enum.Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Text(Base):
    """
    Text storage model.
    Short texts (<1000 chars) stored in DB.
    Long texts stored in files with reference.
    """
    __tablename__ = "texts"

    id = Column(String(64), primary_key=True)
    title = Column(String(500), nullable=False)

    # Storage strategy
    storage_type = Column(SQLEnum(TextStorageType), nullable=False)
    content = Column(SQLText, nullable=True)  # For short texts
    file_path = Column(String(500), nullable=True)  # For long texts

    # Metadata
    length = Column(Integer, nullable=False)
    lines = Column(Integer, nullable=False)
    language = Column(String(10), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    embeddings = relationship("EmbeddingCache", back_populates="text", cascade="all, delete-orphan")
    tasks = relationship("TaskHistory", back_populates="text1", foreign_keys="TaskHistory.text1_id")

    __table_args__ = (
        Index("idx_texts_created_at", "created_at"),
        Index("idx_texts_title", "title"),
    )

    def __repr__(self):
        return f"<Text(id={self.id}, title={self.title}, type={self.storage_type.value})>"


class EmbeddingCache(Base):
    """
    Cache for text embeddings.
    Stores embeddings in DB with Redis cache layer.
    """
    __tablename__ = "embedding_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text_id = Column(String(64), ForeignKey("texts.id", ondelete="CASCADE"), nullable=False)
    model_name = Column(String(200), nullable=False)

    # Embedding data (stored as JSON array)
    embedding = Column(JSON, nullable=False)
    dimensions = Column(Integer, nullable=False)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    text = relationship("Text", back_populates="embeddings")

    __table_args__ = (
        UniqueConstraint("text_id", "model_name", name="uq_text_model"),
        Index("idx_embedding_text_model", "text_id", "model_name"),
    )

    def __repr__(self):
        return f"<EmbeddingCache(text_id={self.text_id}, model={self.model_name})>"


class TaskHistory(Base):
    """
    History of analysis tasks.
    Tracks all background and synchronous tasks.
    """
    __tablename__ = "task_history"

    id = Column(String(64), primary_key=True)
    name = Column(String(200), nullable=False)
    type = Column(String(50), nullable=False)  # semantic, style, tfidf, etc.
    status = Column(SQLEnum(TaskStatus), nullable=False, default=TaskStatus.PENDING)

    # Task parameters and results
    params = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(SQLText, nullable=True)

    # Progress tracking
    progress = Column(Integer, default=0)
    progress_message = Column(String(500), nullable=True)

    # Related texts
    text1_id = Column(String(64), ForeignKey("texts.id", ondelete="SET NULL"), nullable=True)
    text2_id = Column(String(64), nullable=True)  # Not FK because text might be deleted

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    text1 = relationship("Text", back_populates="tasks", foreign_keys=[text1_id])

    __table_args__ = (
        Index("idx_task_status", "status"),
        Index("idx_task_created_at", "created_at"),
        Index("idx_task_type", "type"),
    )

    def __repr__(self):
        return f"<TaskHistory(id={self.id}, type={self.type}, status={self.status.value})>"


class AnalysisHistory(Base):
    """
    History of analysis results for quick access.
    Stores completed analysis for history feature.
    """
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), nullable=True)

    # Analysis type and texts
    type = Column(String(50), nullable=False)
    text1_title = Column(String(500), nullable=False)
    text2_title = Column(String(500), nullable=True)

    # Results
    similarity = Column(Float, nullable=True)
    interpretation = Column(SQLText, nullable=True)
    details = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_analysis_created_at", "created_at"),
        Index("idx_analysis_type", "type"),
    )

    def __repr__(self):
        return f"<AnalysisHistory(id={self.id}, type={self.type})>"