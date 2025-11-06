"""
Repository for task history management.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
import logging

from server.repositories.base import BaseRepository
from server.db.models import TaskHistory, TaskStatus, AnalysisHistory

logger = logging.getLogger(__name__)


class TaskRepository(BaseRepository[TaskHistory]):
    """Repository for task history and tracking."""

    def __init__(self, db: Session):
        """Initialize task repository."""
        super().__init__(TaskHistory, db)

    def get_task(self, task_id: str) -> Optional[TaskHistory]:
        """
        Get task by ID.

        Args:
            task_id: Task ID

        Returns:
            Task or None
        """
        return self.db.query(TaskHistory).filter(
            TaskHistory.id == task_id
        ).first()

    def create_task(
        self,
        task_id: str,
        name: str,
        task_type: str,
        params: Dict[str, Any],
        text1_id: Optional[str] = None,
        text2_id: Optional[str] = None
    ) -> TaskHistory:
        """
        Create new task.

        Args:
            task_id: Unique task ID
            name: Task name
            task_type: Task type (semantic, style, etc.)
            params: Task parameters
            text1_id: First text ID (optional)
            text2_id: Second text ID (optional)

        Returns:
            Created task
        """
        task = TaskHistory(
            id=task_id,
            name=name,
            type=task_type,
            status=TaskStatus.PENDING,
            params=params,
            text1_id=text1_id,
            text2_id=text2_id,
            progress=0
        )

        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)

        logger.info(f"Created task: {task_id}, type: {task_type}")
        return task

    def start_task(self, task_id: str) -> Optional[TaskHistory]:
        """
        Mark task as running.

        Args:
            task_id: Task ID

        Returns:
            Updated task or None
        """
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(task)
        return task

    async def update_progress(
        self,
        task_id: str,
        progress: int,
        message: Optional[str] = None
    ) -> Optional[TaskHistory]:
        """
        Update task progress.

        Args:
            task_id: Task ID
            progress: Progress percentage (0-100)
            message: Progress message

        Returns:
            Updated task or None
        """
        updates = {"progress": progress}
        if message:
            updates["progress_message"] = message

        return await self.update(task_id, **updates)

    def complete_task(
        self,
        task_id: str,
        result: Dict[str, Any]
    ) -> Optional[TaskHistory]:
        """
        Mark task as completed with result.

        Args:
            task_id: Task ID
            result: Task result

        Returns:
            Updated task or None
        """
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.progress = 100
            task.completed_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(task)

            # Also save to analysis history for quick access
            self._save_to_history(task)

        return task

    def fail_task(
        self,
        task_id: str,
        error: str
    ) -> Optional[TaskHistory]:
        """
        Mark task as failed.

        Args:
            task_id: Task ID
            error: Error message

        Returns:
            Updated task or None
        """
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.FAILED
            task.error = error
            task.completed_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(task)
        return task

    async def get_recent_tasks(
        self,
        limit: int = 10,
        status: Optional[TaskStatus] = None
    ) -> List[TaskHistory]:
        """
        Get recent tasks.

        Args:
            limit: Maximum number of tasks
            status: Filter by status (optional)

        Returns:
            List of recent tasks
        """
        query = self.db.query(TaskHistory)

        if status:
            query = query.filter(TaskHistory.status == status)

        return query.order_by(
            desc(TaskHistory.created_at)
        ).limit(limit).all()

    async def get_running_tasks(self) -> List[TaskHistory]:
        """Get all running tasks."""
        return self.db.query(TaskHistory).filter(
            TaskHistory.status == TaskStatus.RUNNING
        ).all()

    async def get_task_stats(self) -> Dict[str, int]:
        """Get task statistics by status."""
        stats = {}
        for status in TaskStatus:
            count = self.db.query(TaskHistory).filter(
                TaskHistory.status == status
            ).count()
            stats[status.value] = count
        return stats

    async def cleanup_old_tasks(self, days: int = 30) -> int:
        """
        Delete old completed/failed tasks.

        Args:
            days: Keep tasks from last N days

        Returns:
            Number of deleted tasks
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        deleted = self.db.query(TaskHistory).filter(
            and_(
                TaskHistory.status.in_([TaskStatus.COMPLETED, TaskStatus.FAILED]),
                TaskHistory.completed_at < cutoff
            )
        ).delete()

        self.db.commit()
        logger.info(f"Cleaned up {deleted} old tasks")
        return deleted

    def _save_to_history(self, task: TaskHistory):
        """Save completed task to analysis history."""
        if not task.result:
            return

        history = AnalysisHistory(
            task_id=task.id,
            type=task.type,
            text1_title=task.params.get("text1_title", "Text 1"),
            text2_title=task.params.get("text2_title"),
            similarity=task.result.get("similarity"),
            interpretation=task.result.get("interpretation"),
            details=task.result
        )

        self.db.add(history)
        self.db.commit()


class AnalysisHistoryRepository(BaseRepository[AnalysisHistory]):
    """Repository for analysis history (quick access to results)."""

    def __init__(self, db: Session):
        """Initialize analysis history repository."""
        super().__init__(AnalysisHistory, db)

    async def get_recent_analyses(
        self,
        limit: int = 10,
        analysis_type: Optional[str] = None
    ) -> List[AnalysisHistory]:
        """
        Get recent analysis results.

        Args:
            limit: Maximum number of results
            analysis_type: Filter by type (optional)

        Returns:
            List of recent analyses
        """
        query = self.db.query(AnalysisHistory)

        if analysis_type:
            query = query.filter(AnalysisHistory.type == analysis_type)

        return query.order_by(
            desc(AnalysisHistory.created_at)
        ).limit(limit).all()

    async def search_history(
        self,
        text_title: str,
        limit: int = 10
    ) -> List[AnalysisHistory]:
        """
        Search analysis history by text title.

        Args:
            text_title: Text title to search
            limit: Maximum results

        Returns:
            List of matching analyses
        """
        return self.db.query(AnalysisHistory).filter(
            or_(
                AnalysisHistory.text1_title.ilike(f"%{text_title}%"),
                AnalysisHistory.text2_title.ilike(f"%{text_title}%")
            )
        ).limit(limit).all()

    async def get_stats_by_type(self) -> Dict[str, Dict[str, Any]]:
        """Get analysis statistics grouped by type."""
        types = self.db.query(AnalysisHistory.type).distinct().all()
        stats = {}

        for (analysis_type,) in types:
            analyses = self.db.query(AnalysisHistory).filter(
                AnalysisHistory.type == analysis_type
            ).all()

            similarities = [a.similarity for a in analyses if a.similarity]
            stats[analysis_type] = {
                "count": len(analyses),
                "avg_similarity": np.mean(similarities) if similarities else None,
                "min_similarity": min(similarities) if similarities else None,
                "max_similarity": max(similarities) if similarities else None
            }

        return stats