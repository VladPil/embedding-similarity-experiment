"""
Background task management system with async support.
Handles async computations with progress tracking.
"""

import asyncio
import uuid
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from loguru import logger

from server.config import settings


@dataclass
class Task:
    """Represents a background task."""
    id: str
    name: str
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0  # 0.0 to 100.0
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        data = asdict(self)
        # Convert datetime to ISO format
        for key in ['created_at', 'started_at', 'completed_at']:
            if data[key] is not None:
                data[key] = data[key].isoformat()
        return data


class TaskManager:
    """
    Manages background tasks with async support.
    Uses asyncio for task execution.
    """

    def __init__(self, max_workers: int = None):
        """
        Initialize task manager.

        Args:
            max_workers: Maximum number of concurrent tasks (None for settings default)
        """
        self.tasks: Dict[str, Task] = {}
        self.max_workers = max_workers or settings.tasks_max_workers
        self.running_tasks: Dict[str, asyncio.Task] = {}
        logger.info(f"Task manager initialized with max_workers={self.max_workers}")

    async def submit_task(
        self,
        func: Callable,
        *args,
        name: str = "Task",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Submit a task for background execution.

        Args:
            func: Async or sync function to execute
            *args: Positional arguments for func
            name: Task name
            metadata: Optional metadata
            **kwargs: Keyword arguments for func

        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            name=name,
            metadata=metadata or {}
        )

        self.tasks[task_id] = task

        # Create asyncio task
        asyncio_task = asyncio.create_task(
            self._run_task(task_id, func, args, kwargs)
        )
        self.running_tasks[task_id] = asyncio_task

        logger.info(f"Submitted task {task_id}: {name}")
        return task_id

    async def _run_task(
        self,
        task_id: str,
        func: Callable,
        args: tuple,
        kwargs: dict
    ):
        """
        Run a task and update its status.

        Args:
            task_id: Task ID
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
        """
        task = self.tasks[task_id]
        task.status = "running"
        task.started_at = datetime.now()

        try:
            # Add task management helpers to kwargs
            kwargs['_task_id'] = task_id
            kwargs['_task_manager'] = self

            # Execute function (handle both async and sync)
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))

            # Mark as completed
            task.status = "completed"
            task.result = result
            task.progress = 100.0
            task.completed_at = datetime.now()

            logger.info(f"Task {task_id} completed successfully")

        except Exception as e:
            # Mark as failed
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.now()

            logger.error(f"Task {task_id} failed: {e}", exc_info=True)

        finally:
            # Remove from running tasks
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def list_tasks(self, status: Optional[str] = None) -> List[Task]:
        """
        List all tasks, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of tasks
        """
        tasks = list(self.tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]

        # Sort by creation time (newest first)
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks

    def delete_task(self, task_id: str) -> bool:
        """
        Delete a task.

        Args:
            task_id: Task ID

        Returns:
            True if deleted, False if not found
        """
        if task_id in self.tasks:
            # Cancel if still running
            if task_id in self.running_tasks:
                self.running_tasks[task_id].cancel()
                del self.running_tasks[task_id]

            del self.tasks[task_id]
            logger.info(f"Deleted task {task_id}")
            return True
        return False

    def clear_completed(self) -> int:
        """
        Clear all completed/failed tasks.

        Returns:
            Number of tasks cleared
        """
        to_delete = [
            tid for tid, task in self.tasks.items()
            if task.status in ("completed", "failed")
        ]

        for tid in to_delete:
            del self.tasks[tid]

        logger.info(f"Cleared {len(to_delete)} completed tasks")
        return len(to_delete)

    def update_progress(self, task_id: str, progress: float, message: str = ""):
        """
        Update task progress (called from within task function).

        Args:
            task_id: Task ID
            progress: Progress percentage (0-100)
            message: Optional progress message
        """
        if task_id in self.tasks:
            self.tasks[task_id].progress = min(100.0, max(0.0, progress))
            if message:
                self.tasks[task_id].metadata['progress_message'] = message
            logger.debug(f"Task {task_id} progress: {progress:.1f}% - {message}")

    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Task:
        """
        Wait for a task to complete.

        Args:
            task_id: Task ID
            timeout: Optional timeout in seconds

        Returns:
            Completed task

        Raises:
            asyncio.TimeoutError: If timeout is reached
            KeyError: If task not found
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        if task_id in self.running_tasks:
            try:
                await asyncio.wait_for(
                    self.running_tasks[task_id],
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Task {task_id} timed out")
                raise

        return self.tasks[task_id]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get task manager statistics.

        Returns:
            Dictionary with stats
        """
        statuses = {}
        for task in self.tasks.values():
            statuses[task.status] = statuses.get(task.status, 0) + 1

        return {
            'total_tasks': len(self.tasks),
            'running_tasks': len(self.running_tasks),
            'by_status': statuses,
            'max_workers': self.max_workers
        }


# Global task manager instance
_task_manager = None


def get_task_manager() -> TaskManager:
    """Get or create global task manager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
