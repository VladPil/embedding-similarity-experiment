"""
Task management API endpoints.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
from loguru import logger

from server.schemas.models import (
    TaskListResponse,
    TaskStatusResponse,
    TaskInfo
)
from server.core.tasks import get_task_manager

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("", response_model=TaskListResponse)
async def list_tasks(status: Optional[str] = None):
    """
    List all tasks, optionally filtered by status.

    Args:
        status: Optional status filter (pending, running, completed, failed)

    Returns:
        List of tasks
    """
    try:
        task_mgr = get_task_manager()
        tasks = task_mgr.list_tasks(status=status)

        # Convert to TaskInfo objects
        task_infos = []
        for task in tasks:
            task_infos.append(TaskInfo(
                id=task.id,
                name=task.name,
                status=task.status,
                progress=task.progress,
                result=task.result,
                error=task.error,
                created_at=task.created_at,
                metadata=task.metadata
            ))

        return TaskListResponse(tasks=task_infos)

    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """
    Get task manager statistics.

    Returns:
        Task statistics
    """
    try:
        task_mgr = get_task_manager()
        stats = task_mgr.get_stats()

        return {
            "success": True,
            **stats
        }

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task(task_id: str):
    """
    Get task status and result.

    Args:
        task_id: Task identifier

    Returns:
        Task status and result
    """
    try:
        task_mgr = get_task_manager()
        task = task_mgr.get_task(task_id)

        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")

        return TaskStatusResponse(
            task=TaskInfo(
                id=task.id,
                name=task.name,
                status=task.status,
                progress=task.progress,
                result=task.result,
                error=task.error,
                created_at=task.created_at,
                metadata=task.metadata
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{task_id}")
async def delete_task(task_id: str):
    """
    Delete task.

    Args:
        task_id: Task identifier

    Returns:
        Success message
    """
    try:
        task_mgr = get_task_manager()
        deleted = task_mgr.delete_task(task_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Task not found")

        return {"success": True, "message": f"Task {task_id} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_completed():
    """
    Clear all completed and failed tasks.

    Returns:
        Number of tasks cleared
    """
    try:
        task_mgr = get_task_manager()
        count = task_mgr.clear_completed()

        return {
            "success": True,
            "cleared": count,
            "message": f"Cleared {count} completed tasks"
        }

    except Exception as e:
        logger.error(f"Failed to clear tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))
