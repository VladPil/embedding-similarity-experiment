"""
Task Manager integrated with FastStream.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from redis import asyncio as aioredis

from .config import get_redis_url, get_queue_settings
from .producers import submit_analysis_task, submit_comparison_task
from .models import TaskStatus, TaskProgress, TaskResult, TaskType


class FastStreamTaskManager:
    """Task manager using FastStream and Redis."""

    def __init__(self):
        self.settings = get_queue_settings()
        self.redis_client: Optional[aioredis.Redis] = None

    async def _get_redis(self) -> aioredis.Redis:
        """Get or create Redis client."""
        if self.redis_client is None:
            redis_url = get_redis_url()
            self.redis_client = await aioredis.from_url(
                redis_url,
                decode_responses=True,
            )
        return self.redis_client

    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()

    async def submit_book_analysis(
        self,
        text_id: str,
        text_title: str,
        text_content: str,
        analyses: List[str],
        chunk_size: int = 2000,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Submit book analysis task.

        Args:
            text_id: Text ID
            text_title: Text title
            text_content: Text content
            analyses: List of analysis types
            chunk_size: Chunk size
            metadata: Additional metadata

        Returns:
            str: Task ID
        """
        task_id = await submit_analysis_task(
            text_id=text_id,
            text_title=text_title,
            text_content=text_content,
            analyses=analyses,
            chunk_size=chunk_size,
            metadata=metadata,
        )

        # Store task metadata in Redis
        redis = await self._get_redis()
        task_key = f"task:{task_id}"
        task_data = {
            "task_id": task_id,
            "task_type": TaskType.ANALYSIS.value,
            "text_id": text_id,
            "text_title": text_title,
            "status": TaskStatus.PENDING.value,
            "created_at": datetime.utcnow().isoformat(),
            "analyses": json.dumps(analyses),
        }
        await redis.hset(task_key, mapping=task_data)
        await redis.expire(task_key, self.settings.status_ttl)

        return task_id

    async def submit_book_comparison(
        self,
        text1_id: str,
        text2_id: str,
        text1_title: str,
        text2_title: str,
        text1_content: str,
        text2_content: str,
        analyses: List[str],
        chunk_size: int = 2000,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Submit book comparison task.

        Args:
            text1_id: First text ID
            text2_id: Second text ID
            text1_title: First text title
            text2_title: Second text title
            text1_content: First text content
            text2_content: Second text content
            analyses: List of analysis types
            chunk_size: Chunk size
            metadata: Additional metadata

        Returns:
            str: Task ID
        """
        task_id = await submit_comparison_task(
            text1_id=text1_id,
            text2_id=text2_id,
            text1_title=text1_title,
            text2_title=text2_title,
            text1_content=text1_content,
            text2_content=text2_content,
            analyses=analyses,
            chunk_size=chunk_size,
            metadata=metadata,
        )

        # Store task metadata in Redis
        redis = await self._get_redis()
        task_key = f"task:{task_id}"
        task_data = {
            "task_id": task_id,
            "task_type": TaskType.COMPARISON.value,
            "text1_id": text1_id,
            "text2_id": text2_id,
            "status": TaskStatus.PENDING.value,
            "created_at": datetime.utcnow().isoformat(),
            "analyses": json.dumps(analyses),
        }
        await redis.hset(task_key, mapping=task_data)
        await redis.expire(task_key, self.settings.status_ttl)

        return task_id

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status.

        Args:
            task_id: Task ID

        Returns:
            Task status dict or None
        """
        redis = await self._get_redis()
        task_key = f"task:{task_id}"

        # Get task data
        task_data = await redis.hgetall(task_key)
        if not task_data:
            return None

        # Parse analyses if present
        if "analyses" in task_data:
            task_data["analyses"] = json.loads(task_data["analyses"])

        return task_data

    async def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        Get all tasks.

        Returns:
            List of task dicts
        """
        redis = await self._get_redis()

        # Find all task keys
        keys = await redis.keys("task:*")

        tasks = []
        for key in keys:
            task_data = await redis.hgetall(key)
            if task_data:
                # Parse analyses if present
                if "analyses" in task_data:
                    task_data["analyses"] = json.loads(task_data["analyses"])
                tasks.append(task_data)

        # Sort by created_at descending
        tasks.sort(
            key=lambda x: x.get("created_at", ""),
            reverse=True,
        )

        return tasks

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel task.

        Args:
            task_id: Task ID

        Returns:
            bool: True if cancelled, False otherwise
        """
        redis = await self._get_redis()
        task_key = f"task:{task_id}"

        # Check if task exists
        exists = await redis.exists(task_key)
        if not exists:
            return False

        # Update status to cancelled
        await redis.hset(
            task_key,
            "status",
            TaskStatus.CANCELLED.value,
        )

        return True

    async def delete_task(self, task_id: str) -> bool:
        """
        Delete task.

        Args:
            task_id: Task ID

        Returns:
            bool: True if deleted, False otherwise
        """
        redis = await self._get_redis()
        task_key = f"task:{task_id}"

        deleted = await redis.delete(task_key)
        return deleted > 0

    async def subscribe_to_progress(self, callback):
        """
        Subscribe to task progress updates.

        Args:
            callback: Async callback function to handle progress updates
        """
        redis = await self._get_redis()
        pubsub = redis.pubsub()

        await pubsub.subscribe(self.settings.progress_channel)

        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    await callback(data)
                except Exception as e:
                    print(f"Error processing progress update: {e}")
