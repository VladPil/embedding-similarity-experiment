"""
Task progress broadcaster for WebSocket connections.
"""

import asyncio
import json
from typing import Set, Dict
from fastapi import WebSocket
from redis import asyncio as aioredis

from .config import get_redis_url, get_queue_settings


class TaskProgressBroadcaster:
    """Broadcasts task progress updates to WebSocket clients."""

    def __init__(self):
        self.settings = get_queue_settings()
        self.redis_client: aioredis.Redis | None = None
        self.active_connections: Dict[str, Set[WebSocket]] = {}  # task_id -> set of websockets
        self._listening = False

    async def connect(self, websocket: WebSocket, task_id: str):
        """
        Connect a WebSocket client to task updates.

        Args:
            websocket: WebSocket connection
            task_id: Task ID to monitor
        """
        await websocket.accept()

        if task_id not in self.active_connections:
            self.active_connections[task_id] = set()

        self.active_connections[task_id].add(websocket)

    def disconnect(self, websocket: WebSocket, task_id: str):
        """
        Disconnect a WebSocket client.

        Args:
            websocket: WebSocket connection
            task_id: Task ID
        """
        if task_id in self.active_connections:
            self.active_connections[task_id].discard(websocket)

            # Clean up empty sets
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]

    async def broadcast_to_task(self, task_id: str, message: dict):
        """
        Broadcast message to all clients monitoring a task.

        Args:
            task_id: Task ID
            message: Message to broadcast
        """
        if task_id not in self.active_connections:
            return

        # Get all connections for this task
        connections = list(self.active_connections[task_id])

        # Send to all connections
        disconnected = []
        for websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception:
                # Connection closed, mark for removal
                disconnected.append(websocket)

        # Remove disconnected clients
        for websocket in disconnected:
            self.disconnect(websocket, task_id)

    async def _get_redis(self) -> aioredis.Redis:
        """Get or create Redis client."""
        if self.redis_client is None:
            redis_url = get_redis_url()
            self.redis_client = await aioredis.from_url(
                redis_url,
                decode_responses=True,
            )
        return self.redis_client

    async def start_listening(self):
        """Start listening to Redis pub/sub for task updates."""
        if self._listening:
            return

        self._listening = True
        redis = await self._get_redis()
        pubsub = redis.pubsub()

        await pubsub.subscribe(self.settings.progress_channel)

        try:
            async for message in pubsub.listen():
                if not self._listening:
                    break

                if message["type"] == "message":
                    try:
                        # Parse progress update
                        data = json.loads(message["data"])
                        task_id = data.get("task_id")

                        if task_id:
                            # Broadcast to all clients monitoring this task
                            await self.broadcast_to_task(task_id, data)

                    except Exception as e:
                        print(f"Error processing progress update: {e}")

        finally:
            await pubsub.unsubscribe(self.settings.progress_channel)
            self._listening = False

    async def stop_listening(self):
        """Stop listening to Redis pub/sub."""
        self._listening = False

        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None

    def get_connection_count(self, task_id: str = None) -> int:
        """
        Get number of active connections.

        Args:
            task_id: Optional task ID to get count for specific task

        Returns:
            Connection count
        """
        if task_id:
            return len(self.active_connections.get(task_id, set()))

        return sum(len(conns) for conns in self.active_connections.values())


# Global broadcaster instance
_broadcaster: TaskProgressBroadcaster | None = None


def get_broadcaster() -> TaskProgressBroadcaster:
    """Get global broadcaster instance."""
    global _broadcaster

    if _broadcaster is None:
        _broadcaster = TaskProgressBroadcaster()

    return _broadcaster


async def start_broadcaster():
    """Start the global broadcaster."""
    broadcaster = get_broadcaster()
    # Start listening in background
    asyncio.create_task(broadcaster.start_listening())


async def stop_broadcaster():
    """Stop the global broadcaster."""
    broadcaster = get_broadcaster()
    await broadcaster.stop_listening()
