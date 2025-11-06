"""
WebSocket endpoints for real-time updates.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from server.system.queue.broadcaster import get_broadcaster


router = APIRouter()


@router.websocket("/ws/tasks/{task_id}")
async def websocket_task_progress(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for real-time task progress updates.

    Args:
        websocket: WebSocket connection
        task_id: Task ID to monitor

    Message format:
    {
        "task_id": "uuid",
        "status": "in_progress",
        "progress": 45.0,
        "elapsed_time": 123.5,
        "estimated_time": 200.0,
        "current_step": "Analyzing pace...",
        "error": null
    }
    """
    broadcaster = get_broadcaster()

    # Connect to broadcaster
    await broadcaster.connect(websocket, task_id)

    try:
        # Keep connection alive and listen for client messages
        while True:
            # Wait for any message from client (ping/pong)
            data = await websocket.receive_text()

            # Echo back if needed (for ping)
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        # Client disconnected
        broadcaster.disconnect(websocket, task_id)
    except Exception as e:
        print(f"WebSocket error for task {task_id}: {e}")
        broadcaster.disconnect(websocket, task_id)
