"""Background task manager for CyberAgent conversations."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from .agent import CyberAgent
from .task_backend import Task, TaskBackend, InMemoryTaskBackend


logger = logging.getLogger(__name__)


class TaskManager:
    """
    Manages background execution of CyberAgent conversations.

    Separates task lifecycle from HTTP connections, allowing conversations
    to continue processing even if clients disconnect.

    Usage:
        agent = CyberAgent(config, store=store)
        task_manager = TaskManager(agent, InMemoryTaskBackend())

        # Start a task
        task_id = await task_manager.start_task(conversation_id, message)

        # Stream events (optional - task continues even if not streaming)
        async for event in task_manager.stream_events(task_id):
            yield event
    """

    def __init__(
        self,
        agent: CyberAgent,
        backend: Optional[TaskBackend] = None,
    ):
        """
        Initialize TaskManager.

        Args:
            agent: CyberAgent instance to use for processing
            backend: TaskBackend for persisting task state (defaults to in-memory)
        """
        self.agent = agent
        self.backend = backend or InMemoryTaskBackend()
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._event_queues: Dict[str, asyncio.Queue[Dict[str, Any]]] = {}

    async def start_task(
        self,
        conversation_id: Optional[str],
        message: str,
        *,
        user_metadata: Optional[Dict[str, Any]] = None,
        logging_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start a background task for processing a conversation message.

        Args:
            conversation_id: Existing conversation ID, or None to create new
            message: User message to process
            user_metadata: Optional metadata to attach to messages
            logging_context: Optional context for logging

        Returns:
            task_id: Unique identifier for this task

        Raises:
            ValueError: If a task is already running for this conversation
        """
        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Check if conversation already has a running task
        if conversation_id:
            existing_tasks = self.backend.list_tasks_by_conversation(conversation_id)
            running = [t for t in existing_tasks if t.status == "running"]
            if running:
                raise ValueError(
                    f"Task already running for conversation {conversation_id}"
                )

        # Create task record
        self.backend.create_task(
            task_id=task_id,
            conversation_id=conversation_id or "new",
            status="running",
            metadata={
                "message": message[:100],  # Store truncated message for debugging
                "user_metadata": user_metadata,
            },
        )

        # Create event queue for this task
        event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1000)
        self._event_queues[task_id] = event_queue

        # Start background task
        async_task = asyncio.create_task(
            self._run_task(
                task_id=task_id,
                conversation_id=conversation_id,
                message=message,
                user_metadata=user_metadata,
                logging_context=logging_context,
                event_queue=event_queue,
            )
        )
        self._running_tasks[task_id] = async_task

        logger.info(f"[TaskManager] Started task {task_id} for conversation {conversation_id}")
        return task_id

    async def _run_task(
        self,
        task_id: str,
        conversation_id: Optional[str],
        message: str,
        user_metadata: Optional[Dict[str, Any]],
        logging_context: Optional[Dict[str, Any]],
        event_queue: asyncio.Queue[Dict[str, Any]],
    ) -> None:
        """Internal method to run the agent in background."""
        actual_conversation_id = conversation_id

        try:
            logger.info(f"[TaskManager] Running task {task_id}")

            # Stream chat through agent
            async for event in self.agent.stream_chat(
                conversation_id=conversation_id,
                message=message,
                user_metadata=user_metadata,
                logging_context=logging_context,
            ):
                # Track actual conversation ID (might be created by agent)
                if event.get("conversation_id"):
                    actual_conversation_id = str(event["conversation_id"])

                # Add task_id to all events
                event["task_id"] = task_id

                # Put event in queue for streaming
                try:
                    event_queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning(f"[TaskManager] Event queue full for task {task_id}")

            # Mark task as completed
            self.backend.update_task_status(task_id, "completed")

            # Update task with actual conversation ID
            if actual_conversation_id and actual_conversation_id != conversation_id:
                task = self.backend.get_task(task_id)
                if task:
                    task.conversation_id = actual_conversation_id

            # Send done event
            event_queue.put_nowait({
                "type": "done",
                "task_id": task_id,
                "conversation_id": actual_conversation_id,
            })

            logger.info(f"[TaskManager] Task {task_id} completed")

        except Exception as exc:
            logger.exception(f"[TaskManager] Task {task_id} failed: {exc}")

            # Mark task as error
            self.backend.update_task_status(task_id, "error", error=str(exc))

            # Send error event
            event_queue.put_nowait({
                "type": "error",
                "task_id": task_id,
                "conversation_id": actual_conversation_id,
                "error": str(exc),
                "detail": "Task failed to complete",
            })
            event_queue.put_nowait({
                "type": "done",
                "task_id": task_id,
                "conversation_id": actual_conversation_id,
            })

        finally:
            # Cleanup
            self._running_tasks.pop(task_id, None)
            # Keep event queue around briefly for final events to be consumed
            await asyncio.sleep(1.0)
            self._event_queues.pop(task_id, None)

    async def stream_events(self, task_id: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream events from a running task.

        This is optional - the task continues running even if no one is streaming.
        Multiple consumers can stream events from the same task.

        Args:
            task_id: Task identifier returned from start_task()

        Yields:
            Event dictionaries (ack, text, tool_call, tool_result, message, done, error)
        """
        event_queue = self._event_queues.get(task_id)
        if not event_queue:
            # Task might not exist or already completed
            task = self.backend.get_task(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")

            if task.status == "completed":
                yield {"type": "done", "task_id": task_id, "conversation_id": task.conversation_id}
            elif task.status == "error":
                yield {
                    "type": "error",
                    "task_id": task_id,
                    "conversation_id": task.conversation_id,
                    "error": task.error,
                }
                yield {"type": "done", "task_id": task_id, "conversation_id": task.conversation_id}
            return

        # Stream events from queue
        async_task = self._running_tasks.get(task_id)
        while True:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                yield event
                if event.get("type") == "done":
                    break
            except asyncio.TimeoutError:
                # Check if task is still running
                if async_task and async_task.done():
                    break

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            task_id: Task identifier

        Returns:
            True if task was cancelled, False if task not found or already completed
        """
        async_task = self._running_tasks.get(task_id)
        if not async_task or async_task.done():
            return False

        async_task.cancel()
        self.backend.update_task_status(task_id, "cancelled")
        logger.info(f"[TaskManager] Cancelled task {task_id}")
        return True

    def get_task_status(self, task_id: str) -> Optional[Task]:
        """
        Get the current status of a task.

        Args:
            task_id: Task identifier

        Returns:
            Task object with current status, or None if not found
        """
        return self.backend.get_task(task_id)

    def get_running_tasks(self) -> List[str]:
        """
        Get list of all currently running task IDs.

        Returns:
            List of task IDs
        """
        return list(self._running_tasks.keys())
