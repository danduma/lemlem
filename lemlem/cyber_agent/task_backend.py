"""Task backend interfaces for persisting background task state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Task:
    """Represents a background task for agent conversation processing."""

    id: str
    conversation_id: str
    status: str  # idle, running, completed, error
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None
    metadata: Optional[Dict] = None


class TaskBackend:
    """Abstract interface for persisting task state."""

    def create_task(
        self,
        task_id: str,
        conversation_id: str,
        status: str = "running",
        metadata: Optional[Dict] = None,
    ) -> Task:
        """Create a new task record."""
        raise NotImplementedError

    def update_task_status(
        self,
        task_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Update the status of an existing task."""
        raise NotImplementedError

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by ID."""
        raise NotImplementedError

    def list_tasks_by_conversation(self, conversation_id: str) -> List[Task]:
        """Get all tasks for a conversation."""
        raise NotImplementedError

    def get_running_tasks(self) -> List[Task]:
        """Get all currently running tasks."""
        raise NotImplementedError


class InMemoryTaskBackend(TaskBackend):
    """Simple in-memory task backend for development and testing."""

    def __init__(self):
        self._tasks: Dict[str, Task] = {}

    def create_task(
        self,
        task_id: str,
        conversation_id: str,
        status: str = "running",
        metadata: Optional[Dict] = None,
    ) -> Task:
        now = datetime.utcnow()
        task = Task(
            id=task_id,
            conversation_id=conversation_id,
            status=status,
            created_at=now,
            updated_at=now,
            metadata=metadata,
        )
        self._tasks[task_id] = task
        return task

    def update_task_status(
        self,
        task_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        task = self._tasks.get(task_id)
        if task:
            task.status = status
            task.updated_at = datetime.utcnow()
            if error:
                task.error = error

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def list_tasks_by_conversation(self, conversation_id: str) -> List[Task]:
        return [
            task for task in self._tasks.values()
            if task.conversation_id == conversation_id
        ]

    def get_running_tasks(self) -> List[Task]:
        return [
            task for task in self._tasks.values()
            if task.status == "running"
        ]
