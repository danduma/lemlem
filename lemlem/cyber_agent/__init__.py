"""CyberAgent platform exports."""
from .config import AgentConfig, ToolSpec, EventHooks
from .store import ConversationStore, InMemoryConversationStore

__all__ = [
    "AgentConfig",
    "ToolSpec",
    "EventHooks",
    "ConversationStore",
    "InMemoryConversationStore",
    "CyberAgent",
    "TaskManager",
    "TaskBackend",
    "InMemoryTaskBackend",
    "Task",
]


def __getattr__(name):
    if name == "CyberAgent":
        from .agent import CyberAgent

        return CyberAgent
    if name == "TaskManager":
        from .task_manager import TaskManager

        return TaskManager
    if name in {"TaskBackend", "InMemoryTaskBackend", "Task"}:
        from .task_backend import InMemoryTaskBackend, Task, TaskBackend

        mapping = {
            "TaskBackend": TaskBackend,
            "InMemoryTaskBackend": InMemoryTaskBackend,
            "Task": Task,
        }
        return mapping[name]
    raise AttributeError(name)
