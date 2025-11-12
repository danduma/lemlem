# TaskManager - Background Task Processing for CyberAgent

## Overview

`TaskManager` enables background processing of CyberAgent conversations that continues independently of HTTP connections. This allows conversations to complete even if clients disconnect, switch tabs, or experience network issues.

## Key Features

- **Persistent execution**: Tasks continue running after clients disconnect
- **Pluggable backends**: Support for in-memory, SQL, Redis, and custom storage
- **Event streaming**: Optional real-time event streaming to connected clients
- **Task tracking**: Monitor status and metadata of running tasks
- **Clean separation**: Decouples task lifecycle from HTTP/WebSocket connections

## Quick Start

### Basic Usage (In-Memory)

```python
from lemlem.cyber_agent import (
    CyberAgent,
    AgentConfig,
    TaskManager,
    InMemoryTaskBackend,
    InMemoryConversationStore,
)

# Create agent
config = AgentConfig(
    agent_id="my-agent",
    system_prompt="You are a helpful assistant",
    model="claude-3-5-sonnet-20241022",
    temperature=0.7,
)
agent = CyberAgent(config, store=InMemoryConversationStore())

# Create task manager
task_manager = TaskManager(agent, InMemoryTaskBackend())

# Start a background task
task_id = await task_manager.start_task(
    conversation_id=None,  # None creates new conversation
    message="Hello, how are you?",
)

# Stream events (optional - task continues without this)
async for event in task_manager.stream_events(task_id):
    print(event)
    if event["type"] == "done":
        break
```

### FastAPI Integration

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from lemlem.cyber_agent import CyberAgent, TaskManager, InMemoryTaskBackend

app = FastAPI()

# Initialize agent and task manager
agent = CyberAgent(config, store=store)
task_manager = TaskManager(agent, InMemoryTaskBackend())

@app.post("/chat")
async def chat(message: str, conversation_id: str | None = None):
    # Start background task
    task_id = await task_manager.start_task(
        conversation_id=conversation_id,
        message=message,
    )

    # Stream events as Server-Sent Events (SSE)
    async def event_stream():
        async for event in task_manager.stream_events(task_id):
            yield f"data: {json.dumps(event)}\\n\\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )

@app.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    task = task_manager.get_task_status(task_id)
    return {
        "id": task.id,
        "status": task.status,
        "conversation_id": task.conversation_id,
        "created_at": task.created_at.isoformat(),
    }
```

## Task Backends

### InMemoryTaskBackend

Simple in-memory storage. Best for development and testing.

```python
from lemlem.cyber_agent import InMemoryTaskBackend

backend = InMemoryTaskBackend()
task_manager = TaskManager(agent, backend)
```

### Custom SQL Backend

Create a custom backend for your database:

```python
from lemlem.cyber_agent import TaskBackend, Task
from datetime import datetime

class MySQLTaskBackend(TaskBackend):
    def __init__(self, db_session):
        self.db = db_session

    def create_task(self, task_id, conversation_id, status="running", metadata=None):
        # Store task in your database
        task_record = TaskModel(
            id=task_id,
            conversation_id=conversation_id,
            status=status,
            metadata=metadata,
        )
        self.db.add(task_record)
        self.db.commit()

        return Task(
            id=task_id,
            conversation_id=conversation_id,
            status=status,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata=metadata,
        )

    def update_task_status(self, task_id, status, error=None):
        task = self.db.query(TaskModel).filter_by(id=task_id).first()
        if task:
            task.status = status
            task.error = error
            task.updated_at = datetime.utcnow()
            self.db.commit()

    def get_task(self, task_id):
        task = self.db.query(TaskModel).filter_by(id=task_id).first()
        if not task:
            return None
        return Task(
            id=task.id,
            conversation_id=task.conversation_id,
            status=task.status,
            created_at=task.created_at,
            updated_at=task.updated_at,
            error=task.error,
            metadata=task.metadata,
        )

    def list_tasks_by_conversation(self, conversation_id):
        tasks = self.db.query(TaskModel).filter_by(
            conversation_id=conversation_id
        ).all()
        return [self._to_task(t) for t in tasks]

    def get_running_tasks(self):
        tasks = self.db.query(TaskModel).filter_by(status="running").all()
        return [self._to_task(t) for t in tasks]

# Usage
backend = MySQLTaskBackend(db_session)
task_manager = TaskManager(agent, backend)
```

## API Reference

### TaskManager

#### `__init__(agent, backend=None)`

Create a new TaskManager instance.

**Parameters:**
- `agent` (CyberAgent): Agent instance to use for processing
- `backend` (TaskBackend, optional): Storage backend (defaults to InMemoryTaskBackend)

#### `async start_task(conversation_id, message, **kwargs) -> str`

Start a background task for processing a message.

**Parameters:**
- `conversation_id` (str | None): Existing conversation ID, or None to create new
- `message` (str): User message to process
- `user_metadata` (dict, optional): Metadata to attach to messages
- `logging_context` (dict, optional): Context for logging

**Returns:**
- `task_id` (str): Unique identifier for the task

**Raises:**
- `ValueError`: If a task is already running for this conversation

#### `async stream_events(task_id) -> AsyncIterator[dict]`

Stream events from a running task.

**Parameters:**
- `task_id` (str): Task identifier

**Yields:**
- Event dictionaries with types: `ack`, `text`, `tool_call`, `tool_result`, `message`, `done`, `error`

**Notes:**
- Task continues running even if no one is streaming
- Multiple consumers can stream events from the same task

#### `async cancel_task(task_id) -> bool`

Cancel a running task.

**Parameters:**
- `task_id` (str): Task identifier

**Returns:**
- `bool`: True if cancelled, False if not found or already done

#### `get_task_status(task_id) -> Task | None`

Get current status of a task.

**Parameters:**
- `task_id` (str): Task identifier

**Returns:**
- `Task`: Task object with status, or None if not found

#### `get_running_tasks() -> list[str]`

Get list of all currently running task IDs.

**Returns:**
- `list[str]`: List of task IDs

### Task

Dataclass representing a task.

**Attributes:**
- `id` (str): Unique task identifier
- `conversation_id` (str): Associated conversation ID
- `status` (str): Task status (running, completed, error, cancelled)
- `created_at` (datetime): Task creation time
- `updated_at` (datetime): Last update time
- `error` (str | None): Error message if status is error
- `metadata` (dict | None): Additional task metadata

## Event Types

Tasks emit the following event types:

| Type | Description | Fields |
|------|-------------|--------|
| `ack` | Task started, message acknowledged | `task_id`, `conversation_id`, `message_id` |
| `text` | Streaming text chunk from assistant | `content`, `conversation_id` |
| `tool_call` | Tool execution started | `tool`, `tool_call_id`, `args`, `status` |
| `tool_result` | Tool execution completed | `tool`, `tool_call_id`, `result`, `status` |
| `message` | Complete assistant message | `message`, `usage`, `conversation_id` |
| `done` | Task completed successfully | `task_id`, `conversation_id` |
| `error` | Task failed | `task_id`, `conversation_id`, `error`, `detail` |

## Best Practices

### 1. Use appropriate backends

- **Development**: `InMemoryTaskBackend` for quick prototyping
- **Production**: SQL/Redis backend for persistence and scalability

### 2. Handle disconnections gracefully

```python
try:
    async for event in task_manager.stream_events(task_id):
        yield event
except asyncio.CancelledError:
    # Client disconnected - task continues in background
    logger.info(f"Client disconnected, task {task_id} continues")
    raise
```

### 3. Poll for updates when reconnecting

```python
# Client reconnects after disconnection
@app.get("/tasks/{task_id}/status")
async def get_status(task_id: str):
    task = task_manager.get_task_status(task_id)
    if task.status == "running":
        return {"status": "running", "message": "Still processing..."}
    elif task.status == "completed":
        # Fetch final conversation from database
        return {"status": "completed", "conversation_id": task.conversation_id}
    else:
        return {"status": "error", "error": task.error}
```

### 4. Set task limits

```python
# Prevent concurrent tasks for same conversation
try:
    task_id = await task_manager.start_task(conversation_id, message)
except ValueError as e:
    return {"error": "Task already running for this conversation"}
```

## Comparison: Before vs After

### Before TaskManager

```python
# Tightly coupled to HTTP connection
async def chat(message: str):
    async for event in agent.stream_chat(None, message):
        yield event
        # If client disconnects, processing stops!
```

### After TaskManager

```python
# Decoupled from HTTP connection
async def chat(message: str):
    task_id = await task_manager.start_task(None, message)
    # Task continues even if client disconnects
    async for event in task_manager.stream_events(task_id):
        yield event
```

## FAQ

**Q: What happens if the server restarts?**
A: Tasks in memory are lost. Use a SQL/Redis backend to persist task state across restarts.

**Q: Can multiple clients stream from the same task?**
A: Yes! Multiple consumers can call `stream_events()` for the same task_id.

**Q: How do I know when a task completes?**
A: Listen for the `done` event, or poll `get_task_status()` until status is `completed`.

**Q: Can I cancel a task after starting it?**
A: Yes, use `await task_manager.cancel_task(task_id)`.

**Q: What's the overhead of TaskManager?**
A: Minimal - just an asyncio queue and task tracking dict. The agent does the heavy lifting.

## Examples

See the `evergreen` app for a complete production example using SQLTaskBackend:
- Backend: `/backend/app/assistant/service.py`
- SQL Backend: `/backend/app/assistant/task_backend.py`
- Frontend: `/frontend/src/components/dashboard/dashboard/assistant/`
