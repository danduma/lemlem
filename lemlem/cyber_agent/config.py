"""Configuration dataclasses for CyberAgent."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence

from ..adapter import LoggingCallbacks


ToolHandler = Callable[[Dict[str, Any]], Any]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: ToolHandler


@dataclass
class EventHooks:
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None
    on_error: Optional[Callable[[Dict[str, Any]], None]] = None


@dataclass
class AgentConfig:
    agent_id: str
    system_prompt: str
    model: str
    temperature: float = 0.7
    max_iterations: int = 6
    tools: Sequence[ToolSpec] = field(default_factory=list)
    logging_callbacks: LoggingCallbacks = field(default_factory=LoggingCallbacks)
    event_hooks: EventHooks = field(default_factory=EventHooks)
