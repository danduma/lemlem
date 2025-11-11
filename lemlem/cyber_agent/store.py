"""Conversation storage abstractions."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


Message = Dict[str, Any]


class ConversationStore:
    """Abstract interface for persisting CyberAgent conversations."""

    def create_conversation(self, *, metadata: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError

    def append_message(self, conversation_id: str, message: Message) -> None:
        raise NotImplementedError

    def list_messages(self, conversation_id: str) -> List[Message]:
        raise NotImplementedError

    def set_title(self, conversation_id: str, title: str) -> None:
        raise NotImplementedError

    def log_tool_execution(self, conversation_id: str, payload: Dict[str, Any]) -> None:
        del conversation_id, payload


@dataclass
class InMemoryConversation:
    title: Optional[str] = None
    messages: List[Message] = field(default_factory=list)


class InMemoryConversationStore(ConversationStore):
    """Simple in-memory store useful for tests or ephemeral agents."""

    def __init__(self) -> None:
        self._conversations: Dict[str, InMemoryConversation] = {}

    def create_conversation(self, *, metadata: Optional[Dict[str, Any]] = None) -> str:
        conversation_id = metadata.get("conversation_id") if metadata else None
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        self._conversations[conversation_id] = InMemoryConversation()
        return conversation_id

    def append_message(self, conversation_id: str, message: Message) -> None:
        convo = self._conversations.setdefault(conversation_id, InMemoryConversation())
        convo.messages.append(message)

    def list_messages(self, conversation_id: str) -> List[Message]:
        convo = self._conversations.get(conversation_id)
        if not convo:
            return []
        return list(convo.messages)

    def set_title(self, conversation_id: str, title: str) -> None:
        convo = self._conversations.setdefault(conversation_id, InMemoryConversation())
        convo.title = title
