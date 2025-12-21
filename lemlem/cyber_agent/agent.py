"""CyberAgent implementation built on top of lemlem's adapter."""
from __future__ import annotations

import asyncio
from asyncio import TimeoutError
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional
import inspect

from ..adapter import LLMAdapter
from .config import AgentConfig, ToolSpec
from .store import ConversationStore, InMemoryConversationStore


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _convert_datetimes_to_str(obj: Any) -> Any:
    """Recursively convert datetime objects to ISO strings in nested structures."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: _convert_datetimes_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_convert_datetimes_to_str(item) for item in obj)
    return obj


def _safe_serialize_usage(usage: Any) -> Optional[Dict[str, Any]]:
    """
    Safely serialize any usage object to a JSON-compatible dict.
    Will always succeed by falling back to str() representation if needed.
    """
    if usage is None:
        return None

    # If it's already a dict, return it as-is
    if isinstance(usage, dict):
        return usage

    try:
        # Try Pydantic model_dump first
        if hasattr(usage, "model_dump"):
            result = usage.model_dump()
            return _convert_datetimes_to_str(result)
    except Exception:
        pass

    try:
        # Try vars() for regular objects
        if hasattr(usage, "__dict__"):
            result = vars(usage)
            return _convert_datetimes_to_str(result)
    except Exception:
        pass

    try:
        # Try extracting known attributes
        result = {}
        for attr in ["prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens", "cached_tokens"]:
            try:
                value = getattr(usage, attr, None)
                if value is not None:
                    result[attr] = value
            except Exception:
                continue
        if result:
            return _convert_datetimes_to_str(result)
    except Exception:
        pass

    # Final fallback: convert to string and wrap in a dict
    return {"usage_string": str(usage)}


class _AdapterTool:
    def __init__(self, spec: ToolSpec) -> None:
        self.name = spec.name
        self.description = spec.description
        self.input_schema = spec.parameters
        self.handler = spec.handler

        async def _wrapped_function(**kwargs: Any) -> Any:
            result = spec.handler(kwargs or {})
            if inspect.isawaitable(result):
                return await result
            return result

        self.function = _wrapped_function


class CyberAgent:
    """High-level assistant wrapper with pluggable storage and hooks."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        store: Optional[ConversationStore] = None,
        adapter: Optional[LLMAdapter] = None,
    ) -> None:
        self.config = config
        self.store = store or InMemoryConversationStore()
        self.llm = adapter or LLMAdapter(logging_callbacks=config.logging_callbacks)
        self._tools = [_AdapterTool(spec) for spec in config.tools]

    async def stream_chat(
        self,
        *,
        conversation_id: Optional[str],
        message: str,
        user_metadata: Optional[Dict[str, Any]] = None,
        logging_context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        conv_id = conversation_id or self.store.create_conversation(metadata={"conversation_id": conversation_id})
        user_message = {
            "id": f"user-{_now_iso()}",
            "role": "user",
            "content": message,
            "created_at": _now_iso(),
            "metadata": user_metadata or {},
        }
        if user_metadata and user_metadata.get("attachments"):
            # Keep attachments both on the message and in metadata for downstream consumers
            user_message["attachments"] = user_metadata.get("attachments")
        self.store.append_message(conv_id, user_message)
        yield {"type": "ack", "conversation_id": conv_id, "message_id": user_message["id"]}

        history = self.store.list_messages(conv_id)
        loop = asyncio.get_running_loop()
        event_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        def _handle_turn(turn: Dict[str, Any]) -> None:
            hook = self.config.event_hooks.on_event
            if hook:
                try:
                    hook(turn)
                except Exception:
                    pass
            loop.call_soon_threadsafe(event_queue.put_nowait, turn)

        payload = {
            "conversation": history,
            "latest_message": user_message,
        }

        logging_payload = dict(logging_context or {})
        logging_payload.setdefault("conversation_id", conv_id)

        response_future = loop.run_in_executor(
            None,
            lambda: self.llm.chat_json(
                system_prompt=self.config.system_prompt,
                user_payload=payload,
                model=self.config.model,
                temperature=self.config.temperature,
                tools=self._tools,
                max_tool_iterations=self.config.max_iterations,
                on_turn=_handle_turn,
                logging_context=logging_payload,
            ),
        )

        while True:
            if response_future.done() and event_queue.empty():
                break
            try:
                turn = await asyncio.wait_for(event_queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            for event in self._process_turn(conv_id, turn):
                yield event

        response = response_future.result()

        final_text = response.get("final_text") or response.get("text") or ""
        usage = response.get("usage")
        # Convert CompletionUsage object to dict for JSON serialization
        usage_dict = _safe_serialize_usage(usage)

        # Extract reasoning traces from the response
        reasoning_traces = response.get("reasoning_traces", [])
        reasoning_text = None
        if reasoning_traces:
            # Combine all reasoning texts into a single string
            reasoning_parts = [trace.get("text", "") for trace in reasoning_traces if trace.get("text")]
            if reasoning_parts:
                reasoning_text = "\n\n".join(reasoning_parts)

        assistant_message = {
            "id": f"assistant-{_now_iso()}",
            "role": "assistant",
            "content": final_text,
            "created_at": _now_iso(),
            "usage": usage_dict,
        }

        # Add reasoning if present
        if reasoning_text:
            assistant_message["reasoning"] = reasoning_text
        self.store.append_message(conv_id, assistant_message)

        chunk_size = 280
        for start in range(0, len(final_text), chunk_size):
            chunk = final_text[start : start + chunk_size]
            if chunk:
                yield {"type": "text", "conversation_id": conv_id, "content": chunk}

        yield {
            "type": "message",
            "conversation_id": conv_id,
            "message": assistant_message,
            "usage": usage_dict,
        }
        yield {"type": "done", "conversation_id": conv_id}

    def _process_turn(self, conversation_id: str, turn: Dict[str, Any]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        turn_type = turn.get("type")
        metadata = turn.get("metadata", {})
        if turn_type == "tool_call":
            tool_name = metadata.get("tool_name", "unknown")
            call_id = metadata.get("call_id")
            arguments = metadata.get("arguments", {})
            self.store.log_tool_execution(
                conversation_id,
                {
                    "phase": "start",
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "arguments": arguments,
                },
            )
            events.append(
                {
                    "type": "tool_call",
                    "conversation_id": conversation_id,
                    "tool": tool_name,
                    "tool_call_id": call_id,
                    "args": arguments,
                    "status": "executing",
                }
            )
        elif turn_type == "tool_result":
            tool_name = metadata.get("tool_name", "unknown")
            call_id = metadata.get("call_id")
            arguments = metadata.get("arguments", {})
            is_error = metadata.get("is_error", False)
            status = "error" if is_error else "completed"
            result_payload = turn.get("content")
            self.store.log_tool_execution(
                conversation_id,
                {
                    "phase": "end",
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "arguments": arguments,
                    "status": status,
                    "result": result_payload,
                    "error": metadata.get("error"),
                },
            )
            tool_message = {
                "id": f"tool-{call_id}-{_now_iso()}",
                "role": "tool",
                "content": result_payload,
                "created_at": _now_iso(),
                "tool_name": tool_name,
                "tool_call_id": call_id,
            }
            self.store.append_message(conversation_id, tool_message)
            events.append(
                {
                    "type": "tool_result",
                    "conversation_id": conversation_id,
                    "tool": tool_name,
                    "tool_call_id": call_id,
                    "status": status,
                    "result": result_payload,
                }
            )
        return events
