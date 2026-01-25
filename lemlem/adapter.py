"""High-level LLM adapter with tool orchestration and logging hooks."""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .client import LLMClient, LLMResult
from .costs import compute_cost_for_model
from .costs import extract_cached_tokens as lemlem_extract_cached_tokens
from .models import load_models_from_env
from openai._exceptions import BadRequestError


logger = logging.getLogger(__name__)


# --- External Model Data Resolution (Decoupled from Main Repo) ---
#
# These callbacks allow lemlem to stay "pure" while still supporting
# the database-backed configuration logic used in the main repository.

@dataclass
class ExternalModelDataResolver:
    """Hooks for resolving model data from an external source (e.g. database)."""
    load_bundle: Optional[Callable[[], Dict[str, Any]]] = None
    get_timestamp: Optional[Callable[[], Any]] = None
    ensure_configured: Optional[Callable[[], None]] = None

_EXTERNAL_RESOLVER: Optional[ExternalModelDataResolver] = None

def set_external_model_data_resolver(resolver: ExternalModelDataResolver) -> None:
    """Configure lemlem to use an external source for model configuration."""
    global _EXTERNAL_RESOLVER
    _EXTERNAL_RESOLVER = resolver
    # Re-initialize global MODEL_DATA on registration
    _refresh_model_data(force=True)

# --- Internal State Management ---

def _apply_database_env_vars(env_map: Dict[str, Dict[str, Any]]) -> None:
    """Hydrate process env with DB-managed secrets (API keys, base URLs, etc.)."""
    for key, payload in (env_map or {}).items():
        if not isinstance(payload, dict):
            continue
        value = payload.get("value")
        if value is None:
            continue
        normalized_key = key.strip().upper()
        if normalized_key:
            os.environ[normalized_key] = str(value)


def _load_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Resolve model configs with the following preference order:
    1) External resolver (e.g. Database-backed model_config_service)
    2) MODELS_CONFIG_PATH (explicit file override)
    3) Empty valid structure (to be populated later)
    """
    if _EXTERNAL_RESOLVER and _EXTERNAL_RESOLVER.load_bundle:
        if _EXTERNAL_RESOLVER.ensure_configured:
            _EXTERNAL_RESOLVER.ensure_configured()
        try:
            bundle = _EXTERNAL_RESOLVER.load_bundle()
            _apply_database_env_vars(bundle.get("env_vars", {}))
            return {
                "models": bundle.get("models", {}),
                "configs": bundle.get("configs", {}),
            }
        except Exception as exc:
            logger.warning("External model data resolution failed: %s", exc)

    if os.environ.get("LEMLEM_MODELS_CONFIG_PATH"):
        try:
            return load_models_from_env()
        except Exception as exc:
            logger.warning("Failed to load models from LEMLEM_MODELS_CONFIG_PATH: %s", exc)

    # Return empty valid structure if no config is available yet.
    # This avoids crashing on module import in monorepo contexts where
    # the external resolver is registered shortly after import.
    return {"models": {}, "configs": {}}


MODEL_DATA = _load_model_configs()
_MODEL_DATA_TIMESTAMP: Optional[Any] = None

# Initialize timestamp if resolver is present
if _EXTERNAL_RESOLVER and _EXTERNAL_RESOLVER.get_timestamp:
    try:
        _MODEL_DATA_TIMESTAMP = _EXTERNAL_RESOLVER.get_timestamp()
    except Exception:
        pass


def _refresh_model_data(force: bool = False) -> None:
    """Refresh MODEL_DATA from the external resolver if configs have been updated."""
    global MODEL_DATA, _MODEL_DATA_TIMESTAMP

    if not _EXTERNAL_RESOLVER:
        return

    if _EXTERNAL_RESOLVER.ensure_configured:
        _EXTERNAL_RESOLVER.ensure_configured()
        
    try:
        remote_timestamp = None
        if _EXTERNAL_RESOLVER.get_timestamp:
            remote_timestamp = _EXTERNAL_RESOLVER.get_timestamp()

        # If we don't have a timestamp or the remote is newer, refresh
        if force or _MODEL_DATA_TIMESTAMP is None or (remote_timestamp is not None and remote_timestamp > _MODEL_DATA_TIMESTAMP):
            logger.info("Model configs updated; refreshing MODEL_DATA")
            MODEL_DATA = _load_model_configs()
            _MODEL_DATA_TIMESTAMP = remote_timestamp
    except Exception as exc:
        logger.debug("Could not refresh model data: %s", exc)


def _extract_error_message(error: Exception) -> str:
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        error_section = body.get("error")
        if isinstance(error_section, dict):
            message = error_section.get("message")
            if isinstance(message, str) and message.strip():
                return message
    return str(error)


def _is_temperature_override_error(error: Exception) -> bool:
    """Detect provider errors complaining about non-default temperature."""
    message = _extract_error_message(error).lower()
    if not message:
        return False
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        error_section = body.get("error", {})
        if isinstance(error_section, dict):
            code = str(error_section.get("code") or "").lower()
            if code == "unsupported_value" and "temperature" in message:
                return True
    return (
        "unsupported value" in message
        and "temperature" in message
        and "default (1" in message
    )


def _model_meta(model: str, *, force_standard: bool = False) -> Dict[str, Any]:
    cfg = MODEL_DATA.get("configs", {}).get(model) or {}
    meta = cfg.get("_meta") if isinstance(cfg, dict) else {}
    meta = dict(meta or {})
    if force_standard:
        meta["is_thinking"] = False
    return meta


def _model_meta_from_data(
    model: str,
    *,
    model_data: Dict[str, Dict[str, Any]],
    force_standard: bool = False,
) -> Dict[str, Any]:
    cfg = model_data.get("configs", {}).get(model) or {}
    meta = cfg.get("_meta") if isinstance(cfg, dict) else {}
    meta = dict(meta or {})
    if force_standard:
        meta["is_thinking"] = False
    return meta


def _ensure_dict(target: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = target.get(key)
    if not isinstance(value, dict):
        value = {}
        target[key] = value
    return value


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
            return usage.model_dump()
    except Exception:
        pass

    try:
        # Try vars() for regular objects
        if hasattr(usage, "__dict__"):
            return vars(usage)
    except Exception:
        pass

    try:
        # Try extracting known attributes
        result = {}
        for attr in [
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "cached_tokens",
        ]:
            try:
                value = getattr(usage, attr, None)
                if value is not None:
                    result[attr] = value
            except Exception:
                continue
        if result:
            return result
    except Exception:
        pass

    # Final fallback: convert to string and wrap in a dict
    return {"usage_string": str(usage)}


@dataclass
class ModelCostEvent:
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    usd_cost: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCostEvent:
    tool_name: str
    usd_cost: float
    arguments: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggingCallbacks:
    on_model_cost: Optional[Callable[[ModelCostEvent], None]] = None
    on_tool_cost: Optional[Callable[[ToolCostEvent], None]] = None


class LLMAdapter:
    """Adapter around lemlem's LLMClient with optional tool + logging support."""

    def __init__(
        self,
        *,
        model_data: Optional[Dict[str, Dict[str, Any]]] = None,
        client: Optional[LLMClient] = None,
        force_standard_completions_api: bool = False,
        logging_callbacks: Optional[LoggingCallbacks] = None,
    ) -> None:
        self._provided_model_data = model_data
        self._provided_client = client
        self.model_data = model_data or MODEL_DATA
        self.client = client or LLMClient(self.model_data)
        self.force_standard = force_standard_completions_api
        self.logging_callbacks = logging_callbacks or LoggingCallbacks()
        self._client_model_timestamp = _MODEL_DATA_TIMESTAMP

    def _get_client(self) -> LLMClient:
        """Get the LLM client, refreshing if configs have been updated."""
        # Only refresh if we're using the global MODEL_DATA (not a provided model_data)
        if self._provided_model_data is None and self._provided_client is None:
            _refresh_model_data()
            # If timestamp changed, recreate the client
            if self._client_model_timestamp != _MODEL_DATA_TIMESTAMP:
                logger.info("Model configs timestamp changed; refreshing LLM client")
                self.model_data = MODEL_DATA
                self.client = LLMClient(self.model_data)
                self._client_model_timestamp = _MODEL_DATA_TIMESTAMP
        return self.client

    def _thinking_adjustments(
        self,
        model: str,
        temperature: Optional[float],
        max_output_tokens: Optional[int],
    ) -> Tuple[Optional[float], Optional[int]]:
        meta = _model_meta_from_data(
            model,
            model_data=self.model_data,
            force_standard=self.force_standard,
        )
        is_thinking = bool(meta.get("is_thinking"))
        if is_thinking:
            return 1.0, None
        return temperature, max_output_tokens

    def chat_json(
        self,
        system_prompt: str,
        user_payload: Dict[str, Any],
        *,
        model: str,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        tools: Optional[Sequence[Any]] = None,
        max_tool_iterations: int = 6,
        on_turn: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_model_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        logging_callbacks: Optional[LoggingCallbacks] = None,
        logging_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call the configured LLM with optional tool support and JSON output enforcement."""

        # Pin the effective client + model_data for the duration of this run.
        #
        # This avoids refreshing configs/client mid-loop (which can cause subtle drift across
        # multi-turn tool iterations).
        pinned_client = self._get_client()
        pinned_model_data = self.model_data
        pinned_logging_callbacks = logging_callbacks or self.logging_callbacks

        caller_frame = inspect.currentframe().f_back
        caller_name = caller_frame.f_code.co_name if caller_frame else "unknown"
        logger.info(
            "Calling LLM API (chat_json) from %s | model=%s | temp=%s | tools=%s",
            caller_name,
            model,
            temperature,
            bool(tools),
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, default=str)},
        ]

        temperature, max_output_tokens = self._thinking_adjustments(
            model, temperature, max_output_tokens
        )
        base_extra: Dict[str, Any] = {}
        if isinstance(max_output_tokens, int) and max_output_tokens > 0:
            base_extra["max_completion_tokens"] = max_output_tokens

        meta = _model_meta_from_data(
            model,
            model_data=pinned_model_data,
            force_standard=self.force_standard,
        )
        is_thinking = bool(meta.get("is_thinking"))
        use_tools = bool(tools)

        if json_schema:
            base_extra["structured_schema"] = json_schema
            base_extra["structured_name"] = "emit"
        elif not is_thinking and not use_tools:
            base_extra["force_json_object"] = True

        if is_thinking:
            reasoning_effort = meta.get("reasoning_effort")
            verbosity = meta.get("verbosity")
            if reasoning_effort:
                reasoning_section = _ensure_dict(base_extra, "reasoning")
                reasoning_section.setdefault("effort", reasoning_effort)
            if verbosity:
                text_section = _ensure_dict(base_extra, "text")
                format_section = text_section.get("format")
                if not isinstance(format_section, dict):
                    text_section["format"] = {"type": "text"}
                text_section.setdefault("verbosity", verbosity)

        tool_specs, tool_registry = self._prepare_tooling(tools)
        use_tools = bool(tool_specs)
        if use_tools:
            logger.info(
                "LLM chat_json with tools enabled | tools=%s | max_iterations=%s",
                list(tool_registry.keys()),
                max_tool_iterations,
            )

        last_usage = None
        model_used = None
        text_payload = "{}"
        tool_interactions: List[Dict[str, Any]] = []
        reasoning_traces: List[Dict[str, Any]] = []
        internal_turns: List[Dict[str, Any]] = []
        cost_entries: List[float] = []
        pending_turn_cost: Optional[float] = None
        pending_turn_usage: Optional[Any] = None

        def record_turn(turn: Dict[str, Any]) -> None:
            nonlocal pending_turn_cost, pending_turn_usage
            if pending_turn_cost is not None and "cost" not in turn:
                turn["cost"] = pending_turn_cost
                pending_turn_cost = None
            if pending_turn_usage is not None and "usage" not in turn:
                turn["usage"] = pending_turn_usage
                pending_turn_usage = None
            internal_turns.append(turn)
            if on_turn:
                on_turn(turn)

        for iteration in range(1, (max_tool_iterations if use_tools else 2) + 1):
            logger.info(
                "LLM tool iteration %s/%s | messages=%s | use_tools=%s | available_tools=%s",
                iteration,
                max_tool_iterations if use_tools else 2,
                len(messages),
                use_tools,
                list(tool_registry.keys()) if use_tools else [],
            )

            extra = dict(base_extra)
            if use_tools:
                extra["tools"] = tool_specs
                extra.setdefault("tool_choice", "auto")

            def _call_client(temp: Optional[float]) -> LLMResult:
                return pinned_client.generate(
                    model=model,
                    messages=messages,
                    temperature=temp,
                    extra=extra or None,
                    on_model_event=on_model_event,
                )

            try:
                result: LLMResult = _call_client(temperature)
            except BadRequestError as exc:
                if temperature not in (None, 1.0) and _is_temperature_override_error(
                    exc
                ):
                    logger.warning(
                        "LLMAdapter: temperature override triggered | model=%s | requested_temp=%s | error=%s",
                        model,
                        temperature,
                        _extract_error_message(exc),
                    )
                    result = _call_client(1.0)
                    temperature = 1.0
                else:
                    raise
            model_used = result.model_used

            turn_cost = result.get_cost(pinned_model_data)
            pending_turn_cost = turn_cost
            cost_entries.append(turn_cost)

            usage = getattr(result.raw, "usage", None)
            if usage is not None:
                last_usage = usage
                pending_turn_usage = usage

            logger.info(
                "LLM iteration %s response | has_output=%s | text_length=%s | text_preview=%s",
                iteration,
                hasattr(result.raw, "output") and bool(result.raw.output),
                len(result.text or ""),
                (result.text or "")[:200] if result.text else "<no text>",
            )

            if not use_tools:
                text_payload = self._extract_response_text(result)
                logger.info(
                    "LLM non-tool response: breaking with text_payload length %s",
                    len(text_payload),
                )
                break

            tool_calls: List[Any] = []
            content = ""

            if hasattr(result.raw, "output") and result.raw.output:
                for item in result.raw.output:
                    item_type = getattr(item, "type", None)
                    if item_type == "function_call":
                        tool_calls.append(item)
                    elif item_type == "reasoning":
                        reasoning_text_parts: List[str] = []
                        summary = getattr(item, "summary", None)
                        if summary:
                            for summary_item in summary:
                                extracted = getattr(summary_item, "text", None)
                                if extracted:
                                    reasoning_text_parts.append(str(extracted))
                        thought = getattr(item, "thought", None)
                        if thought:
                            reasoning_text_parts.append(str(thought))
                        reasoning_text = "".join(reasoning_text_parts).strip()
                        if reasoning_text:
                            reasoning_traces.append(
                                {
                                    "iteration": iteration,
                                    "type": "responses_api_reasoning",
                                    "text": reasoning_text,
                                }
                            )
                            record_turn(
                                {
                                    "type": "reasoning",
                                    "iteration": iteration,
                                    "content": reasoning_text,
                                    "metadata": {"source": "responses_api"},
                                }
                            )
                            content += reasoning_text
                    elif item_type == "message" and getattr(item, "content", None):
                        if isinstance(item.content, list):
                            for content_item in item.content:
                                if hasattr(content_item, "text"):
                                    content += content_item.text
                                elif (
                                    isinstance(content_item, dict)
                                    and "text" in content_item
                                ):
                                    content += str(content_item["text"])
                        else:
                            content += str(item.content)

            if not tool_calls:
                assistant_message = self._extract_assistant_message(result.raw)
                if assistant_message:
                    tool_calls = getattr(assistant_message, "tool_calls", None) or []
                    content = assistant_message.content or ""

            if tool_calls:
                # Filter out any tool calls with empty names to avoid provider errors
                valid_tool_calls = []
                for call in tool_calls:
                    if hasattr(call, "name"):
                        t_name = getattr(call, "name", "")
                    elif hasattr(call, "function"):
                        t_name = getattr(call.function, "name", "")
                    else:
                        t_name = ""
                    
                    if t_name:
                        valid_tool_calls.append(call)
                    else:
                        logger.warning(f"Skipping tool call with empty name in adapter loop: {call}")
                
                if not valid_tool_calls:
                    # If no valid tool calls remain, but we had some, check if we have content
                    if not content:
                        logger.warning("Tool calls were present but all had empty names, and no content found. Continuing to next turn.")
                    # We still need to add something to history if we want to continue, 
                    # but if we add an empty assistant message, some providers might complain.
                    # For now, we'll just not add this assistant_payload if it's completely empty.
                else:
                    assistant_payload = {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": getattr(call, "call_id", getattr(call, "id", "")),
                                "type": "function",
                                "function": {
                                    "name": getattr(
                                        call,
                                        "name",
                                        getattr(
                                            getattr(call, "function", None), "name", ""
                                        ),
                                    ),
                                    "arguments": getattr(
                                        call,
                                        "arguments",
                                        getattr(
                                            getattr(call, "function", None),
                                            "arguments",
                                            "{}",
                                        ),
                                    )
                                    or "{}",
                                },
                            }
                            for call in valid_tool_calls
                        ],
                    }
                    if content:
                        assistant_payload["content"] = content
                    messages.append(assistant_payload)

                for call in valid_tool_calls:
                    if hasattr(call, "name"):
                        tool_name = getattr(call, "name", "")
                        call_id = getattr(call, "call_id", "")
                        raw_args = getattr(call, "arguments", "") or "{}"
                    else:
                        tool_name = (
                            getattr(call.function, "name", "")
                            if hasattr(call, "function")
                            else ""
                        )
                        call_id = getattr(call, "id", "")
                        raw_args = (
                            getattr(call.function, "arguments", "")
                            if hasattr(call, "function")
                            else "{}"
                        )

                    # Handle case where arguments might already be a dict
                    if isinstance(raw_args, dict):
                        parsed_args = raw_args
                    elif isinstance(raw_args, str):
                        try:
                            parsed_args = json.loads(raw_args) if raw_args else {}
                        except json.JSONDecodeError as json_err:
                            logging.warning(
                                "Failed to parse tool arguments as JSON | tool=%s | raw_args_preview=%s | error=%s",
                                tool_name,
                                raw_args[:200] if raw_args else "",
                                json_err,
                            )
                            parsed_args = {"raw": raw_args}
                    else:
                        # Unexpected type - log and use empty dict
                        logging.warning(
                            "Unexpected type for tool arguments | tool=%s | type=%s",
                            tool_name,
                            type(raw_args).__name__,
                        )
                        parsed_args = {}

                    tool_output = self._execute_tool(
                        tool_registry, tool_name, parsed_args
                    )
                    tool_cost = float(tool_output.get("total_cost", 0.0) or 0.0)
                    if tool_cost > 0 and pinned_logging_callbacks.on_tool_cost:
                        event = ToolCostEvent(
                            tool_name=tool_name or "unknown",
                            usd_cost=tool_cost,
                            arguments=parsed_args,
                            context=dict(logging_context or {}),
                        )
                        pinned_logging_callbacks.on_tool_cost(event)

                    record_turn(
                        {
                            "type": "tool_result",
                            "iteration": iteration,
                            "content": tool_output,
                            "metadata": {
                                "tool_name": tool_name,
                                "call_id": call_id,
                                "arguments": parsed_args,
                                "is_error": tool_output.get("is_error", False),
                            },
                        }
                    )
                    tool_interactions.append(
                        {
                            "iteration": iteration,
                            "tool_name": tool_name,
                            "call_id": call_id,
                            "arguments": parsed_args,
                            "output": tool_output,
                        }
                    )
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": json.dumps(tool_output, default=str),
                    }
                    messages.append(tool_message)

                continue

            if content:
                stripped = content.strip()
                if stripped.startswith("{") and stripped.endswith("}"):
                    text_payload = stripped
                    break
                if use_tools:
                    text_payload = content
                    break
                assistant_payload = {"role": "assistant", "content": content}
                reasoning_traces.append(
                    {
                        "iteration": iteration,
                        "type": "assistant_reasoning_text",
                        "text": content,
                    }
                )
                record_turn(
                    {
                        "type": "reasoning",
                        "iteration": iteration,
                        "content": content,
                        "metadata": {"source": "assistant_message"},
                    }
                )
                messages.append(assistant_payload)
                continue
            else:
                result_text = self._extract_response_text(result)
                stripped = result_text.strip()
                if stripped.startswith("{") and stripped.endswith("}"):
                    text_payload = stripped
                    break

        else:
            logger.warning(
                "Tool loop exited without producing a response; attempting fallback"
            )
            for message in reversed(messages[-5:]):
                if message.get("role") == "assistant" and message.get("content"):
                    content = message["content"].strip()
                    if content.startswith("{") and content.endswith("}"):
                        text_payload = content
                        break

        if not text_payload or text_payload == "{}":
            final_text = self._extract_response_text(result)
            if final_text and final_text != text_payload:
                text_payload = final_text

        logger.info(
            "LLM final text_payload | length=%s | type=%s | preview=%s",
            len(text_payload) if isinstance(text_payload, str) else 0,
            type(text_payload).__name__,
            str(text_payload)[:300],
        )

        usage = last_usage
        cached_tokens = lemlem_extract_cached_tokens(result.raw, result.provider)
        if usage is not None and pinned_logging_callbacks.on_model_cost:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            turn_cost = compute_cost_for_model(
                model_used or model,
                prompt_tokens,
                completion_tokens,
                cached_tokens,
                pinned_model_data,
            )
            event = ModelCostEvent(
                model_used=model_used or model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                usd_cost=turn_cost,
                context=dict(logging_context or {}),
            )
            pinned_logging_callbacks.on_model_cost(event)

        total_cost = sum(cost_entries) if cost_entries else 0.0
        return {
            "text": result.text,
            "usage": _safe_serialize_usage(usage),
            "model_used": model_used,
            "tool_interactions": tool_interactions,
            "reasoning_traces": reasoning_traces,
            "internal_turns": internal_turns,
            "final_text": text_payload,
            "cost": {
                "total_cost": total_cost,
                "cost_entries": cost_entries,
            },
        }

    def _prepare_tooling(
        self, tools: Optional[Sequence[Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not tools:
            return [], {}
        specs: List[Dict[str, Any]] = []
        registry: Dict[str, Any] = {}
        for tool in tools:
            name = getattr(tool, "name", None)
            if not name:
                continue
            parameters = self._tool_parameters(getattr(tool, "input_schema", {}))
            description = getattr(tool, "description", "")
            specs.append(
                {
                    "type": "function",
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                }
            )
            registry[name] = tool
        return specs, registry

    def _tool_parameters(self, schema: Any) -> Dict[str, Any]:
        if isinstance(schema, dict):
            if "type" in schema:
                return schema
            properties: Dict[str, Any] = {}
            required: List[str] = []
            for key, value in schema.items():
                required.append(key)
                if isinstance(value, dict):
                    properties[key] = value
                    continue
                json_type = self._python_type_to_json_type(value)
                properties[key] = {"type": json_type}
            return {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        }

    @staticmethod
    def _python_type_to_json_type(value: Any) -> str:
        mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            dict: "object",
            list: "array",
        }
        return mapping.get(value, "string")

    @staticmethod
    def _extract_response_text(result: Any) -> str:
        if hasattr(result, "text") and result.text:
            text = result.text
            if isinstance(text, str):
                return text
            if hasattr(text, "content"):
                content = getattr(text, "content", "")
                if isinstance(content, str):
                    return content
                return str(content) if content else ""
            return str(text)

        raw = getattr(result, "raw", None)
        if raw is not None:
            for attr in ("output_text", "content", "text"):
                candidate = getattr(raw, attr, None)
                if isinstance(candidate, str):
                    return candidate
        return ""

    @staticmethod
    def _extract_assistant_message(raw_response: Any) -> Any:
        choices = getattr(raw_response, "choices", None)
        if choices:
            first = choices[0]
            return getattr(first, "message", None)
        return None

    def _execute_tool(
        self, registry: Dict[str, Any], name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        tool = registry.get(name)
        if tool is None:
            return {
                "content": [
                    {"type": "text", "text": f"Error: tool '{name}' is not configured."}
                ],
                "is_error": True,
                "error": f"tool_not_configured: {name}",
            }

        if hasattr(tool, "function"):
            handler = tool.function
        else:
            handler = getattr(tool, "handler", None) or tool

        try:
            if hasattr(tool, "function"):
                try:
                    result = handler(**arguments)
                except TypeError as exc:
                    message = str(exc)
                    if (
                        "unexpected keyword" in message
                        or "positional arguments" in message
                    ):
                        result = handler(arguments or {})
                    else:
                        raise
            else:
                result = handler(arguments or {})
            if inspect.isawaitable(result):
                result = self._run_coroutine(result)
        except Exception as exc:  # pragma: no cover - defensive
            stacktrace = traceback.format_exc()
            return {
                "content": [
                    {"type": "text", "text": f"Error executing tool '{name}': {exc}"}
                ],
                "is_error": True,
                "error": str(exc),
                "stacktrace": stacktrace,
            }

        if isinstance(result, dict):
            return result
        return {"content": [{"type": "text", "text": str(result)}]}

    @staticmethod
    def _run_coroutine(coro: Any) -> Any:
        # Check if there's already a running event loop
        try:
            loop = asyncio.get_running_loop()
            # If we get here, there's already a loop running
            # We need to use a different approach - run in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            try:
                return asyncio.run(coro)
            except RuntimeError:
                # Fallback: create a new event loop
                loop = asyncio.get_event_loop_policy().new_event_loop()
                try:
                    return loop.run_until_complete(coro)
                finally:
                    try:

                        async def cleanup() -> None:
                            await loop.shutdown_asyncgens()

                        loop.run_until_complete(cleanup())
                    finally:
                        loop.close()


__all__ = [
    "LLMAdapter",
    "MODEL_DATA",
    "ModelCostEvent",
    "ToolCostEvent",
    "LoggingCallbacks",
    "ExternalModelDataResolver",
    "set_external_model_data_resolver",
]
