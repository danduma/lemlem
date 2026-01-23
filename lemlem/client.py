from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass
import logging
import os
import random
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Union

# Conditionally import OpenAI based on LANGFUSE_BASE_URL environment variable
if os.getenv("LANGFUSE_BASE_URL") and len(os.getenv("LANGFUSE_BASE_URL", "").strip()) > 0:
    from langfuse.openai import OpenAI
else:
    from openai import OpenAI

# Import exceptions from the underlying openai library (same for both cases)
from openai._exceptions import (
    APIConnectionError,
    APIError,
    BadRequestError,
    RateLimitError,
)

try:
    from google.api_core import exceptions as google_exceptions  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    google_exceptions = None

GOOGLE_RATE_LIMIT_EXCEPTIONS: tuple[type, ...] = ()
if google_exceptions:
    _candidates = []
    for _name in ("ResourceExhausted", "TooManyRequests", "RateLimitExceeded"):
        _exc = getattr(google_exceptions, _name, None)
        if _exc is not None:
            _candidates.append(_exc)
    GOOGLE_RATE_LIMIT_EXCEPTIONS = tuple(_candidates)

logger = logging.getLogger("lemlem.client")

Messages = List[Dict[str, str]]  # [{"role": "user"|"system"|"assistant", "content": str}]


@dataclass
class LLMResult:
    text: str
    model_used: str
    provider: str
    raw: Any

    def get_usage(self) -> Optional[Any]:
        """Extract usage information from the raw response."""
        return getattr(self.raw, "usage", None)

    def get_cached_tokens(self) -> int:
        """Extract cached token count from the response."""
        from .costs import extract_cached_tokens
        return extract_cached_tokens(self.raw, self.provider)

    def get_generation_id(self) -> Optional[str]:
        """Extract generation ID from OpenRouter responses."""
        # OpenRouter responses include an 'id' field that can be used for cost retrieval
        if hasattr(self.raw, 'id'):
            generation_id = getattr(self.raw, 'id')
            if generation_id and isinstance(generation_id, str):
                return generation_id
        return None

    def get_cost(self, model_configs: Optional[Dict[str, Dict[str, Any]]] = None) -> float:
        """Compute cost for this LLM result.

        Args:
            model_configs: Model configuration dict (loaded automatically if None)

        Returns:
            Total cost in USD
        """
        from .costs import compute_cost_for_model

        usage = self.get_usage()
        if not usage:
            return 0.0

        def _extract_cost(value: Any) -> Optional[float]:
            """Return explicit cost if the provider supplies it (e.g., OpenRouter)."""
            try:
                if isinstance(value, dict):
                    raw_cost = value.get("cost")
                    if raw_cost is None:
                        raw_cost = value.get("total_cost")
                else:
                    raw_cost = getattr(value, "cost", None)
                    if raw_cost is None:
                        raw_cost = getattr(value, "total_cost", None)
                if raw_cost is None:
                    return None
                return float(raw_cost)
            except (TypeError, ValueError):
                return None

        direct_cost = _extract_cost(usage)
        if direct_cost is not None:
            return direct_cost

        prompt_tokens = getattr(usage, 'prompt_tokens', 0)
        completion_tokens = getattr(usage, 'completion_tokens', 0)
        cached_tokens = self.get_cached_tokens()

        try:
            return compute_cost_for_model(
                self.model_used,
                prompt_tokens,
                completion_tokens,
                cached_tokens,
                model_configs
            )
        except ValueError as e:
            # Log error but don't crash - cost computation failures shouldn't break LLM calls
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to compute cost for model {self.model_used}: {e}")
            return 0.0


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("utf-8", errors="ignore")
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, default=str)
        except Exception:
            return str(value)
    return str(value)


def prepare_tools_for_api(tools: List[Dict[str, Any]], use_responses_api: bool = False) -> List[Dict[str, Any]]:
    """
    Prepare tools for the appropriate LLM API format.

    Args:
        tools: List of tool definitions (expected in standard nested OpenAI format)
        use_responses_api: Whether formatting for OpenAI Responses API (flat) or Chat Completions API (nested)
                          Responses API uses FLAT format, Chat Completions uses NESTED format!

    Returns:
        List of formatted tool definitions
    """
    if not tools:
        return []

    formatted_tools = []
    for tool in tools:
        if use_responses_api:
            # Responses API wants FLAT format: {"type": "function", "name": "...", "description": "...", "parameters": {...}}
            if "function" in tool:
                # Convert nested to flat
                func = tool.get("function", {})
                formatted_tools.append({
                    "type": tool.get("type", "function"),
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })
            else:
                # Already flat - pass through
                formatted_tools.append(tool)
        else:
            # Chat Completions API wants NESTED format: {"type": "function", "function": {"name": "...", ...}}
            if "function" not in tool:
                # Convert flat to nested
                formatted_tools.append({
                    "type": tool.get("type", "function"),
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    }
                })
            else:
                # Already nested - pass through
                formatted_tools.append(tool)

    return formatted_tools


def _collect_text_parts(parts: Any) -> str:
    if not parts:
        return ""
    if isinstance(parts, str):
        return parts
    if isinstance(parts, bytes):
        return _stringify(parts)
    texts: List[str] = []
    iterable = parts if isinstance(parts, (list, tuple)) else [parts]
    for item in iterable:
        if item is None:
            continue
        if isinstance(item, str):
            candidate = item
        elif isinstance(item, bytes):
            candidate = _stringify(item)
        elif isinstance(item, dict):
            candidate = (
                item.get("text")
                or item.get("content")
                or item.get("output_text")
                or item.get("input_text")
                or item.get("value")
            )
        else:
            candidate = getattr(item, "text", None) or getattr(item, "content", None) or getattr(item, "output_text", None)
        candidate_text = _stringify(candidate) if candidate else ""
        if candidate_text:
            texts.append(candidate_text)
    return "".join(texts)


def _extract_chat_message_text(msg: Any) -> str:
    if msg is None:
        return ""

    candidates: List[str] = []

    candidates.append(_collect_text_parts(getattr(msg, "content", None)))
    candidates.append(_collect_text_parts(getattr(msg, "reasoning_content", None)))

    response_metadata = getattr(msg, "response_metadata", None)
    if response_metadata:
        if isinstance(response_metadata, dict):
            for key in ("output_text", "parsed", "final_output"):
                if response_metadata.get(key):
                    candidates.append(_stringify(response_metadata[key]))
        else:
            for key in ("output_text", "parsed", "final_output"):
                if hasattr(response_metadata, key):
                    value = getattr(response_metadata, key)
                    if value:
                        candidates.append(_stringify(value))

    model_extra = getattr(msg, "model_extra", None)
    if isinstance(model_extra, dict):
        for key in ("output_text", "parsed", "final_output"):
            if model_extra.get(key):
                candidates.append(_stringify(model_extra[key]))

    if hasattr(msg, "__dict__"):
        extra_dict = {k: v for k, v in msg.__dict__.items() if k not in {"content", "tool_calls"}}
        for key in ("output_text", "parsed", "final_output", "text"):
            if extra_dict.get(key):
                candidates.append(_stringify(extra_dict[key]))

    for candidate in candidates:
        if candidate:
            return candidate

    return ""


def _extract_responses_output_text(resp: Any) -> str:
    if resp is None:
        return ""

    texts: List[str] = []
    output_items = getattr(resp, "output", None)
    if not output_items and isinstance(resp, dict):
        output_items = resp.get("output")

    if output_items:
        iterable = output_items if isinstance(output_items, (list, tuple)) else [output_items]
        for item in iterable:
            if item is None:
                continue
            item_type = getattr(item, "type", None)
            if not item_type and isinstance(item, dict):
                item_type = item.get("type")
            if item_type == "message":
                content = getattr(item, "content", None)
                if content is None and isinstance(item, dict):
                    content = item.get("content")
                message_text = _collect_text_parts(content)
                if message_text:
                    texts.append(message_text)
            elif item_type == "reasoning":
                # Extract reasoning summary text
                summary = getattr(item, "summary", None)
                if summary and isinstance(summary, list):
                    for summary_item in summary:
                        if hasattr(summary_item, 'text'):
                            texts.append(summary_item.text)
                        elif isinstance(summary_item, dict) and 'text' in summary_item:
                            texts.append(summary_item['text'])
                # Also extract any direct content from reasoning items
                reasoning_content = getattr(item, "content", None)
                if reasoning_content:
                    reasoning_text = _collect_text_parts(reasoning_content)
                    if reasoning_text:
                        texts.append(reasoning_text)
            elif item_type in {"function_call_output", "tool_result"}:
                output_text = getattr(item, "output", None)
                if output_text is None and isinstance(item, dict):
                    output_text = item.get("output")
                candidate = _stringify(output_text)
                if candidate:
                    texts.append(candidate)
            else:
                candidate = _collect_text_parts(getattr(item, "content", None))
                if not candidate and isinstance(item, dict):
                    candidate = _collect_text_parts(item.get("content"))
                if candidate:
                    texts.append(candidate)
    if texts:
        return "".join(texts)

    fallback = getattr(resp, "output_text", None)
    fallback_text = _stringify(fallback)
    if fallback_text:
        return fallback_text

    body = getattr(resp, "body", None)
    body_content = getattr(body, "content", None)
    body_text = _collect_text_parts(body_content) if body_content else ""
    if body_text:
        return body_text

    # Final fallback - but skip resp.text if it's a config object
    generic = getattr(resp, "content", None)
    if generic:
        return _stringify(generic)

    # Last resort - check if text is actually a string (not a config object)
    text_attr = getattr(resp, "text", None)
    if text_attr and isinstance(text_attr, str):
        return text_attr

    logger.warning(f"[lemlem] Could not extract text from Responses API response. Response type: {type(resp)}")
    return ""


def _is_openai_base_url(base_url: Optional[str]) -> bool:
    if not base_url:
        return True
    return "api.openai.com" in base_url


def _messages_to_prompt(messages: Messages) -> str:
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


class LLMClient:
    def __init__(self, model_data: Dict[str, Dict[str, Any]]):
        # NEW FORMAT ONLY: Expect {"models": {...}, "configs": {...}}
        if not isinstance(model_data, dict):
            raise ValueError("model_data must be a dict")
        if "models" not in model_data or "configs" not in model_data:
            raise ValueError("model_data must have both 'models' and 'configs' sections")

        self.models_section = model_data["models"]
        self.configs_section = model_data["configs"]
        self._logger = logging.getLogger(__name__)
        # Cache GeminiWrapper instances to preserve thought_signatures across calls
        self._gemini_clients: Dict[str, Any] = {}
        # Track per-model key rotation, cooldowns, and rpm windows
        self._key_state: Dict[str, Dict[str, Any]] = {}
        self._rng = random.Random()

    def generate(
        self,
        *,
        model: Union[str, Sequence[str]],
        messages: Optional[Messages] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries_per_model: int = 1,
        retry_on_status: Iterable[int] = (408, 409, 429, 500, 502, 503, 504),
        backoff_base: float = 0.5,
        backoff_max: float = 8.0,
        extra: Optional[Dict[str, Any]] = None,
        on_model_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> LLMResult:
        chain = self._build_chain(model)
        retry_codes = set(retry_on_status)

        last_error: Optional[Exception] = None

        def _status_from_exc(exc: Exception, default: Optional[int] = None) -> Optional[int]:
            try:
                return getattr(exc, "status_code", None) or getattr(exc, "status", None) or default
            except Exception:
                return default

        def _emit_model_event(event: Dict[str, Any]) -> None:
            if not on_model_event:
                return
            try:
                on_model_event(event)
            except Exception:
                logger.debug("on_model_event callback failed", exc_info=True)

        for chain_idx, config_id in enumerate(chain):
            config_data = self.configs_section.get(config_id)
            if not config_data:
                raise KeyError(f"Unknown config: {config_id}")
            if not bool(config_data.get("enabled", True)):
                continue

            model_ids = config_data.get("models") or [config_data.get("model")]
            if not model_ids:
                continue

            for model_id in model_ids:
                config_failed = False
                failure_details: Optional[Dict[str, Any]] = None
                cfg = self._resolve_config(config_id, model_override=model_id)
                if not cfg.get("enabled", True) or not cfg.get("_model_enabled", True):
                    continue

                base_url = cfg.get("base_url")
                default_temp = cfg.get("default_temp")

                temp = temperature if temperature is not None else default_temp

                # Defensive: expand env vars and treat unresolved placeholders like "${VAR}" or empty strings as missing
                def _sanitize(value: Optional[str]) -> Optional[str]:
                    if not value:
                        return None
                    s = os.path.expandvars(str(value).strip())
                    if s.startswith("${") and s.endswith("}"):
                        return None
                    return s

                resolved_base_url = _sanitize(base_url)
                meta = cfg.get("_meta") if isinstance(cfg, dict) else {}
                if not isinstance(meta, dict):
                    meta = {}
                is_thinking_model = bool(meta.get("is_thinking"))
                is_openai_endpoint = _is_openai_base_url(resolved_base_url)
                use_responses = is_thinking_model and is_openai_endpoint

                # Prepare inputs
                if messages is not None:
                    input_text = _messages_to_prompt(messages)
                    chat_messages = messages
                else:
                    input_text = prompt or ""
                    chat_messages = [{"role": "user", "content": input_text}]

                keys = cfg.get("_keys", [])
                strategy = cfg.get("key_strategy", "sequential_on_failure")
                model_key = f"{config_id}:{model_id}"

                attempt = 0
                while True:
                    key_idx = self._choose_key_index(model_key, keys, strategy)
                    if key_idx is None:
                        state = self._ensure_key_state(model_key, len(keys))
                        cooldowns = state.get("cooldowns", {})
                        header_limits = state.get("header_limits", {})
                        rpm_windows = state.get("rpm", {})
                        rpd_windows = state.get("rpd", {})
                        tpm_windows = state.get("tpm", {})
                        tpd_windows = state.get("tpd", {})
                        limits_snapshot: Dict[str, Any] = {
                            "keys_total": len(keys),
                            "cooldowns": {str(idx): cooldowns.get(idx) for idx in range(len(keys))},
                            "header_limits": {str(idx): header_limits.get(idx) for idx in range(len(keys))},
                            "rpm_usage": {str(idx): len(rpm_windows.get(idx, [])) for idx in range(len(keys))},
                            "rpd_usage": {str(idx): len(rpd_windows.get(idx, [])) for idx in range(len(keys))},
                            "tpm_usage": {str(idx): sum(v[1] for v in tpm_windows.get(idx, [])) for idx in range(len(keys))},
                            "tpd_usage": {str(idx): sum(v[1] for v in tpd_windows.get(idx, [])) for idx in range(len(keys))},
                        }
                        err = RuntimeError("No available API keys (rate limit or cooldown)")
                        last_error = err
                        failure_details = {
                            "config_id": config_id,
                            "model_id": model_id,
                            "model_name": cfg.get("model_name"),
                            "base_url": resolved_base_url,
                            "error_type": type(err).__name__,
                            "error_message": str(err),
                            "status_code": None,
                            "action": "no_available_keys",
                            "limits": limits_snapshot,
                        }
                        self._logger.warning(
                            "LLM no available keys | config=%s | model=%s | limits=%s",
                            config_id,
                            cfg.get("model_name"),
                            limits_snapshot,
                        )
                        config_failed = True
                        break
                    attempt += 1
                    key_entry = keys[key_idx]
                    resolved_api_key = _sanitize(key_entry.get("key"))
                    if not resolved_api_key:
                        self._mark_failure(model_key, strategy, key_idx, len(keys))
                        continue

                    # Record rpm usage before the call to enforce limits
                    now_ts = time.time()
                    self._record_usage(model_key, key_idx, now_ts)

                    # Check if this is a Gemini endpoint - use native wrapper instead of OpenAI SDK
                    is_gemini_endpoint = resolved_base_url and "generativelanguage.googleapis.com" in resolved_base_url
                    if is_gemini_endpoint:
                        from .gemini_wrapper import GeminiWrapper
                        # Cache GeminiWrapper to preserve thought_signatures across calls
                        cache_key = f"{resolved_api_key}:{cfg['model_name']}"
                        if cache_key not in self._gemini_clients:
                            self._logger.info(f"Creating new GeminiWrapper for {cfg['model_name']}")
                            self._gemini_clients[cache_key] = GeminiWrapper(api_key=resolved_api_key, model_name=cfg['model_name'])
                        else:
                            self._logger.info(f"Reusing cached GeminiWrapper for {cfg['model_name']}")
                        gemini_client = self._gemini_clients[cache_key]
                    else:
                        client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url) if resolved_base_url else OpenAI(api_key=resolved_api_key)

                    try:
                        if is_gemini_endpoint:
                            # Use Gemini native API wrapper
                            extra_payload = dict(extra or {})
                            include_tools = extra_payload.pop("tools", None)
                            
                            # Enforce a 120s timeout by default if not specified
                            request_timeout = extra_payload.pop("timeout", 120.0)

                            self._logger.info(f"LLM Request Starting: {cfg['model_name']} (Gemini Native)")
                            start_time = time.time()
                            
                            # Pass timeout to the config/client if supported, or via http_options
                            # google-genai client supports 'config' with http_options for timeout
                            try:
                                response_dict = gemini_client.generate_content(
                                    messages=chat_messages,
                                    tools=include_tools,
                                    temperature=temp,
                                    timeout=request_timeout, 
                                )
                            except Exception as e:
                                duration = time.time() - start_time
                                self._logger.error(f"LLM Request Failed: {cfg['model_name']} after {duration:.2f}s - {e}")
                                raise e
                                
                            duration = time.time() - start_time
                            self._logger.info(f"LLM Request Finished: {cfg['model_name']} in {duration:.2f}s")

                            # Extract text and tool calls from response
                            choice = response_dict["choices"][0]
                            message = choice["message"]
                            text = message.get("content", "")
                            tool_calls = message.get("tool_calls")

                            # Create a mock response object similar to OpenAI
                            class MockResponse:
                                def __init__(self, data):
                                    # Convert tool_calls dicts to objects with attributes
                                    tool_calls_data = data["choices"][0]["message"].get("tool_calls")
                                    tool_calls_objects = None
                                    if tool_calls_data:
                                        tool_calls_objects = []
                                        for tc in tool_calls_data:
                                            # Create nested function object
                                            func_obj = type('obj', (object,), {
                                                'name': tc["function"]["name"],
                                                'arguments': tc["function"]["arguments"]
                                            })()
                                            # Create tool call object with function attribute
                                            tc_obj = type('obj', (object,), {
                                                'id': tc["id"],
                                                'type': tc["type"],
                                                'function': func_obj
                                            })()
                                            tool_calls_objects.append(tc_obj)

                                    self.choices = [type('obj', (object,), {
                                        'message': type('obj', (object,), {
                                            'content': data["choices"][0]["message"].get("content"),
                                            'tool_calls': tool_calls_objects
                                        })(),
                                        'finish_reason': data["choices"][0].get("finish_reason")
                                    })()]
                                    self.usage = type('obj', (object,), data["usage"])()
                                    self.model = data["model"]

                            resp = MockResponse(response_dict)

                            result = LLMResult(
                                text=text or "",
                                model_used=cfg["model_name"],
                                provider="gemini-native",
                                raw=resp,
                            )
                            usage = result.get_usage()
                            if usage:
                                total_tokens = getattr(usage, "total_tokens", 0)
                                self._record_token_usage(model_key, key_idx, now_ts, total_tokens)
                            return result
                        elif use_responses:
                            # Responses API for reasoning models (e.g., o1). Use `input`, not `messages`.
                            payload: Dict[str, Any] = {
                                "model": cfg["model_name"],
                                "input": input_text,
                            }
                            extra_payload = dict(extra or {})
                            extra_payload.pop("max_completion_tokens", None)
                            text_payload = extra_payload.pop("text", None)
                            if text_payload is not None:
                                payload["text"] = text_payload
                            else:
                                text_cfg: Dict[str, Any] = {"format": {"type": "text"}}
                                text_verbosity = meta.get("verbosity") or cfg.get("verbosity")
                                if text_verbosity:
                                    text_cfg["verbosity"] = text_verbosity
                                payload["text"] = text_cfg

                            reasoning_payload = extra_payload.pop("reasoning", None) or {}
                            effort = meta.get("reasoning_effort") or cfg.get("reasoning_effort")
                            if effort and "effort" not in reasoning_payload:
                                reasoning_payload["effort"] = effort
                            summary_value = meta.get("reasoning_summary")
                            if summary_value is None and "reasoning_summary" in cfg:
                                summary_value = cfg.get("reasoning_summary")
                            if summary_value is not None and "summary" not in reasoning_payload:
                                reasoning_payload["summary"] = summary_value
                            if reasoning_payload:
                                payload["reasoning"] = reasoning_payload

                            include_payload = extra_payload.pop("include", None)
                            if include_payload is not None:
                                payload["include"] = include_payload
                            else:
                                include_cfg = meta.get("include")
                                if include_cfg is None and "include" in cfg:
                                    include_cfg = cfg.get("include")
                                if include_cfg:
                                    payload["include"] = include_cfg

                            store_payload = extra_payload.pop("store", None)
                            if store_payload is not None:
                                payload["store"] = bool(store_payload)
                            else:
                                store_cfg = meta.get("store")
                                if store_cfg is None and "store" in cfg:
                                    store_cfg = cfg.get("store")
                                if store_cfg is not None:
                                    payload["store"] = bool(store_cfg)

                            include_tools = extra_payload.pop("tools", None)
                            if include_tools:
                                include_tools = prepare_tools_for_api(include_tools, use_responses_api=True)
                            include_tool_choice = extra_payload.pop("tool_choice", None)

                            # Note: temperature is not supported in Responses API for reasoning models
                            # Structured outputs via text.format for Responses API
                            structured_schema = extra_payload.pop("structured_schema", None)
                            structured_name = extra_payload.pop("structured_name", "response")
                            if structured_schema:
                                # For Responses API, structured outputs go in text.format, not response_format
                                if "text" not in payload:
                                    payload["text"] = {}
                                if not isinstance(payload["text"], dict):
                                    payload["text"] = {}
                                payload["text"]["format"] = {
                                    "type": "json_schema",
                                    "name": structured_name or "response",
                                    "schema": structured_schema,
                                    "strict": True,
                                }

                            # Remove usage parameter - not supported in Responses API
                            extra_payload.pop("usage", None)
                            payload.update(extra_payload)
                            if "tools" not in payload:
                                payload["tools"] = []
                            if include_tools is not None:
                                payload["tools"] = include_tools
                            if include_tool_choice is not None:
                                payload["tool_choice"] = include_tool_choice

                            resp = client.responses.create(**payload)
                            text = _extract_responses_output_text(resp)
                            result = LLMResult(
                                text=text or "",
                                model_used=cfg["model_name"],
                                provider="openai-responses",
                                raw=resp,
                            )
                            usage = result.get_usage()
                            if usage:
                                total_tokens = getattr(usage, "total_tokens", 0)
                                self._record_token_usage(model_key, key_idx, now_ts, total_tokens)
                            return result
                        else:
                            payload = {
                                "model": cfg["model_name"],
                                "messages": chat_messages,
                            }
                            extra_payload = dict(extra or {})
                            usage_payload = extra_payload.pop("usage", None)
                            extra_payload.pop("max_completion_tokens", None)
                            if temp is not None:
                                payload["temperature"] = temp
                            if extra_payload:
                                # Support structured outputs for chat.completions via function calling
                                structured_schema = extra_payload.pop("structured_schema", None)
                                structured_name = extra_payload.pop("structured_name", "emit")
                                force_json_object = extra_payload.pop("force_json_object", False)
                                if structured_schema:
                                    payload["tools"] = [{
                                        "type": "function",
                                        "function": {
                                            "name": structured_name or "emit",
                                            "description": "Return JSON matching the provided schema",
                                            "parameters": structured_schema,
                                        },
                                    }]
                                    payload["tool_choice"] = {"type": "function", "function": {"name": structured_name or "emit"}}
                                elif force_json_object:
                                    payload["response_format"] = {"type": "json_object"}

                                # Format tools for Chat Completions API
                                include_tools = extra_payload.pop("tools", None)
                                if include_tools:
                                    extra_payload["tools"] = prepare_tools_for_api(include_tools, use_responses_api=False)

                                payload.update(extra_payload)
                            # OpenRouter: Enable usage accounting (includes exact cost in final chunk)
                            if resolved_base_url and "openrouter.ai" in resolved_base_url:
                                usage_cfg = usage_payload
                                if not isinstance(usage_cfg, dict):
                                    usage_cfg = {} if usage_cfg is None else {"value": usage_cfg}
                                usage_cfg["include"] = True
                                # OpenAI Python client no longer accepts "usage" as a top-level kwarg.
                                # For OpenRouter, the flag must live under extra_body to avoid TypeError.
                                extra_body = payload.get("extra_body") or {}
                                extra_body.setdefault("usage", {}).update(usage_cfg)
                                payload["extra_body"] = extra_body
                            else:
                                # Ensure usage is not in payload for non-OpenRouter providers
                                payload.pop("usage", None)
                            resp_raw = client.chat.completions.with_raw_response.create(**payload)
                            resp = resp_raw.parse()
                            self._update_limits_from_headers(model_key, key_idx, dict(resp_raw.headers))
                        if not hasattr(resp, "choices"):
                            raw_preview = str(resp)
                            err = TypeError(
                                f"Unexpected chat.completions response type: {type(resp).__name__}"
                            )
                            last_error = err
                            failure_details = {
                                "config_id": config_id,
                                "model_id": model_id,
                                "model_name": cfg.get("model_name"),
                                "base_url": resolved_base_url,
                                "error_type": type(err).__name__,
                                "error_message": raw_preview[:500],
                                "status_code": None,
                                "action": "unexpected_response",
                            }
                            config_failed = True
                            break

                        text = ""
                        if resp.choices:
                            msg = resp.choices[0].message
                            text = _extract_chat_message_text(msg)
                            self._logger.debug(
                                "lemlem.chat_completion finish_reason=%s content_len=%s tool_calls=%s reasoning_present=%s",
                                getattr(resp.choices[0], "finish_reason", None),
                                len(getattr(msg, "content", "") or ""),
                                bool(getattr(msg, "tool_calls", None)),
                                bool(getattr(msg, "reasoning_content", None)),
                            )
                        result = LLMResult(
                            text=text or "",
                            model_used=cfg["model_name"],
                            provider="openai-compatible",
                            raw=resp,
                        )
                        usage = result.get_usage()
                        if usage:
                            total_tokens = getattr(usage, "total_tokens", 0)
                            self._record_token_usage(model_key, key_idx, now_ts, total_tokens)
                        return result
                    except GOOGLE_RATE_LIMIT_EXCEPTIONS as e:  # type: ignore[misc]
                        last_error = e
                        self._apply_cooldown(
                            model_key,
                            key_idx,
                            backoff_base=backoff_base,
                            backoff_max=backoff_max,
                        )
                        status_code = _status_from_exc(e, 429)
                        retry_rate_limits = bool(meta.get("retry_on_rate_limit", True))
                        failure_details = {
                            "config_id": config_id,
                            "model_id": model_id,
                            "model_name": cfg.get("model_name"),
                            "base_url": resolved_base_url,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "status_code": status_code,
                            "action": "rate_limit",
                        }
                        if retry_rate_limits and _should_retry_status(e, retry_codes, default=429):
                            self._mark_failure(model_key, strategy, key_idx, len(keys))
                            continue
                        config_failed = True
                        break
                    except RateLimitError as e:
                        last_error = e
                        # Try to extract retry-after from headers if available in exception
                        headers = getattr(e, "headers", {})
                        retry_after = headers.get("retry-after")
                        if retry_after:
                            state = self._ensure_key_state(model_key, key_idx + 1)
                            state["cooldowns"][key_idx] = time.time() + self._parse_duration(retry_after)
                        else:
                            self._apply_cooldown(
                                model_key,
                                key_idx,
                                backoff_base=backoff_base,
                                backoff_max=backoff_max,
                            )
                        
                        # Also update limits from other headers if present
                        if headers:
                            self._update_limits_from_headers(model_key, key_idx, dict(headers))
                            
                        retry_rate_limits = bool(meta.get("retry_on_rate_limit", True))
                        failure_details = {
                            "config_id": config_id,
                            "model_id": model_id,
                            "model_name": cfg.get("model_name"),
                            "base_url": resolved_base_url,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "status_code": _status_from_exc(e, 429),
                            "action": "rate_limit",
                        }
                        if retry_rate_limits and _should_retry_status(e, retry_codes, default=429):
                            self._mark_failure(model_key, strategy, key_idx, len(keys))
                            continue
                        config_failed = True
                        break
                    except APIConnectionError as e:
                        last_error = e
                        self._logger.warning(
                            "LLM connection error (model=%s, endpoint=%s, attempt=%s): %s",
                            cfg.get("model_name"),
                            resolved_base_url or "openai-default",
                            attempt + 1,
                            str(e),
                        )
                        failure_details = {
                            "config_id": config_id,
                            "model_id": model_id,
                            "model_name": cfg.get("model_name"),
                            "base_url": resolved_base_url,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "status_code": _status_from_exc(e, 503),
                            "action": "connection_error",
                        }
                        if _should_retry_status(e, retry_codes, default=503):
                            self._mark_failure(model_key, strategy, key_idx, len(keys))
                            continue
                        config_failed = True
                        break
                    except APIError as e:
                        last_error = e
                        failure_details = {
                            "config_id": config_id,
                            "model_id": model_id,
                            "model_name": cfg.get("model_name"),
                            "base_url": resolved_base_url,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "status_code": _status_from_exc(e),
                            "action": "api_error",
                        }
                        if _should_retry_status(e, retry_codes):
                            self._mark_failure(model_key, strategy, key_idx, len(keys))
                            continue
                        config_failed = True
                        break
                    except BadRequestError as e:
                        last_error = e
                        failure_details = {
                            "config_id": config_id,
                            "model_id": model_id,
                            "model_name": cfg.get("model_name"),
                            "base_url": resolved_base_url,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "status_code": _status_from_exc(e),
                            "action": "bad_request",
                        }
                        config_failed = True
                        break
                    except Exception as e:  # pragma: no cover
                        last_error = e
                        failure_details = {
                            "config_id": config_id,
                            "model_id": model_id,
                            "model_name": cfg.get("model_name"),
                            "base_url": resolved_base_url,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "status_code": _status_from_exc(e),
                            "action": "unexpected_error",
                        }
                        if attempt >= max_retries_per_model:
                            config_failed = True
                            break

                    if attempt > max_retries_per_model:
                        config_failed = True
                        break
                    sleep_s = min(backoff_max, backoff_base * (2 ** (attempt - 1)))
                    time.sleep(sleep_s)

                if config_failed and failure_details:
                    next_config = chain[chain_idx + 1] if chain_idx + 1 < len(chain) else None
                    failure_details["next_config"] = next_config
                    failure_details["action"] = "fallback" if next_config else "exhausted"
                    _emit_model_event(failure_details)
                    break

        if last_error:
            raise last_error
        raise RuntimeError("LLM generation failed with unknown error and no result")

    def _resolve_config(self, config_id: str, *, model_override: Optional[str] = None) -> Dict[str, Any]:
        """Resolve a config ID to a merged config with model defaults + config overrides.

        Returns a dict with all necessary fields for API calls:
        - model_name: The actual API model string
        - base_url: API endpoint (config overrides model)
        - api_key: API key (config overrides model)
        - default_temp, reasoning_effort, etc.: From config
        - _meta: Model metadata for is_thinking, pricing, etc.
        """
        # Get the config
        config = self.configs_section.get(config_id)
        if not config:
            raise KeyError(f"Unknown config: {config_id}")
        config_enabled = bool(config.get("enabled", True))

        # Get the referenced model
        model_id = model_override or config.get("model")
        if not model_id:
            raise ValueError(f"Config '{config_id}' missing required 'model' field")

        model = self.models_section.get(model_id)
        if not model:
            raise KeyError(f"Config '{config_id}' references unknown model '{model_id}'")
        model_enabled = bool(model.get("enabled", True))

        # Validate model has required fields
        if "model_name" not in model:
            raise ValueError(f"Model '{model_id}' missing required 'model_name' field")
        if "meta" not in model:
            raise ValueError(f"Model '{model_id}' missing required 'meta' field")

        key_strategy = (
            config.get("key_strategy")
            or model.get("key_strategy")
            or "sequential_on_failure"
        )
        max_rpm_default = config.get("max_rpm", model.get("max_rpm"))
        max_rpd_default = config.get("max_rpd", model.get("max_rpd"))
        max_tpm_default = config.get("max_tpm", model.get("max_tpm"))
        max_tpd_default = config.get("max_tpd", model.get("max_tpd"))
        cooldown_default = config.get("cooldown_seconds", model.get("cooldown_seconds"))
        keys_field = config.get("keys") or model.get("keys")
        fallback_key = config.get("api_key") or model.get("api_key")
        normalized_keys = self._normalize_key_entries(
            keys_field=keys_field,
            fallback_key=fallback_key,
            max_rpm_default=max_rpm_default,
            max_rpd_default=max_rpd_default,
            max_tpm_default=max_tpm_default,
            max_tpd_default=max_tpd_default,
            cooldown_default=cooldown_default,
        )

        # Merge: Start with model defaults, then apply config overrides
        resolved = {
            "model_name": model["model_name"],  # Always from model
            "base_url": config.get("base_url") or model.get("base_url"),  # Config overrides model
            "api_key": config.get("api_key") or model.get("api_key"),  # Config overrides model
            "_meta": model["meta"],  # Metadata always from model
            "enabled": config_enabled,
            "_model_enabled": model_enabled,
            "models": config.get("models") or [model_id],
            "key_strategy": key_strategy,
            "_keys": normalized_keys,
        }

        # Add all config-specific fields (default_temp, reasoning_effort, verbosity, etc.)
        for key, value in config.items():
            if key not in (
                "model",
                "models",
                "base_url",
                "api_key",
                "keys",
                "enabled",
                "key_strategy",
                "max_rpm",
                "cooldown_seconds",
            ):
                resolved[key] = value
        # Keep api_key in sync with first entry for compatibility
        if resolved["_keys"]:
            resolved["api_key"] = resolved["_keys"][0]["key"]

        return resolved

    def _build_chain(self, model: Union[str, Sequence[str]]) -> List[str]:
        """
        Build the model attempt chain.

        - For a single model id (string), follow its configured `fallback`
          chain, repeating a config according to `retries_before_fallback`.
        - For an explicit sequence passed by the caller, expand each entry's
          fallback chain in order.
        """
        if isinstance(model, str):
            return self._expand_fallback_chain(model)
        else:
            # Explicit chain provided by caller
            chain: List[str] = []
            for name in model:
                if name not in self.configs_section:
                    raise KeyError(f"Unknown config: {name}")
                chain.extend(self._expand_fallback_chain(name))
            return chain

    def _expand_fallback_chain(self, config_id: str) -> List[str]:
        chain: List[str] = []
        seen: set[str] = set()
        current = config_id

        while current and current not in seen:
            seen.add(current)
            cfg = self.configs_section.get(current)
            if not cfg:
                break

            retries_raw = cfg.get("retries_before_fallback", 0) if isinstance(cfg, dict) else 0
            try:
                retries = int(retries_raw) if retries_raw is not None else 0
            except (TypeError, ValueError):
                retries = 0
            if retries < 0:
                retries = 0
            attempts = retries + 1

            chain.extend([current] * attempts)

            fallback = (cfg.get("fallback") or cfg.get("fallback_preset") or "").strip() if isinstance(cfg, dict) else ""
            if not fallback or fallback in seen:
                break
            current = fallback

        return chain or [config_id]

    def _ensure_key_state(self, model_key: str, num_keys: int) -> Dict[str, Any]:
        state = self._key_state.setdefault(
            model_key,
            {
                "cursor": 0,
                "fail_counts": {},
                "cooldowns": {},
                "rpm": {},
                "rpd": {},
                "tpm": {},
                "tpd": {},
                "header_limits": {},  # New: store limits from headers
            },
        )
        # Ensure deques exist for all keys
        for key in ["rpm", "rpd", "tpm", "tpd"]:
            state_dict: Dict[int, Deque[Any]] = state[key]
            for idx in range(num_keys):
                state_dict.setdefault(idx, deque())
        return state

    def _normalize_key_entries(
        self,
        *,
        keys_field: Optional[Any],
        fallback_key: Optional[str],
        max_rpm_default: Optional[int],
        cooldown_default: Optional[float],
        max_rpd_default: Optional[int] = None,
        max_tpm_default: Optional[int] = None,
        max_tpd_default: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []

        def _coerce_entry(raw: Any) -> Optional[Dict[str, Any]]:
            if isinstance(raw, str):
                return {
                    "key": raw,
                    "max_rpm": max_rpm_default,
                    "max_rpd": max_rpd_default,
                    "max_tpm": max_tpm_default,
                    "max_tpd": max_tpd_default,
                    "cooldown_seconds": cooldown_default,
                }
            if isinstance(raw, dict):
                key_val = raw.get("key") or raw.get("api_key") or raw.get("value")
                if not key_val:
                    return None
                entry = {
                    "key": str(key_val),
                    "max_rpm": raw.get("max_rpm", max_rpm_default),
                    "max_rpd": raw.get("max_rpd", max_rpd_default),
                    "max_tpm": raw.get("max_tpm", max_tpm_default),
                    "max_tpd": raw.get("max_tpd", max_tpd_default),
                    "cooldown_seconds": raw.get("cooldown_seconds", cooldown_default),
                }
                return entry
            return None

        if isinstance(keys_field, list):
            for raw in keys_field:
                coerced = _coerce_entry(raw)
                if coerced:
                    entries.append(coerced)
        elif keys_field is not None:
            coerced = _coerce_entry(keys_field)
            if coerced:
                entries.append(coerced)

        if not entries and fallback_key:
            entries.append(
                {
                    "key": str(fallback_key),
                    "max_rpm": max_rpm_default,
                    "max_rpd": max_rpd_default,
                    "max_tpm": max_tpm_default,
                    "max_tpd": max_tpd_default,
                    "cooldown_seconds": cooldown_default,
                }
            )

        if not entries:
            raise ValueError("No API keys available for this model/config")

        return entries

    def _key_available(
        self,
        model_key: str,
        key_idx: int,
        key_entry: Dict[str, Any],
        now: float,
    ) -> bool:
        state = self._ensure_key_state(model_key, key_idx + 1)
        cooldowns = state["cooldowns"]
        if cooldowns.get(key_idx, 0) > now:
            return False

        # Header-based Limits (Most accurate if available)
        header_limits = state["header_limits"].get(key_idx, {})
        if header_limits:
            # Check requests
            rem_reqs = header_limits.get("remaining_requests")
            reset_reqs = header_limits.get("requests_reset_at", 0)
            if rem_reqs is not None and rem_reqs <= 0 and reset_reqs > now:
                return False
                
            # Check tokens
            rem_tokens = header_limits.get("remaining_tokens")
            reset_tokens = header_limits.get("tokens_reset_at", 0)
            if rem_tokens is not None and rem_tokens <= 0 and reset_tokens > now:
                return False

        # RPM Limit (1 minute)
        rpm_limit = key_entry.get("max_rpm")
        rpm_window = state["rpm"][key_idx]
        while rpm_window and rpm_window[0] < now - 60:
            rpm_window.popleft()
        if isinstance(rpm_limit, int) and rpm_limit > 0 and len(rpm_window) >= rpm_limit:
            return False

        # RPD Limit (24 hours)
        rpd_limit = key_entry.get("max_rpd")
        rpd_window = state["rpd"][key_idx]
        while rpd_window and rpd_window[0] < now - 86400:
            rpd_window.popleft()
        if isinstance(rpd_limit, int) and rpd_limit > 0 and len(rpd_window) >= rpd_limit:
            return False

        # TPM Limit (1 minute)
        tpm_limit = key_entry.get("max_tpm")
        tpm_window = state["tpm"][key_idx]
        while tpm_window and tpm_window[0][0] < now - 60:
            tpm_window.popleft()
        current_tpm = sum(t[1] for t in tpm_window)
        if isinstance(tpm_limit, int) and tpm_limit > 0 and current_tpm >= tpm_limit:
            return False

        # TPD Limit (24 hours)
        tpd_limit = key_entry.get("max_tpd")
        tpd_window = state["tpd"][key_idx]
        while tpd_window and tpd_window[0][0] < now - 86400:
            tpd_window.popleft()
        current_tpd = sum(t[1] for t in tpd_window)
        if isinstance(tpd_limit, int) and tpd_limit > 0 and current_tpd >= tpd_limit:
            return False

        return True

    def _record_usage(self, model_key: str, key_idx: int, timestamp: float) -> None:
        state = self._ensure_key_state(model_key, key_idx + 1)
        state["rpm"][key_idx].append(timestamp)
        state["rpd"][key_idx].append(timestamp)
        # Successful call resets failure counter
        state["fail_counts"][key_idx] = 0

    def _record_token_usage(self, model_key: str, key_idx: int, timestamp: float, tokens: int) -> None:
        if tokens <= 0:
            return
        state = self._ensure_key_state(model_key, key_idx + 1)
        state["tpm"][key_idx].append((timestamp, tokens))
        state["tpd"][key_idx].append((timestamp, tokens))

    def _apply_cooldown(
        self,
        model_key: str,
        key_idx: int,
        *,
        backoff_base: float,
        backoff_max: float,
    ) -> None:
        state = self._ensure_key_state(model_key, key_idx + 1)
        fail_counts = state["fail_counts"]
        fail = int(fail_counts.get(key_idx, 0) or 0) + 1
        fail_counts[key_idx] = fail
        delay = min(backoff_max, backoff_base * (2 ** (fail - 1)))
        state["cooldowns"][key_idx] = time.time() + delay

    def _choose_key_index(
        self,
        model_key: str,
        keys: List[Dict[str, Any]],
        strategy: str,
    ) -> Optional[int]:
        now = time.time()
        state = self._ensure_key_state(model_key, len(keys))

        available = [idx for idx, entry in enumerate(keys) if self._key_available(model_key, idx, entry, now)]
        if not available:
            return None

        total = len(keys)
        if strategy == "random":
            return self._rng.choice(available)

        def _cycled_order(start: int) -> List[int]:
            return [((start + offset) % total) for offset in range(total)]

        if strategy == "round_robin":
            start = state["cursor"] % total
            for idx in _cycled_order(start):
                if idx in available:
                    state["cursor"] = (idx + 1) % total
                    return idx
            return None

        # default: sequential_on_failure
        start = state["cursor"] % total
        for idx in _cycled_order(start):
            if idx in available:
                return idx
        return None

    def _mark_failure(
        self,
        model_key: str,
        strategy: str,
        failed_idx: int,
        num_keys: int,
    ) -> None:
        if strategy != "sequential_on_failure":
            return
        state = self._ensure_key_state(model_key, num_keys)
        state["cursor"] = (failed_idx + 1) % num_keys

    def _parse_duration(self, duration: str) -> float:
        """Parse duration strings like '2m59.56s', '7.66s', '1h2m3s' into seconds."""
        if not duration:
            return 0.0
        try:
            # Try plain float first
            return float(duration)
        except ValueError:
            pass
        
        total = 0.0
        pattern = re.compile(r'(\d+(?:\.\d+)?)([hms])')
        matches = pattern.findall(duration)
        if not matches:
            return 0.0
            
        for val, unit in matches:
            v = float(val)
            if unit == 'h': total += v * 3600
            elif unit == 'm': total += v * 60
            elif unit == 's': total += v
        return total

    def _update_limits_from_headers(self, model_key: str, key_idx: int, headers: Dict[str, str]) -> None:
        state = self._ensure_key_state(model_key, key_idx + 1)
        now = time.time()
        
        # Groq specific headers (and common patterns)
        # RPD (Requests Per Day)
        remaining_reqs = headers.get("x-ratelimit-remaining-requests")
        reset_reqs = headers.get("x-ratelimit-reset-requests")
        
        # TPM (Tokens Per Minute)
        remaining_tokens = headers.get("x-ratelimit-remaining-tokens")
        reset_tokens = headers.get("x-ratelimit-reset-tokens")
        
        limits = state["header_limits"].setdefault(key_idx, {})
        
        if remaining_reqs is not None:
            limits["remaining_requests"] = int(remaining_reqs)
            if reset_reqs:
                limits["requests_reset_at"] = now + self._parse_duration(reset_reqs)
            else:
                # Default to 1s if remaining is 0 but no reset header
                limits["requests_reset_at"] = now + 1.0 if int(remaining_reqs) <= 0 else 0

        if remaining_tokens is not None:
            limits["remaining_tokens"] = int(remaining_tokens)
            if reset_tokens:
                limits["tokens_reset_at"] = now + self._parse_duration(reset_tokens)
            else:
                limits["tokens_reset_at"] = now + 1.0 if int(remaining_tokens) <= 0 else 0

        limits["last_updated"] = now


def _should_retry_status(exc: Exception, retry_codes: set[int], default: Optional[int] = None) -> bool:
    status = None
    try:
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    except Exception:
        status = None
    if status is None:
        status = default
    return status in retry_codes if status is not None else False
