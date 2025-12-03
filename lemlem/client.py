from __future__ import annotations

import json
import time
from dataclasses import dataclass
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import os

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
            return json.dumps(value)
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
    ) -> LLMResult:
        chain = self._build_chain(model)
        retry_codes = set(retry_on_status)

        last_error: Optional[Exception] = None

        for config_id in chain:
            cfg = self._resolve_config(config_id)
            base_url = cfg.get("base_url")
            api_key = cfg.get("api_key")
            default_temp = cfg.get("default_temp")

            temp = temperature if temperature is not None else default_temp

            # Defensive: treat unresolved env placeholders like "${VAR}" or empty strings as missing
            def _sanitize(value: Optional[str]) -> Optional[str]:
                if not value:
                    return None
                s = str(value).strip()
                if s.startswith("${") and s.endswith("}"):
                    return None
                return s

            resolved_base_url = _sanitize(base_url)
            resolved_api_key = _sanitize(api_key)

            if not resolved_api_key:
                # Surface a clear error rather than a vague connection/auth error
                raise RuntimeError(f"Missing API key for config '{config_id}' (model_name={cfg['model_name']})")

            client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url) if resolved_base_url else OpenAI(api_key=resolved_api_key)
            # Use Responses API for reasoning models (o1, thinking models) on OpenAI
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

            attempt = 0
            while True:
                try:
                    if use_responses:
                        # Responses API for reasoning models (e.g., o1). Use `input`, not `messages`.
                        payload: Dict[str, Any] = {
                            "model": cfg["model_name"],
                            "input": input_text,
                        }
                        extra_payload = dict(extra or {})
                        max_completion_tokens = extra_payload.pop("max_completion_tokens", None)
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
                        if max_completion_tokens is not None:
                            payload["max_completion_tokens"] = max_completion_tokens
                            payload.setdefault("max_output_tokens", max_completion_tokens)

                        payload.update(extra_payload)
                        if "tools" not in payload:
                            payload["tools"] = []
                        if include_tools is not None:
                            payload["tools"] = include_tools
                        if include_tool_choice is not None:
                            payload["tool_choice"] = include_tool_choice

                        resp = client.responses.create(**payload)
                        text = _extract_responses_output_text(resp)
                        return LLMResult(
                            text=text or "",
                            model_used=cfg["model_name"],
                            provider="openai-responses",
                            raw=resp,
                        )
                    else:
                        payload = {
                            "model": cfg["model_name"],
                            "messages": chat_messages,
                        }
                        extra_payload = dict(extra or {})
                        usage_payload = extra_payload.pop("usage", None)
                        max_completion_tokens = extra_payload.pop("max_completion_tokens", None)
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
                        if max_completion_tokens is not None:
                            payload["max_completion_tokens"] = max_completion_tokens
                        if usage_payload is not None:
                            payload["usage"] = usage_payload
                        # OpenRouter: Enable usage accounting (includes exact cost in final chunk)
                        if resolved_base_url and "openrouter.ai" in resolved_base_url:
                            usage_cfg = payload.get("usage")
                            if not isinstance(usage_cfg, dict):
                                usage_cfg = {} if usage_cfg is None else {"value": usage_cfg}
                            usage_cfg["include"] = True
                            payload["usage"] = usage_cfg
                        resp = client.chat.completions.create(**payload)
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
                        return LLMResult(
                            text=text or "",
                            model_used=cfg["model_name"],
                            provider="openai-compatible",
                            raw=resp,
                        )
                except RateLimitError as e:
                    last_error = e
                    retry_rate_limits = bool(meta.get("retry_on_rate_limit"))
                    if not retry_rate_limits or not _should_retry_status(e, retry_codes, default=429):
                        break
                except APIConnectionError as e:
                    last_error = e
                    # Helpful trace for connection-class failures
                    self._logger.warning(
                        "LLM connection error (model=%s, endpoint=%s, attempt=%s): %s",
                        model_name,
                        resolved_base_url or "openai-default",
                        attempt + 1,
                        str(e),
                    )
                    if not _should_retry_status(e, retry_codes, default=503):
                        break
                except APIError as e:
                    last_error = e
                    if not _should_retry_status(e, retry_codes):
                        break
                except BadRequestError as e:
                    last_error = e
                    break
                except Exception as e:  # pragma: no cover
                    last_error = e
                    if attempt >= max_retries_per_model:
                        break

                attempt += 1
                if attempt > max_retries_per_model:
                    break
                sleep_s = min(backoff_max, backoff_base * (2 ** (attempt - 1)))
                time.sleep(sleep_s)

            continue

        if last_error:
            raise last_error
        raise RuntimeError("LLM generation failed with unknown error and no result")

    def _resolve_config(self, config_id: str) -> Dict[str, Any]:
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

        # Get the referenced model
        model_id = config.get("model")
        if not model_id:
            raise ValueError(f"Config '{config_id}' missing required 'model' field")

        model = self.models_section.get(model_id)
        if not model:
            raise KeyError(f"Config '{config_id}' references unknown model '{model_id}'")

        # Validate model has required fields
        if "model_name" not in model:
            raise ValueError(f"Model '{model_id}' missing required 'model_name' field")
        if "meta" not in model:
            raise ValueError(f"Model '{model_id}' missing required 'meta' field")

        # Merge: Start with model defaults, then apply config overrides
        resolved = {
            "model_name": model["model_name"],  # Always from model
            "base_url": config.get("base_url") or model.get("base_url"),  # Config overrides model
            "api_key": config.get("api_key") or model.get("api_key"),  # Config overrides model
            "_meta": model["meta"],  # Metadata always from model
        }

        # Add all config-specific fields (default_temp, reasoning_effort, verbosity, etc.)
        for key, value in config.items():
            if key not in ("model", "base_url", "api_key"):  # Skip already-processed fields
                resolved[key] = value

        return resolved

    def _build_chain(self, model: Union[str, Sequence[str]]) -> List[str]:
        """
        Build the model attempt chain.

        We only support explicit fallback sequences passed by the caller
        (e.g., model=["primary", "backup1", "backup2"]).

        Any `fallback` keys present inside model configs are ignored, so that
        routing behavior is fully controlled by call sites.
        """
        if isinstance(model, str):
            # Single model: no implicit fallbacks from config
            self._resolve_config(model)  # validate existence
            return [model]
        else:
            # Explicit chain provided by caller
            chain: List[str] = []
            for name in model:
                self._resolve_config(name)  # validate each exists
                if name not in chain:
                    chain.append(name)
            return chain


def _should_retry_status(exc: Exception, retry_codes: set[int], default: Optional[int] = None) -> bool:
    status = None
    try:
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    except Exception:
        status = None
    if status is None:
        status = default
    return status in retry_codes if status is not None else False
