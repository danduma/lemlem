from __future__ import annotations

import time
from dataclasses import dataclass
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from openai import OpenAI
from openai._exceptions import (
    APIConnectionError,
    APIError,
    BadRequestError,
    RateLimitError,
)


Messages = List[Dict[str, str]]  # [{"role": "user"|"system"|"assistant", "content": str}]


@dataclass
class LLMResult:
    text: str
    model_used: str
    provider: str
    raw: Any


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
    def __init__(self, models_config: Dict[str, Dict[str, Any]]):
        self.models_config = models_config
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

        for model_name in chain:
            cfg = self._get_cfg(model_name)
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
                raise RuntimeError(f"Missing API key for model '{model_name}' (model_name={cfg.get('model_name')})")

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
                            "model": cfg.get("model_name", model_name),
                            "input": input_text,
                        }
                        extra_payload = dict(extra or {})
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
                        include_tool_choice = extra_payload.pop("tool_choice", None)

                        # Note: temperature is not supported in Responses API for reasoning models
                        structured_schema = extra_payload.pop("structured_schema", None)
                        structured_name = extra_payload.pop("structured_name", "response")
                        if structured_schema:
                            payload["response_format"] = {
                                "type": "json_schema",
                                "json_schema": {
                                    "name": structured_name or "response",
                                    "schema": structured_schema,
                                    "strict": True,
                                },
                            }
                        payload.update(extra_payload)
                        if "tools" not in payload:
                            payload["tools"] = []
                        if include_tools is not None:
                            payload["tools"] = include_tools
                        if include_tool_choice is not None:
                            payload["tool_choice"] = include_tool_choice
                        resp = client.responses.create(**payload)
                        # Prefer SDK's normalized attribute when present
                        text = getattr(resp, "output_text", None) or (
                            getattr(getattr(resp, "body", None), "content", "") or ""
                        )
                        return LLMResult(
                            text=text or "",
                            model_used=cfg.get("model_name", model_name),
                            provider="openai-responses",
                            raw=resp,
                        )
                    else:
                        payload = {
                            "model": cfg.get("model_name", model_name),
                            "messages": chat_messages,
                        }
                        extra_payload = dict(extra or {})
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
                            payload.update(extra_payload)
                        resp = client.chat.completions.create(**payload)
                        # If function/tool call was used, prefer arguments as the text
                        text = ""
                        if resp.choices:
                            msg = resp.choices[0].message
                            tc = getattr(msg, "tool_calls", None)
                            if tc:
                                try:
                                    text = tc[0].function.arguments or ""
                                except Exception:
                                    text = msg.content or ""
                            else:
                                text = msg.content or ""
                        return LLMResult(
                            text=text or "",
                            model_used=cfg.get("model_name", model_name),
                            provider="openai-compatible",
                            raw=resp,
                        )
                except RateLimitError as e:
                    last_error = e
                    if not _should_retry_status(e, retry_codes, default=429):
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

    def _get_cfg(self, model_name: str) -> Dict[str, Any]:
        if model_name not in self.models_config:
            raise KeyError(f"Unknown model: {model_name}")
        return self.models_config[model_name]

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
            self._get_cfg(model)  # validate existence
            return [model]
        else:
            # Explicit chain provided by caller
            chain: List[str] = []
            for name in model:
                self._get_cfg(name)  # validate each exists
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
