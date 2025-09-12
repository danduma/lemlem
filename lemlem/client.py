from __future__ import annotations

import time
from dataclasses import dataclass
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

            client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            # Use Responses API for reasoning models (o1, thinking models) on OpenAI
            is_thinking_model = cfg.get("thinking", False)
            is_openai_endpoint = _is_openai_base_url(base_url)
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
                        # Note: temperature is not supported in Responses API for reasoning models
                        if extra:
                            payload.update(extra)
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
                        if temp is not None:
                            payload["temperature"] = temp
                        if extra:
                            payload.update(extra)
                        resp = client.chat.completions.create(**payload)
                        text = resp.choices[0].message.content if resp.choices else ""
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

