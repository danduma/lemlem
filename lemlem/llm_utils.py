"""
LLM utilities and guardrails.

Note on "thinking" models:
- Some reasoning/thinking models (e.g., OpenAI o3 family) only support the default
  temperature of 1. Passing any other temperature (e.g., 0.3) will error with
  unsupported_value. To avoid this class of failures, use `coerce_thinking_temperature`.
"""
from __future__ import annotations

import ast
import re
from typing import Any, Dict, Optional


def is_thinking_model(model: str) -> bool:
    """Heuristic: detect models that only support temperature=1.

    Adjust this list/logic as your providers evolve. We intentionally keep it
    conservative to avoid breaking calls accidentally.
    """
    if not model:
        return False

    model_id = model.strip()

    # 1. Check metadata from the loaded model configs (most reliable source)
    try:
        import lemlem.adapter as lemlem_adapter
        lemlem_adapter._refresh_model_data()
        model_data = lemlem_adapter.MODEL_DATA
        configs = model_data.get("configs", {}) if isinstance(model_data, dict) else {}
        models = model_data.get("models", {}) if isinstance(model_data, dict) else {}

        def _meta_says_thinking(meta_candidate: Any) -> bool:
            return bool(isinstance(meta_candidate, dict) and meta_candidate.get("is_thinking"))

        cfg = configs.get(model_id)
        if isinstance(cfg, dict):
            if _meta_says_thinking(cfg.get("_meta")):
                return True
            target_model = cfg.get("model")
            if isinstance(target_model, str):
                model_meta = models.get(target_model, {}).get("meta")
                if _meta_says_thinking(model_meta):
                    return True

        # Sometimes callers pass the raw model identifier from the models section
        direct_meta = models.get(model_id, {}).get("meta")
        if _meta_says_thinking(direct_meta):
            return True
    except Exception:  # pragma: no cover - guard against malformed configs without breaking runtime
        pass


    m = model_id.lower()
    if 'thinking' in m or 'reason' in m or 'deepresearch' in m:
        return True
    return False


def coerce_thinking_temperature(model: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure allowed temperature for thinking models.

    - For thinking models: force temperature=1 (or remove custom value).
    - For others: keep the provided temperature.
    Returns a shallow-copied params dict.
    """
    p = dict(params or {})
    if is_thinking_model(model):
        # Some SDKs require omitting temperature entirely; leaving explicit 1 is safe
        # enough for most. If you hit issues, comment the next line and rely on SDK default.
        p['temperature'] = 1
    return p


def _parse_duration_to_seconds(value: str) -> Optional[float]:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip().lower()
    try:
        return float(raw)
    except ValueError:
        pass

    pattern = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")
    matches = pattern.findall(raw)
    if not matches:
        return None
    total = 0.0
    for amount, unit in matches:
        try:
            val = float(amount)
        except ValueError:
            continue
        if unit == "ms":
            total += val / 1000.0
        elif unit == "s":
            total += val
        elif unit == "m":
            total += val * 60.0
        elif unit == "h":
            total += val * 3600.0
    return total if total > 0 else None


def _extract_retry_delay_from_details(details: Any) -> Optional[float]:
    if not isinstance(details, list):
        return None
    for detail in details:
        if isinstance(detail, dict):
            detail_type = str(detail.get("@type") or "")
            if detail_type.endswith("RetryInfo") and "retryDelay" in detail:
                delay_val = detail.get("retryDelay")
                if isinstance(delay_val, (int, float)):
                    return float(delay_val)
                if isinstance(delay_val, str):
                    parsed = _parse_duration_to_seconds(delay_val)
                    if parsed is not None:
                        return parsed
        else:
            retry_delay = getattr(detail, "retry_delay", None)
            if retry_delay is not None:
                if isinstance(retry_delay, (int, float)):
                    return float(retry_delay)
                if isinstance(retry_delay, str):
                    parsed = _parse_duration_to_seconds(retry_delay)
                    if parsed is not None:
                        return parsed
    return None


def extract_retry_after_seconds(error: Any) -> Optional[float]:
    if error is None:
        return None

    if not isinstance(error, str):
        for attr in (
            "retry_after_seconds",
            "retry_after",
            "retry_delay",
            "retry_delay_seconds",
        ):
            val = getattr(error, attr, None)
            if val is None:
                continue
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                parsed = _parse_duration_to_seconds(val)
                if parsed is not None:
                    return parsed

        details = getattr(error, "details", None)
        parsed = _extract_retry_delay_from_details(details)
        if parsed is not None:
            return parsed
        message = str(error)
    else:
        message = error

    if not message:
        return None

    lower_message = message.lower()

    retry_match = re.search(
        r"retry\s+(?:in|after)\s+([0-9]+(?:\.[0-9]+)?)\s*(ms|s|m|h)?",
        lower_message,
    )
    if retry_match:
        number = retry_match.group(1)
        unit = retry_match.group(2) or "s"
        parsed = _parse_duration_to_seconds(f"{number}{unit}")
        if parsed is not None:
            return parsed

    retry_delay_match = re.search(
        r"retrydelay\s*[:=]\s*['\"]?([0-9]+(?:\.[0-9]+)?)(ms|s|m|h)?",
        lower_message,
    )
    if retry_delay_match:
        number = retry_delay_match.group(1)
        unit = retry_delay_match.group(2) or "s"
        parsed = _parse_duration_to_seconds(f"{number}{unit}")
        if parsed is not None:
            return parsed

    if "{" in message and "}" in message:
        payload_start = message.find("{")
        if payload_start != -1:
            payload_str = message[payload_start:]
            try:
                payload = ast.literal_eval(payload_str)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                error_section = payload.get("error")
                if isinstance(error_section, dict):
                    parsed = _extract_retry_delay_from_details(error_section.get("details"))
                    if parsed is not None:
                        return parsed
                    err_message = error_section.get("message")
                    if isinstance(err_message, str):
                        parsed = extract_retry_after_seconds(err_message)
                        if parsed is not None:
                            return parsed
                parsed = _extract_retry_delay_from_details(payload.get("details"))
                if parsed is not None:
                    return parsed

    return None


# Developer reminder: thinking models use ONLY temperature=1
# If you add new reasoning-capable models, extend `is_thinking_model` above.
