from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class JSONPayloadParseError(ValueError):
    """Raised when an LLM payload cannot be coerced into JSON."""


_JSON_OBJECT_PATTERN = re.compile(r"\\{.*\\}", re.DOTALL)


def parse_json_payload_best_effort(text_payload: Optional[str]) -> Dict[str, Any]:
    """
    Best-effort parsing of JSON payloads returned by a model.

    Handles common patterns:
    - ```json fenced blocks
    - JSON embedded in prose (extract longest {...} candidate)
    - Trailing content after a valid JSON object (raw_decode)
    """
    candidate = (text_payload or "{}").strip()
    if not candidate:
        return {}

    # Extract JSON from common fenced code block formats.
    if candidate.startswith("```"):
        fence_end = candidate.rfind("```")
        if fence_end > 0:
            inner = candidate[3:fence_end].strip()
            newline_idx = inner.find("\n")
            if newline_idx != -1:
                possible_lang = inner[:newline_idx].strip().lower()
                if possible_lang in {"json", "```json"}:
                    inner = inner[newline_idx + 1 :]
            candidate = inner.strip()

    original_candidate = candidate
    candidate = candidate.strip()

    matches = _JSON_OBJECT_PATTERN.findall(candidate)
    if matches:
        for match in sorted(matches, key=len, reverse=True):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    try:
        return json.loads(candidate or "{}")
    except json.JSONDecodeError:
        try:
            decoder = json.JSONDecoder()
            obj, idx = decoder.raw_decode(candidate)
            trailing = candidate[idx:].strip()
            if trailing:
                logger.warning(
                    "LLM JSON payload had trailing content | trailing_preview=%s",
                    trailing[:120],
                )
            if isinstance(obj, dict):
                return obj
            return {"value": obj}
        except Exception as exc:
            raise JSONPayloadParseError(
                f"LLM did not return valid JSON: text={original_candidate[:200]}"
            ) from exc
