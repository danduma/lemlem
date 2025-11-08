"""
LLM utilities and guardrails.

Note on "thinking" models:
- Some reasoning/thinking models (e.g., OpenAI o3 family) only support the default
  temperature of 1. Passing any other temperature (e.g., 0.3) will error with
  unsupported_value. To avoid this class of failures, use `coerce_thinking_temperature`.
"""
from __future__ import annotations

from typing import Dict, Any


def is_thinking_model(model: str) -> bool:
    """Heuristic: detect models that only support temperature=1.

    Adjust this list/logic as your providers evolve. We intentionally keep it
    conservative to avoid breaking calls accidentally.
    """
    m = (model or '').lower().strip()
    if not m:
        return False
    # OpenAI o3 family (reasoning)
    if m.startswith('o3'):
        return True
    # Common suffix/patterns some vendors use
    if 'thinking' in m or 'reason' in m:
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


# Developer reminder: thinking models use ONLY temperature=1
# If you add new reasoning-capable models, extend `is_thinking_model` above.







