"""
LLM utilities and guardrails.

Note on "thinking" models:
- Some reasoning/thinking models (e.g., OpenAI o3 family) only support the default
  temperature of 1. Passing any other temperature (e.g., 0.3) will error with
  unsupported_value. To avoid this class of failures, use `coerce_thinking_temperature`.
"""
from __future__ import annotations

from typing import Any, Dict

import lemlem.adapter as lemlem_adapter


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


# Developer reminder: thinking models use ONLY temperature=1
# If you add new reasoning-capable models, extend `is_thinking_model` above.




