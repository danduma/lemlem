"""
Cost computation utilities for LLM usage tracking.

This module provides cost calculation functions for different LLM providers,
supporting both fresh and cached token pricing.
"""

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def load_model_configs() -> Dict[str, Dict[str, Any]]:
    """Load model configurations from environment."""
    try:
        from .models import load_models_from_env
        return load_models_from_env()
    except Exception as e:
        logger.warning(f"Failed to load model configs: {e}")
        return {}


def extract_cached_tokens(result_raw: Any, provider: str) -> int:
    """Extract cached token count from LLM response based on provider.

    Args:
        result_raw: The raw response object from the LLM API
        provider: The API provider (e.g., 'openai', 'google', 'openrouter')

    Returns:
        Number of cached tokens, or 0 if not available
    """
    try:
        # OpenAI: usage.cached_tokens
        if provider.lower() in ['openai', 'openrouter']:
            usage = getattr(result_raw, "usage", None)
            if usage:
                return getattr(usage, 'cached_tokens', 0)

        # Gemini: usage_metadata.cachedContentTokenCount
        elif provider.lower() == 'google':
            usage_metadata = getattr(result_raw, "usage_metadata", {})
            if usage_metadata:
                return usage_metadata.get('cachedContentTokenCount', 0)

        # GLM/Zhipu: usage.prompt_tokens_details.cached_tokens
        elif provider.lower() in ['glm', 'zhipu']:
            usage = getattr(result_raw, "usage", None)
            if usage and hasattr(usage, 'prompt_tokens_details'):
                details = getattr(usage, 'prompt_tokens_details', {})
                return getattr(details, 'cached_tokens', 0)

    except (AttributeError, TypeError):
        # Gracefully handle parsing errors
        pass

    return 0


def _resolve_model_entry(models_section: Dict[str, Dict[str, Any]], identifier: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Return the canonical model key & data for either a logical ID or provider model_name."""
    model_data = models_section.get(identifier)
    if isinstance(model_data, dict):
        return identifier, model_data

    for key, data in models_section.items():
        if isinstance(data, dict) and data.get("model_name") == identifier:
            return key, data

    return None, None


def compute_cost_for_model(
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    model_configs: Optional[Dict[str, Dict[str, Any]]] = None
) -> float:
    """Compute cost for a model based on token usage.

    Args:
        model_id: The model identifier
        prompt_tokens: Total prompt/input tokens used
        completion_tokens: Completion/output tokens used
        cached_tokens: Number of cached input tokens (default: 0)
        model_configs: Model configuration dict (loaded automatically if None)

    Returns:
        Total cost in USD

    Raises:
        ValueError: If model config or pricing is missing
    """
    if model_configs is None:
        model_configs = load_model_configs()

    # NEW FORMAT ONLY: Requires structured {"models": {...}, "configs": {...}}
    models_section = model_configs.get("models", {})
    configs_section = model_configs.get("configs", {})

    if not models_section or not configs_section:
        raise ValueError("Model configs must have both 'models' and 'configs' sections")

    # First try to find as a config ID
    cfg = configs_section.get(model_id)
    if cfg:
        # Config must reference a model
        model_ref = cfg.get("model")
        if not model_ref:
            raise ValueError(f"Config '{model_id}' missing required 'model' field")

        # Get pricing from the referenced model
        resolved_model_id, model_data = _resolve_model_entry(models_section, model_ref)
        if not model_data:
            raise ValueError(f"Config '{model_id}' references unknown model '{model_ref}'")

        meta = model_data.get("meta", {})
        if not meta:
            missing_id = resolved_model_id or model_ref
            raise ValueError(f"Model '{missing_id}' missing required 'meta' field")
    else:
        # Try to find as a direct model ID
        resolved_model_id, model_data = _resolve_model_entry(models_section, model_id)
        if not model_data:
            raise ValueError(f"No config or model found with ID '{model_id}'")

        meta = model_data.get("meta", {})
        if not meta:
            missing_id = resolved_model_id or model_id
            raise ValueError(f"Model '{missing_id}' missing required 'meta' field")

    cost_in = meta.get("cost_per_1m_input_tokens")
    cost_cached_in = meta.get("cost_per_1m_cached_input")
    cost_out = meta.get("cost_per_1m_output_tokens")

    if cost_in is None or cost_out is None:
        raise ValueError(f"Missing pricing configuration for model '{model_id}'. Required: cost_per_1m_input_tokens and cost_per_1m_output_tokens")

    # Ensure cached pricing exists, fallback to regular pricing if not
    if cost_cached_in is None:
        cost_cached_in = cost_in

    # Calculate fresh tokens (non-cached portion)
    fresh_tokens = prompt_tokens - cached_tokens

    return (
        (fresh_tokens / 1_000_000.0) * float(cost_in) +
        (cached_tokens / 1_000_000.0) * float(cost_cached_in) +
        (completion_tokens / 1_000_000.0) * float(cost_out)
    )


def validate_model_pricing(model_configs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, str]:
    """Validate that all models have required pricing configuration.

    Args:
        model_configs: Model configuration dict (loaded automatically if None)

    Returns:
        Dict of model_id -> error_message for models with missing pricing
    """
    if model_configs is None:
        model_configs = load_model_configs()

    errors = {}

    # Handle structured format only - validate models section
    models_section = model_configs.get("models", {})
    for model_id, model_data in models_section.items():
        meta = model_data.get("meta", {})
        cost_in = meta.get("cost_per_1m_input_tokens")
        cost_out = meta.get("cost_per_1m_output_tokens")

        if cost_in is None:
            errors[model_id] = "Missing cost_per_1m_input_tokens"
        if cost_out is None:
            errors[model_id] = "Missing cost_per_1m_output_tokens"

    return errors
