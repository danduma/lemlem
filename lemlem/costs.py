"""
Cost computation utilities for LLM usage tracking.

This module provides cost calculation functions for different LLM providers,
supporting both fresh and cached token pricing.
"""

import logging
import requests
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

    def _get_field(source: Any, key: str) -> Optional[Any]:
        """Get an attribute or dict entry safely."""
        if source is None:
            return None
        if isinstance(source, dict):
            return source.get(key)
        return getattr(source, key, None)

    def _cached_from_source(source: Any) -> Optional[int]:
        if source is None:
            return None
        if isinstance(source, dict):
            raw_value = source.get('cached_tokens')
        else:
            raw_value = getattr(source, 'cached_tokens', None)
        if raw_value is None:
            return None
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return None

    try:
        # OpenAI/OpenRouter: Check multiple locations for cached tokens
        # - usage.cached_tokens (direct)
        # - usage.input_tokens_details.cached_tokens (OpenAI Responses API)
        # - usage.prompt_tokens_details.cached_tokens (OpenRouter with usage accounting)
        if provider.lower() in ['openai', 'openrouter', 'openai-compatible', 'openai-responses']:
            usage = getattr(result_raw, "usage", None)
            if usage:
                # Direct cached_tokens field
                cached = _cached_from_source(usage)
                if cached is not None:
                    return cached
                # OpenAI Responses API: input_tokens_details.cached_tokens
                details = _get_field(usage, 'input_tokens_details')
                cached = _cached_from_source(details)
                if cached is not None:
                    return cached
                # OpenRouter: prompt_tokens_details.cached_tokens
                details = _get_field(usage, 'prompt_tokens_details')
                cached = _cached_from_source(details)
                if cached is not None:
                    return cached

        # Gemini: usage_metadata.cachedContentTokenCount
        elif provider.lower() == 'google':
            usage_metadata = getattr(result_raw, "usage_metadata", {})
            if usage_metadata:
                return usage_metadata.get('cachedContentTokenCount', 0)

        # GLM/Zhipu: usage.prompt_tokens_details.cached_tokens
        elif provider.lower() in ['glm', 'zhipu']:
            usage = getattr(result_raw, "usage", None)
            if usage:
                details = _get_field(usage, 'prompt_tokens_details')
                cached = _cached_from_source(details)
                if cached is not None:
                    return cached

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

    # Ensure cached pricing exists, defaulting to zero-cost cache hits unless overridden
    if cost_cached_in is None:
        cost_cached_in = 0.0

    # Calculate fresh tokens (non-cached portion) and guard against negatives
    cached_tokens = max(0, min(cached_tokens, prompt_tokens))
    fresh_tokens = max(prompt_tokens - cached_tokens, 0)

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


def fetch_openrouter_generation_cost(generation_id: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Fetch final cost and usage data for an OpenRouter generation.

    Args:
        generation_id: The OpenRouter generation ID
        api_key: OpenRouter API key

    Returns:
        Dict containing final cost data, or None if failed
    """
    try:
        url = "https://openrouter.ai/api/v1/generation"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        params = {"id": generation_id}

        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        if "data" not in data:
            logger.warning(f"Unexpected OpenRouter response format for generation {generation_id}")
            return None

        generation_data = data["data"]

        # Extract relevant cost information
        result = {
            "generation_id": generation_id,
            "total_cost": generation_data.get("total_cost"),
            "cache_discount": generation_data.get("cache_discount"),
            "upstream_inference_cost": generation_data.get("upstream_inference_cost"),
            "tokens_prompt": generation_data.get("tokens_prompt"),
            "tokens_completion": generation_data.get("tokens_completion"),
            "native_tokens_prompt": generation_data.get("native_tokens_prompt"),
            "native_tokens_completion": generation_data.get("native_tokens_completion"),
            "native_tokens_cached": generation_data.get("native_tokens_cached"),
            "latency": generation_data.get("latency"),
            "generation_time": generation_data.get("generation_time"),
            "finish_reason": generation_data.get("finish_reason"),
            "provider_name": generation_data.get("provider_name"),
            "model": generation_data.get("model"),
            "created_at": generation_data.get("created_at"),
        }

        # Remove None values for cleaner data
        result = {k: v for k, v in result.items() if v is not None}

        logger.debug(f"Successfully fetched final cost data for generation {generation_id}")
        return result

    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch OpenRouter generation cost for {generation_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching OpenRouter generation cost for {generation_id}: {e}")
        return None
