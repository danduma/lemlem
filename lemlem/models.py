import os
from typing import Any, Dict, Optional, Tuple
import json

_YAML_EXTS = {".yaml", ".yml"}


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def _split_meta(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg_copy = dict(cfg)
    meta = cfg_copy.pop("meta", {}) or {}
    if not isinstance(meta, dict):
        meta = {}
    return cfg_copy, meta


def load_models_config(models_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return a deep-copied, env-expanded models config with separate models and configs sections."""
    # Handle structured format only - REQUIRES both models and configs sections
    if "models" not in models_config or "configs" not in models_config:
        raise ValueError("Models config must have both 'models' and 'configs' sections")

    models_section = models_config["models"]
    configs_section = models_config["configs"]

    # Process models section - expand all fields (model_name, base_url, api_key, meta, etc.)
    expanded_models: Dict[str, Dict[str, Any]] = {}
    for name, cfg in models_section.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"Model '{name}' must be a dict")
        cfg_expanded = _expand_env(cfg)
        # Validate required fields
        if "model_name" not in cfg_expanded:
            raise ValueError(f"Model '{name}' missing required 'model_name' field")
        if "meta" not in cfg_expanded:
            raise ValueError(f"Model '{name}' missing required 'meta' field")
        cfg_expanded["enabled"] = bool(cfg_expanded.get("enabled", True))
        expanded_models[name] = cfg_expanded

    # Process configs section - expand all fields
    expanded_configs: Dict[str, Dict[str, Any]] = {}
    for name, cfg in configs_section.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"Config '{name}' must be a dict")
        cfg_expanded = _expand_env(cfg)
        # Validate required field
        if "model" not in cfg_expanded:
            raise ValueError(f"Config '{name}' missing required 'model' field")
        cfg_expanded["enabled"] = bool(cfg_expanded.get("enabled", True))
        models_chain = cfg_expanded.get("models")
        if models_chain is None:
            models_chain = [cfg_expanded["model"]]
        if not isinstance(models_chain, list) or not models_chain:
            raise ValueError(f"Config '{name}' must specify a non-empty 'models' list")
        if not all(isinstance(m, str) and m.strip() for m in models_chain):
            raise ValueError(f"Config '{name}' has invalid entries in 'models'")
        cfg_expanded["models"] = models_chain
        expanded_configs[name] = cfg_expanded

    return {"models": expanded_models, "configs": expanded_configs}


def load_models_file(path: str) -> Dict[str, Dict[str, Any]]:
    """Load a models config from a .json or .yaml/.yml file and expand env vars."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Models config not found: {path}")

    lower = path.lower()
    if lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return load_models_config(data)
    elif any(lower.endswith(ext) for ext in _YAML_EXTS):
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "PyYAML is required to load YAML model configs. Install 'pyyaml'."
            ) from e
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("YAML models config must be a mapping at the top level")
        return load_models_config(data)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unsupported models config extension: {path}")


def load_models_from_env(var_name: str = "MODELS_CONFIG_PATH", default_path: Optional[str] = None, validate_costs: bool = True) -> Dict[str, Dict[str, Any]]:
    """Load models config from env var path or a default path."""
    path = os.environ.get(var_name) or default_path
    if not path:
        raise FileNotFoundError(
            f"No models config path provided; set ${var_name} or pass default_path"
        )

    models_config = load_models_file(path)

    # Validate cost configuration if requested
    if validate_costs:
        from .costs import validate_model_pricing
        errors = validate_model_pricing(models_config)
        if errors:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("Models with missing cost configuration:")
            for model_id, error in errors.items():
                logger.warning(f"  {model_id}: {error}")
            # Don't raise an exception - just log warnings to avoid breaking existing functionality

    return models_config


def get_model_metadata(model_id: str, models_data: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """Get metadata for a specific model by ID."""
    if models_data is None:
        models_data = load_models_from_env()
    if isinstance(models_data, dict) and "models" in models_data:
        return models_data["models"].get(model_id)
    return None


def get_config(config_id: str, models_data: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """Get a deployment config by ID, with model metadata resolved."""
    if models_data is None:
        models_data = load_models_from_env()

    configs = models_data.get("configs", {})
    config = configs.get(config_id)
    if not config:
        return None

    # If config references a model, merge in the model metadata
    if "model" in config:
        model_id = config["model"]
        model_metadata = get_model_metadata(model_id, models_data)
        if model_metadata and "meta" in model_metadata:
            # Create a merged config with model metadata
            merged_config = config.copy()
            merged_config["_meta"] = model_metadata["meta"]
            return merged_config

    return config

