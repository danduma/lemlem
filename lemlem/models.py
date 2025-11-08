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


def load_models_config(models_config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return a deep-copied, env-expanded models config."""
    expanded: Dict[str, Dict[str, Any]] = {}
    for name, cfg in models_config.items():
        cfg_expanded = _expand_env(cfg)
        runtime_cfg, meta = _split_meta(cfg_expanded)
        if meta:
            runtime_cfg["_meta"] = meta
        expanded[name] = runtime_cfg
    return expanded


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

