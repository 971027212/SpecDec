from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when an experiment config is missing or malformed."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    if config_path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    elif config_path.suffix == ".json":
        data = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        raise ConfigError(f"Unsupported config suffix: {config_path.suffix}")

    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {config_path}")
    return data


def get_in(config: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def as_list(value: Any, *, item_type: type | None = None) -> list[Any]:
    if value is None:
        items: list[Any] = []
    elif isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    else:
        items = [value]

    if item_type is not None:
        return [item_type(item) for item in items]
    return items

