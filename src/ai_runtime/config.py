from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from utils.yaml_handler import YamlHandler


DEFAULT_AI_CONFIG_PATH = Path("config/ai_config.yaml")


def load_ai_config(path: str | Path = DEFAULT_AI_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return YamlHandler().load_yaml(config_path) or {}


def runtime_mode(default_mode: str = "strict") -> str:
    mode = os.environ.get("UI_AI_MODE") or default_mode
    return str(mode).lower()


def is_ai_mode(mode: str) -> bool:
    return str(mode).lower() in {"smart", "ai"}
