from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ai_playwright.project_paths import resolve_config_file
from ai_playwright.utils.yaml_handler import YamlHandler


DEFAULT_AI_CONFIG_PATH = "ai_config.yaml"
LLM_DATA_POLICY_ENV = "LLM_DATA_POLICY"
LLM_DATA_POLICY_EXTERNAL = "external"
LLM_DATA_POLICY_TRUSTED_LOCAL = "trusted_local"


def load_ai_config(path: str | Path = DEFAULT_AI_CONFIG_PATH) -> dict[str, Any]:
    config_path = (
        resolve_config_file(str(path))
        if str(path) == str(DEFAULT_AI_CONFIG_PATH)
        else Path(path)
    )
    if not config_path.exists():
        return {}
    return YamlHandler().load_yaml(config_path) or {}


def runtime_mode(default_mode: str = "strict") -> str:
    mode = os.environ.get("UI_AI_MODE") or default_mode
    return str(mode).lower()


def normalize_llm_data_policy(
    value: Any, *, default: str = LLM_DATA_POLICY_EXTERNAL
) -> str:
    policy = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "local": LLM_DATA_POLICY_TRUSTED_LOCAL,
        "private": LLM_DATA_POLICY_TRUSTED_LOCAL,
        "trusted": LLM_DATA_POLICY_TRUSTED_LOCAL,
        LLM_DATA_POLICY_TRUSTED_LOCAL: LLM_DATA_POLICY_TRUSTED_LOCAL,
        "remote": LLM_DATA_POLICY_EXTERNAL,
        "hosted": LLM_DATA_POLICY_EXTERNAL,
        "third_party": LLM_DATA_POLICY_EXTERNAL,
        LLM_DATA_POLICY_EXTERNAL: LLM_DATA_POLICY_EXTERNAL,
    }
    return aliases.get(policy, default)


def llm_data_policy(
    config: dict[str, Any] | None = None,
    *,
    default: str = LLM_DATA_POLICY_EXTERNAL,
) -> str:
    env_policy = os.environ.get(LLM_DATA_POLICY_ENV)
    if env_policy:
        return normalize_llm_data_policy(env_policy, default=default)
    data_boundary = (config or {}).get("data_boundary")
    if isinstance(data_boundary, dict):
        return normalize_llm_data_policy(data_boundary.get("policy"), default=default)
    return default
