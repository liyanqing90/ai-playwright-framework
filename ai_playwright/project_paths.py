from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
TEMPLATE_ROOT = PACKAGE_ROOT / "templates"
CONFIG_DIR_ENV = "AI_PLAYWRIGHT_CONFIG_DIR"


def resolve_config_file(file_name: str) -> Path:
    configured_dir = os.environ.get(CONFIG_DIR_ENV)
    if configured_dir:
        return Path(configured_dir).expanduser().resolve() / file_name

    cwd_path = Path.cwd() / "config" / file_name
    if cwd_path.exists():
        return cwd_path

    template_path = TEMPLATE_ROOT / "config" / file_name
    if template_path.exists():
        return template_path

    return cwd_path


def resolve_project_test_dir(project_config: dict[str, Any], project: str) -> Path:
    raw_test_dir = str(project_config.get("test_dir") or f"test_data/{project}")
    path = Path(raw_test_dir).expanduser()
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    template_path = TEMPLATE_ROOT / path
    if template_path.exists():
        return template_path

    return cwd_path


def is_packaged_template_path(path: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(TEMPLATE_ROOT.resolve())
    except ValueError:
        return False
    return True
