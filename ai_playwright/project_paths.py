from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
TEMPLATE_ROOT = PACKAGE_ROOT / "templates"
CONFIG_DIR_ENV = "AI_PLAYWRIGHT_CONFIG_DIR"
DEMO_TEMPLATE_PATH = Path("test_data") / "demo"


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

    packaged_path = _resolve_packaged_relative_path(path)
    if packaged_path:
        return packaged_path

    return cwd_path


def demo_template_root() -> Path:
    """Return the single packaged/source demo seed used by init_project."""
    for candidate in _demo_template_candidates():
        if candidate.exists():
            return candidate
    return _demo_template_candidates()[0]


def is_packaged_template_path(path: str | Path) -> bool:
    resolved = Path(path).resolve()
    if _is_relative_to(resolved, (Path.cwd() / "test_data").resolve()):
        return False
    read_only_roots = [
        TEMPLATE_ROOT.resolve(),
        (PROJECT_ROOT / "test_data").resolve(),
    ]
    return any(_is_relative_to(resolved, root) for root in read_only_roots)


def _resolve_packaged_relative_path(path: Path) -> Path | None:
    for root in (PROJECT_ROOT, TEMPLATE_ROOT):
        candidate = root / path
        if candidate.exists():
            return candidate
    return None


def _demo_template_candidates() -> list[Path]:
    return [
        PROJECT_ROOT / DEMO_TEMPLATE_PATH,
        TEMPLATE_ROOT / DEMO_TEMPLATE_PATH,
    ]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
