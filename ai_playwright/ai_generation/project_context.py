from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_playwright.project_paths import resolve_config_file, resolve_project_test_dir
from ai_playwright.utils.yaml_handler import YamlHandler


@dataclass(frozen=True)
class ProjectContext:
    project: str
    test_dir: Path
    base_url: str
    elements: dict[str, Any]
    modules: dict[str, Any]
    variables: dict[str, Any]
    test_cases: list[dict[str, Any]]
    test_data: dict[str, Any]


def load_project_context(project: str, env: str = "prod") -> ProjectContext:
    yaml = YamlHandler()
    env_config = yaml.load_yaml(resolve_config_file("env_config.yaml"))
    project_cfg = (env_config.get("projects") or {}).get(project)
    if not project_cfg:
        raise ValueError(f"项目不存在: {project}")

    test_dir = resolve_project_test_dir(project_cfg, project)
    environments = project_cfg.get("environments") or {}
    base_url = environments.get(env) or next(iter(environments.values()), "")
    elements = (_merge_yaml_dir(yaml, test_dir / "elements") or {}).get("elements", {})
    modules = _merge_yaml_dir(yaml, test_dir / "modules") or {}
    variables = _merge_vars(_load_yaml_dir(yaml, test_dir / "vars"))
    cases_data = _merge_yaml_dir(yaml, test_dir / "cases") or {}
    test_cases = cases_data.get("test_cases", [])
    test_data = (_merge_yaml_dir(yaml, test_dir / "data") or {}).get("test_data", {})
    return ProjectContext(
        project=project,
        test_dir=test_dir,
        base_url=base_url,
        elements=elements or {},
        modules=modules or {},
        variables=variables or {},
        test_cases=test_cases or [],
        test_data=test_data or {},
    )


def summarize_context(
    context: ProjectContext, *, max_items: int = 160
) -> dict[str, Any]:
    return {
        "project": context.project,
        "base_url": context.base_url,
        "existing_case_names": [
            item.get("name")
            for item in context.test_cases[:max_items]
            if isinstance(item, dict)
        ],
        "element_keys": list(context.elements.keys())[:max_items],
        "module_names": list(context.modules.keys())[:max_items],
        "variable_keys": list(context.variables.keys())[:max_items],
        "sample_modules": {
            name: context.modules[name]
            for name in list(context.modules.keys())[: min(10, max_items)]
        },
    }


def _merge_vars(var_files: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in var_files:
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(value, dict) and key in {"dev", "test", "stage", "prod"}:
                    continue
                result[key] = value
    return result


def _merge_yaml_dir(yaml: YamlHandler, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.merge_yaml_files(path) or {}


def _load_yaml_dir(yaml: YamlHandler, path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return yaml.load_yaml_dir(path)
