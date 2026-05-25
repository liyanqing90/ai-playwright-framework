from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from src.ai_generation.project_context import (
    ProjectContext,
    load_project_context,
    summarize_context,
)
from src.ai_generation.harness import GenerationHarness
from src.ai_runtime.config import load_ai_config
from src.ai_runtime.contracts import GeneratedCasePayload
from src.ai_runtime.provider import (
    ChatCompletionProvider,
)
from utils.yaml_handler import YamlHandler


@dataclass(frozen=True)
class CaseGenerationResult:
    project: str
    case_file: Path
    data_file: Path
    elements_file: Path | None
    modules_file: Path | None
    vars_file: Path | None
    payload: dict[str, Any]
    warnings: list[str]


def generate_case_files(
    *,
    project: str,
    spec_path: str | Path,
    env: str = "prod",
    output_name: str | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    use_ai: bool = True,
) -> CaseGenerationResult:
    context = load_project_context(project, env=env)
    spec_path = Path(spec_path)
    spec = _load_spec(spec_path)
    _validate_spec_project_scope(project=project, spec_path=spec_path, spec=spec)
    payload = _build_payload(context, spec, use_ai=use_ai)
    harness = GenerationHarness(context=context, spec=spec, output_name=output_name)
    payload = harness.normalize(payload)
    warnings = harness.validate(payload)
    result = _result_paths(context, payload, output_name=output_name)
    if not dry_run:
        _write_payload(result, overwrite=overwrite)
    return CaseGenerationResult(
        project=project,
        case_file=result["case_file"],
        data_file=result["data_file"],
        elements_file=result.get("elements_file"),
        modules_file=result.get("modules_file"),
        vars_file=result.get("vars_file"),
        payload=payload,
        warnings=warnings,
    )


def _load_spec(spec_path: str | Path) -> dict[str, Any]:
    path = Path(spec_path)
    if not path.exists():
        raise FileNotFoundError(f"生成规格不存在: {path}")
    data = YamlHandler().load_yaml(path) or {}
    if not isinstance(data, dict):
        raise ValueError("生成规格必须是YAML对象")
    return data


def _validate_spec_project_scope(
    *, project: str, spec_path: str | Path, spec: dict[str, Any]
) -> None:
    explicit_project = spec.get("project")
    if explicit_project and str(explicit_project) != project:
        raise ValueError(
            f"生成规格 project={explicit_project} 与 --project {project} 不一致"
        )

    parts = Path(spec_path).parts
    if "generation_specs" not in parts:
        return
    spec_root_index = parts.index("generation_specs")
    if spec_root_index + 1 >= len(parts):
        return
    scoped_project = parts[spec_root_index + 1]
    if Path(scoped_project).suffix:
        return
    if scoped_project != project:
        raise ValueError(
            f"生成规格目录 generation_specs/{scoped_project} 与 --project {project} 不一致"
        )


def _build_payload(
    context: ProjectContext, spec: dict[str, Any], *, use_ai: bool
) -> dict[str, Any]:
    if _has_explicit_steps(spec):
        return _payload_from_explicit_spec(spec)
    if not use_ai:
        raise ValueError("规格没有显式steps，关闭AI时无法从自然语言生成用例")

    ai_config = load_ai_config()
    generation_cfg = ai_config.get("generation", {})
    prompts_cfg = ai_config.get("prompts", {})
    llm_cfg = ai_config.get("llm", {})
    max_items = int(generation_cfg.get("max_context_items", 160))
    prompt_version = str(prompts_cfg.get("generation_version", "generation-v1"))
    schema_version = str(llm_cfg.get("schema_version", "ui-ai-schema-v1"))
    provider = ChatCompletionProvider()
    result = provider.complete_model(
        [
            {
                "role": "system",
                "content": (
                    "你是UI自动化用例生成器。必须输出当前框架格式的 json 对象，"
                    "响应必须是合法 JSON，不要输出解释。"
                    "字段为 cases, data, elements, modules, vars。"
                    "cases只用于组织用例顺序，只允许包含name。"
                    "description和用例默认mode必须放到data对应用例下。"
                    "用户的generation_spec.cases只描述业务场景，不要求用户指定module、element或变量。"
                    "你必须根据project_context自动选择可复用资产。"
                    "generation_spec中的自然语言用例可以是字符串，也可以是包含name、description、steps的对象。"
                    "当steps是字符串列表时，必须逐条理解为业务步骤，再转换为框架action步骤。"
                    "优先复用已有module、element key、变量key；只有项目资产确实不存在且业务必须新增时，才输出新的elements、modules或vars。"
                    "如果已有module能覆盖登录、进入页面、通用前置步骤，优先用use_module而不是重复生成步骤。"
                    "每个用例必须至少包含一个断言步骤，断言必须使用项目格式："
                    "assert_visible需要selector；assert_text/assert_text_contains需要selector和value；"
                    "assert_url/assert_url_contains/assert_title需要value。"
                    "找不到元素时用 target + mode: smart，不要编造selector。"
                    "不要输出解释。"
                ),
            },
            {
                "role": "user",
                "content": _json_payload(
                    {
                        "project_context": summarize_context(
                            context, max_items=max_items
                        ),
                        "prompt_version": prompt_version,
                        "schema_version": schema_version,
                        "generation_spec": spec,
                        "required_output_shape": {
                            "cases": [{"name": "test_xxx"}],
                            "data": {
                                "test_xxx": {
                                    "description": "说明",
                                    "mode": "strict|smart|ai",
                                    "steps": [
                                        {
                                            "action": "click",
                                            "selector": "已有元素key",
                                        },
                                        {
                                            "action": "assert_text",
                                            "selector": "已有元素key",
                                            "value": "期望文本",
                                        },
                                    ],
                                }
                            },
                            "elements": {},
                            "modules": {},
                            "vars": {},
                        },
                    }
                ),
            },
        ],
        GeneratedCasePayload,
        schema_name="GeneratedCasePayload",
        usage_operation="generation.case_generation",
        usage_metadata={
            "project": context.project,
            "test_dir": str(context.test_dir),
            "schema_name": "GeneratedCasePayload",
        },
    )
    return result.model_dump(exclude_none=True)


def _payload_from_explicit_spec(spec: dict[str, Any]) -> dict[str, Any]:
    raw_cases = spec.get("cases")
    if isinstance(raw_cases, list) and raw_cases and isinstance(raw_cases[0], dict):
        cases: list[dict[str, Any]] = []
        data: dict[str, Any] = {}
        for index, item in enumerate(raw_cases, start=1):
            name = item.get("name") or f"test_generated_{index}"
            cases.append({"name": name})
            steps = []
            for module_name in item.get("reuse_modules", []):
                steps.append({"use_module": module_name})
            steps.extend(item.get("steps") or [])
            data[name] = {
                "description": item.get("description", ""),
                "mode": item.get("mode", spec.get("mode", "strict")),
                "steps": steps,
            }
        return {
            "cases": cases,
            "data": data,
            "elements": spec.get("elements") or {},
            "modules": spec.get("modules") or {},
            "vars": spec.get("vars") or {},
        }

    name = spec.get("case_name") or spec.get("name") or "test_generated"
    steps = []
    for module_name in spec.get("reuse_modules", []):
        steps.append({"use_module": module_name})
    steps.extend(spec.get("steps") or [])
    return {
        "cases": [{"name": name}],
        "data": {
            name: {
                "description": spec.get("description", ""),
                "mode": spec.get("mode", "strict"),
                "steps": steps,
            }
        },
        "elements": spec.get("elements") or {},
        "modules": spec.get("modules") or {},
        "vars": spec.get("vars") or {},
    }


def _result_paths(
    context: ProjectContext, payload: dict[str, Any], *, output_name: str | None
) -> dict[str, Any]:
    first_case = payload["cases"][0]["name"]
    stem = output_name or f"generated_{first_case}"
    result: dict[str, Any] = {
        "payload": payload,
        "case_file": context.test_dir / "cases" / f"{stem}.yaml",
        "data_file": context.test_dir / "data" / f"{stem}.yaml",
    }
    if payload.get("elements"):
        result["elements_file"] = context.test_dir / "elements" / f"{stem}.yaml"
    if payload.get("modules"):
        result["modules_file"] = context.test_dir / "modules" / f"{stem}.yaml"
    if payload.get("vars"):
        result["vars_file"] = context.test_dir / "vars" / f"{stem}.yaml"
    return result


def _write_payload(result: dict[str, Any], *, overwrite: bool) -> None:
    payload = result["payload"]
    files = {
        result["case_file"]: {"test_cases": payload["cases"]},
        result["data_file"]: {"test_data": payload["data"]},
    }
    if result.get("elements_file"):
        files[result["elements_file"]] = {"elements": payload["elements"]}
    if result.get("modules_file"):
        files[result["modules_file"]] = payload["modules"]
    if result.get("vars_file"):
        files[result["vars_file"]] = payload["vars"]

    yaml = YAML()
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"文件已存在，使用 --overwrite 覆盖: {path}")
        with path.open("w", encoding="utf-8") as fh:
            yaml.dump(data, fh)


def _has_explicit_steps(spec: dict[str, Any]) -> bool:
    if _has_structured_steps(spec.get("steps")):
        return True
    cases = spec.get("cases")
    return bool(
        cases
        and isinstance(cases, list)
        and isinstance(cases[0], dict)
        and _has_structured_steps(cases[0].get("steps"))
    )


def _has_structured_steps(steps: Any) -> bool:
    if not isinstance(steps, list) or not steps:
        return False
    return all(isinstance(step, dict) for step in steps)


def _json_payload(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data, ensure_ascii=False)
