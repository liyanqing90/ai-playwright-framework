from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

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
from src.yaml_schema import (
    YamlSchemaValidationError,
    load_validation_context,
    validate_case_file,
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
    progress: Callable[[str], None] | None = None,
) -> CaseGenerationResult:
    spec_ref = Path(spec_path)
    artifacts = _GenerationArtifacts(
        project=project, spec_name=_default_output_name(spec_ref)
    )
    try:
        _emit(progress, f"加载项目上下文: project={project}, env={env}")
        context = load_project_context(project, env=env)
        _emit(progress, f"解析生成规格: {spec_ref}")
        spec_path = resolve_generation_spec_path(context, spec_ref)
        resolved_output_name = output_name or _default_output_name(spec_ref)
        _emit(progress, f"读取生成规格: {spec_path}")
        spec = _load_spec(spec_path)
        artifacts.write_json(
            "00_spec.json",
            {"project": project, "env": env, "spec_path": str(spec_path), "spec": spec},
        )
        _emit(progress, "校验生成规格与项目匹配")
        _validate_spec_project_scope(project=project, spec_path=spec_path, spec=spec)
        payload = _build_payload(context, spec, use_ai=use_ai, progress=progress)
        artifacts.write_json("01_model_payload.json", payload)
        payload, warnings = _normalize_validate_payload(
            context=context,
            spec=spec,
            output_name=resolved_output_name,
            payload=payload,
            use_ai=use_ai,
            progress=progress,
            artifacts=artifacts,
        )
        artifacts.write_json("90_final_payload.json", payload)
        _emit(progress, f"准备输出文件: {resolved_output_name}")
        result = _result_paths(context, payload, output_name=resolved_output_name)
        artifacts.write_json(
            "91_write_plan.json",
            {
                "case_file": str(result["case_file"]),
                "data_file": str(result["data_file"]),
                "elements_file": str(result.get("elements_file") or ""),
                "modules_file": str(result.get("modules_file") or ""),
                "vars_file": str(result.get("vars_file") or ""),
            },
        )
        if not dry_run:
            _emit(progress, "写入生成文件")
            _write_payload(
                result,
                overwrite=overwrite,
                verify=lambda: _validate_written_case_file(result),
            )
            _emit(progress, "写入后YAML schema校验通过")
        else:
            _emit(progress, "dry-run模式，跳过写入")
        _emit(progress, "生成完成")
        artifacts.cleanup()
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
    except Exception as exc:
        artifacts.write_text("99_error.txt", f"{type(exc).__name__}: {exc}\n")
        _emit(progress, f"生成过程产物已保留: {artifacts.path}")
        raise


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress:
        progress(message)


class _GenerationArtifacts:
    def __init__(self, *, project: str, spec_name: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_project = _safe_artifact_name(project)
        safe_spec = _safe_artifact_name(spec_name or "generated")
        self.path = (
            Path("logs") / "generation_runs" / f"{timestamp}_{safe_project}_{safe_spec}"
        )

    def write_json(self, name: str, data: Any) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def write_text(self, name: str, text: str) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / name).write_text(text, encoding="utf-8")

    def cleanup(self) -> None:
        if self.path.exists():
            shutil.rmtree(self.path)


def _safe_artifact_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return name or "generated"


def _normalize_validate_payload(
    *,
    context: ProjectContext,
    spec: dict[str, Any],
    output_name: str,
    payload: dict[str, Any],
    use_ai: bool,
    progress: Callable[[str], None] | None,
    artifacts: _GenerationArtifacts | None = None,
) -> tuple[dict[str, Any], list[str]]:
    generation_cfg = load_ai_config().get("generation", {})
    max_repair_attempts = int(generation_cfg.get("max_repair_attempts", 1))
    current_payload = payload

    for attempt in range(max_repair_attempts + 1):
        harness = GenerationHarness(context=context, spec=spec, output_name=output_name)
        try:
            _emit(progress, "Harness归一化生成结果")
            normalized = harness.normalize(current_payload)
            if artifacts:
                artifacts.write_json(
                    f"{attempt + 2:02d}_normalized_payload.json", normalized
                )
            _emit(progress, "Harness校验生成结果")
            warnings = harness.validate(normalized)
            return normalized, warnings
        except Exception as exc:
            if artifacts:
                artifacts.write_text(
                    f"{attempt + 2:02d}_harness_error.txt",
                    f"{type(exc).__name__}: {exc}\n",
                )
            if not use_ai or attempt >= max_repair_attempts:
                raise ValueError(f"Harness校验失败: {exc}") from exc
            _emit(
                progress,
                f"Harness校验失败，调用模型修正({attempt + 1}/{max_repair_attempts}): {exc}",
            )
            current_payload = _repair_payload_with_ai(
                context=context,
                spec=spec,
                invalid_payload=current_payload,
                error=str(exc),
            )
            if artifacts:
                artifacts.write_json(
                    f"{attempt + 2:02d}_repair_payload.json", current_payload
                )

    raise ValueError("Harness校验失败: 未知错误")


def resolve_generation_spec_path(
    context: ProjectContext, spec_path: str | Path
) -> Path:
    raw = Path(spec_path)
    candidates: list[Path] = []

    if len(raw.parts) == 1:
        stem = raw.stem if raw.suffix else raw.name
        candidates.extend(
            [
                context.test_dir / "generation" / f"{stem}.yaml",
                context.test_dir / "generation" / f"{stem}.yml",
            ]
        )

    if raw.suffix:
        candidates.append(raw)
    else:
        candidates.extend([raw.with_suffix(".yaml"), raw.with_suffix(".yml")])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"生成规格不存在: {spec_path}. 已查找: {searched}")


def _default_output_name(spec_path: str | Path) -> str:
    path = Path(spec_path)
    return path.stem if path.suffix else path.name


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
    if "test_data" in parts:
        test_data_index = parts.index("test_data")
        if test_data_index + 2 < len(parts):
            scoped_project = parts[test_data_index + 1]
            scoped_kind = parts[test_data_index + 2]
            if scoped_kind == "generation" and scoped_project != project:
                raise ValueError(
                    f"生成规格目录 test_data/{scoped_project}/generation 与 --project {project} 不一致"
                )
        return

    raise ValueError(
        "生成规格必须放在当前项目目录下: " f"test_data/{project}/generation/<name>.yaml"
    )


def _build_payload(
    context: ProjectContext,
    spec: dict[str, Any],
    *,
    use_ai: bool,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if _has_explicit_steps(spec):
        _emit(progress, "检测到结构化 action steps，跳过模型调用")
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
    _emit(progress, "调用模型生成用例，等待模型响应...")
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
                    "可以生成原生AI步骤 action=ai_step，但只用于自然语言探索或无法稳定拆分的低确定性步骤；"
                    "稳定业务主链路优先生成明确 action、selector/target、value 和断言。"
                    "Module output shape must be modules: {module_name: [step, ...]}, not {module_name: {steps: [...]}}. "
                    "If a module needs case-specific data, use ${param_name} inside the module and pass use_module.params in every call. "
                    "Do not invent variable names such as username_var unless they are also emitted under vars. "
                    "For Saucedemo login, standard_user and locked_out_user must be selected by params, not by sharing one fixed login user. "
                    "每个用例必须至少包含一个断言步骤，断言必须使用项目格式："
                    "assert_visible需要selector；assert_text/assert_text_contains需要selector和value；"
                    "assert_url/assert_url_contains/assert_title需要value。"
                    "断言必须验证前置步骤实际造成的页面状态，不允许用空value或泛化描述充当断言。"
                    "负向登录、错误提示、购物车数量等结果必须断言可观察的准确文案或准确数量。"
                    "使用target而没有selector时，该步骤或data用例层必须声明mode为smart或ai。"
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


def _repair_payload_with_ai(
    *,
    context: ProjectContext,
    spec: dict[str, Any],
    invalid_payload: dict[str, Any],
    error: str,
) -> dict[str, Any]:
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
                    "你是UI自动化生成结果修正器。必须只返回当前框架格式JSON，不要解释。"
                    "修正 invalid_payload 中导致 Harness/schema 失败的问题，保留原业务意图。"
                    "cases层只允许name；description/mode/steps必须在data层；"
                    "selector、target、use_module、action、mode等字段必须是字符串，不能是对象。"
                    "modules必须是 {module_name: [step, ...]}。"
                    "每个用例必须至少有一个准确断言。"
                    "优先复用project_context已有 elements/modules/vars。"
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
                        "validation_error": error,
                        "invalid_payload": invalid_payload,
                        "required_output_shape": {
                            "cases": [{"name": "test_xxx"}],
                            "data": {
                                "test_xxx": {
                                    "description": "说明",
                                    "mode": "strict|smart|ai",
                                    "steps": [
                                        {
                                            "action": "click",
                                            "selector": "已有元素key或原始selector",
                                        },
                                        {
                                            "action": "assert_text",
                                            "selector": "已有元素key或原始selector",
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
        usage_operation="generation.case_repair",
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


def _write_payload(
    result: dict[str, Any],
    *,
    overwrite: bool,
    verify: Callable[[], None] | None = None,
) -> None:
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
    backups: dict[Path, bytes] = {}
    created_files: list[Path] = []
    temp_files: list[Path] = []

    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"文件已存在，使用 --overwrite 覆盖: {path}")
        if path.exists():
            backups[path] = path.read_bytes()
        else:
            created_files.append(path)
        tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            yaml.dump(data, fh)
        temp_files.append(tmp_path)

    try:
        for tmp_path, path in zip(temp_files, files, strict=True):
            tmp_path.replace(path)
        if verify:
            verify()
    except Exception:
        for path, content in backups.items():
            path.write_bytes(content)
        for path in created_files:
            if path.exists():
                path.unlink()
        raise
    finally:
        for tmp_path in temp_files:
            if tmp_path.exists():
                tmp_path.unlink()


def _validate_written_case_file(result: dict[str, Any]) -> None:
    case_file = result["case_file"]
    context = load_validation_context(case_file.parent.parent)
    validate_case_file(case_file, context)
    if context.issues:
        raise YamlSchemaValidationError(context.issues)


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
