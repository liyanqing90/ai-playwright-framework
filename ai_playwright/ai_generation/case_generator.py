from __future__ import annotations

import json
import os
import re
import shutil
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from ruamel.yaml import YAML

from ai_playwright.ai_generation.project_context import (
    ProjectContext,
    load_project_context,
    summarize_context,
)
from ai_playwright.ai_generation.harness import GenerationHarness
from ai_playwright.ai_generation.pipeline import compile_case_payload
from ai_playwright.ai_runtime.cache_scope import resolve_entry_scope
from ai_playwright.ai_runtime.config import load_ai_config
from ai_playwright.ai_runtime.contracts import GeneratedCasePayload
from ai_playwright.ai_runtime.provider import (
    ChatCompletionProvider,
    load_llm_settings,
)
from ai_playwright.step_actions.action_types import StepAction
from ai_playwright.step_actions.step_executor import (
    commit_pending_selector_cache,
    discard_pending_selector_cache,
)
from ai_playwright.project_paths import is_packaged_template_path
from ai_playwright.yaml_schema import (
    YamlSchemaValidationError,
    load_validation_context,
    validate_case_file,
)
from ai_playwright.utils.yaml_handler import YamlHandler


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
    verify: bool | None = None,
    progress: Callable[[str], None] | None = None,
) -> CaseGenerationResult:
    spec_ref = Path(spec_path)
    artifacts = _GenerationArtifacts(
        project=project, spec_name=_default_output_name(spec_ref)
    )
    try:
        _emit(progress, f"加载项目上下文: project={project}, env={env}")
        context = load_project_context(project, env=env)
        if not dry_run and is_packaged_template_path(context.test_dir):
            raise ValueError(
                "当前使用的是包内只读 demo 模板。请先运行 "
                "`ai-playwright-init` 初始化本地工作区，或在当前目录提供 "
                "config/env_config.yaml 和 test_data。"
            )
        _emit(progress, f"解析生成规格: {spec_ref}")
        spec_path = resolve_generation_spec_path(context, spec_ref)
        resolved_output_name = output_name or _default_output_name(
            spec_path,
            context=context,
        )
        _emit(progress, f"读取生成规格: {spec_path}")
        spec = _load_spec(spec_path)
        artifacts.write_json(
            "00_spec.json",
            {"project": project, "env": env, "spec_path": str(spec_path), "spec": spec},
        )
        _emit(progress, "校验生成规格与项目匹配")
        _validate_spec_project_scope(project=project, spec_path=spec_path, spec=spec)
        ai_config = load_ai_config()
        generation_cfg = ai_config.get("generation", {})
        verify_after_generate = (
            bool(generation_cfg.get("verify_after_generate", True))
            if verify is None
            else bool(verify)
        )
        compiled = compile_case_payload(
            context=context,
            spec=spec,
            env=env,
            output_name=resolved_output_name,
            build_payload=_build_payload,
            normalize_payload=_normalize_validate_payload,
            use_ai=use_ai,
            use_cache=False,
            artifacts=artifacts,
            progress=progress,
        )
        artifacts.write_json("01_model_payload.json", compiled.raw_payload)
        payload = compiled.payload
        warnings = compiled.warnings
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
            if verify_after_generate:
                previous_selector_cache_hold = os.environ.get(
                    "UI_SELECTOR_CACHE_HOLD_UNTIL_GENERATION_VERIFIED"
                )
                os.environ["UI_SELECTOR_CACHE_HOLD_UNTIL_GENERATION_VERIFIED"] = "1"
                try:
                    _verify_candidate_persist_formal(
                        context=context,
                        env=env,
                        payload=payload,
                        result=result,
                        overwrite=overwrite,
                        output_name=resolved_output_name,
                        artifacts=artifacts,
                        progress=progress,
                    )
                    commit_pending_selector_cache()
                except Exception as exc:
                    discard_pending_selector_cache(str(exc))
                    if not use_ai:
                        raise
                    try:
                        payload, warnings, result = _repair_and_verify_runtime_failure(
                            context=context,
                            spec=spec,
                            env=env,
                            output_name=resolved_output_name,
                            use_ai=use_ai,
                            overwrite=overwrite,
                            failed_payload=payload,
                            failed_error=str(exc),
                            progress=progress,
                            artifacts=artifacts,
                        )
                    except Exception as repair_exc:
                        discard_pending_selector_cache(str(repair_exc))
                        raise
                    commit_pending_selector_cache()
                finally:
                    if previous_selector_cache_hold is None:
                        os.environ.pop(
                            "UI_SELECTOR_CACHE_HOLD_UNTIL_GENERATION_VERIFIED",
                            None,
                        )
                    else:
                        os.environ[
                            "UI_SELECTOR_CACHE_HOLD_UNTIL_GENERATION_VERIFIED"
                        ] = previous_selector_cache_hold
            else:
                _emit(progress, "写入生成文件")
                _write_payload(
                    result,
                    overwrite=overwrite,
                    verify=lambda: _validate_written_case_file(result),
                )
                _emit(progress, "写入后YAML schema校验通过")
                _emit(progress, "跳过生成后执行验证")
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


_VALUE_ASSERTION_ACTIONS = {
    "assert_text",
    "hard_assert",
    "assertion",
    "verify",
    "assert_text_contains",
    "assert_url",
    "assert_url_contains",
    "assert_title",
    "assert_title_contains",
    "assert_value",
    "assert_attribute",
    "assert_element_count",
    "assert_have_values",
}

_STATE_ASSERTION_ACTIONS = {
    "assert_visible",
    "assert_be_hidden",
    "assert_exists",
    "assert_not_exists",
    "assert_enabled",
    "assert_disabled",
}


def _assert_effective_verification_payload(
    context: ProjectContext,
    payload: dict[str, Any],
) -> None:
    modules = dict(context.modules or {})
    modules.update(payload.get("modules") or {})
    data = payload.get("data") or {}
    for case in payload.get("cases") or []:
        if not isinstance(case, dict):
            continue
        case_name = str(case.get("name") or "")
        data_name = str(case.get("data_name") or case_name)
        case_data = data.get(data_name) or {}
        steps = case_data.get("steps") or []
        if not _steps_have_effective_assertion(steps, modules, seen=set()):
            raise ValueError(
                "生成用例缺少有效信息断言，不能进入真实页面验证或缓存: "
                f"case={case_name or data_name}"
            )


def _steps_have_effective_assertion(
    steps: Any,
    modules: dict[str, Any],
    *,
    seen: set[str],
) -> bool:
    if not isinstance(steps, list):
        return False
    for step in steps:
        if not isinstance(step, dict):
            continue
        if _is_effective_assertion_step(step):
            return True
        module_name = step.get("use_module")
        if module_name:
            module_key = str(module_name)
            if module_key in seen:
                continue
            module_steps = _module_steps_for_effective_check(modules.get(module_key))
            if _steps_have_effective_assertion(
                module_steps,
                modules,
                seen=seen | {module_key},
            ):
                return True
    return False


def _module_steps_for_effective_check(raw_module: Any) -> list[dict[str, Any]]:
    if isinstance(raw_module, list):
        return raw_module
    if isinstance(raw_module, dict) and isinstance(raw_module.get("steps"), list):
        return raw_module["steps"]
    return []


def _is_effective_assertion_step(step: dict[str, Any]) -> bool:
    action = str(step.get("action") or "").lower()
    if action in _STATE_ASSERTION_ACTIONS:
        return _has_nonempty_value(step.get("selector") or step.get("target"))
    if action not in _VALUE_ASSERTION_ACTIONS:
        return False
    if action == "assert_attribute":
        return _has_nonempty_value(step.get("attribute")) and _has_any_nonempty(
            step,
            ("expected", "value"),
        )
    if action == "assert_element_count":
        return _has_any_nonempty(step, ("expected", "value", "expression"))
    if action == "assert_have_values":
        return _has_any_nonempty(step, ("expected_values", "expected", "value"))
    return _has_any_nonempty(step, ("expected", "value"))


def _has_any_nonempty(step: dict[str, Any], fields: tuple[str, ...]) -> bool:
    return any(_has_nonempty_value(step.get(field)) for field in fields)


def _has_nonempty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def _write_and_verify_candidate(
    *,
    context: ProjectContext,
    env: str,
    payload: dict[str, Any],
    output_name: str,
    artifacts: _GenerationArtifacts,
    progress: Callable[[str], None] | None,
) -> tuple[ProjectContext, dict[str, Any]]:
    candidate_context = _candidate_context(context, artifacts)
    candidate_result = _result_paths(
        candidate_context,
        payload,
        output_name=output_name,
    )
    artifacts.write_json(
        "92_candidate_write_plan.json",
        {
            "test_dir": str(candidate_context.test_dir),
            "case_file": str(candidate_result["case_file"]),
            "data_file": str(candidate_result["data_file"]),
            "elements_file": str(candidate_result.get("elements_file") or ""),
            "modules_file": str(candidate_result.get("modules_file") or ""),
            "vars_file": str(candidate_result.get("vars_file") or ""),
        },
    )
    _emit(progress, f"写入候选验证文件: {candidate_result['case_file']}")
    _write_payload(
        candidate_result,
        overwrite=True,
        verify=lambda: _validate_written_case_file(candidate_result),
    )
    _emit(progress, "候选YAML schema校验通过")
    _verify_generated_case(
        context=candidate_context,
        env=env,
        result=candidate_result,
        progress=progress,
        stage="候选",
        artifacts=artifacts,
    )
    return candidate_context, candidate_result


def _verify_candidate_persist_formal(
    *,
    context: ProjectContext,
    env: str,
    payload: dict[str, Any],
    result: dict[str, Any],
    overwrite: bool,
    output_name: str,
    artifacts: _GenerationArtifacts,
    progress: Callable[[str], None] | None,
) -> None:
    _assert_effective_verification_payload(context, payload)
    _write_and_verify_candidate(
        context=context,
        env=env,
        payload=payload,
        output_name=output_name,
        artifacts=artifacts,
        progress=progress,
    )
    _emit(progress, "候选用例真实页面验证通过，写入正式用例")
    _write_payload(
        result,
        overwrite=overwrite,
        verify=lambda: _validate_written_case_file(result),
        post_verify=lambda: _verify_generated_case(
            context=context,
            env=env,
            result=result,
            progress=progress,
            stage="正式存储后",
            artifacts=artifacts,
        ),
    )
    _emit(progress, "正式用例再次执行验证通过")
    _emit(progress, "写入后YAML schema校验通过")


def _repair_and_verify_runtime_failure(
    *,
    context: ProjectContext,
    spec: dict[str, Any],
    env: str,
    output_name: str,
    use_ai: bool,
    overwrite: bool,
    failed_payload: dict[str, Any],
    failed_error: str,
    progress: Callable[[str], None] | None,
    artifacts: _GenerationArtifacts,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    generation_cfg = load_ai_config().get("generation", {})
    max_attempts = int(generation_cfg.get("runtime_repair_attempts", 1))
    if max_attempts <= 0:
        raise AssertionError(failed_error)

    current_payload = failed_payload
    current_error = failed_error
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        _emit(
            progress,
            f"候选真实页面验证失败，调用模型针对错误点修复({attempt}/{max_attempts})",
        )
        repaired_payload = _repair_payload_with_ai(
            context=context,
            spec=spec,
            invalid_payload=_payload_with_referenced_context_modules(
                current_payload, context=context
            ),
            error=current_error,
        )
        repaired_payload = _payload_preserving_referenced_context_modules(
            repaired_payload,
            previous_payload=current_payload,
            context=context,
        )
        artifacts.write_json(
            f"9{attempt + 2}_runtime_repair_payload.json", repaired_payload
        )
        payload, warnings = _normalize_validate_payload(
            context=context,
            spec=spec,
            output_name=output_name,
            payload=repaired_payload,
            use_ai=use_ai,
            progress=progress,
            artifacts=artifacts,
        )
        result = _result_paths(context, payload, output_name=output_name)
        try:
            _verify_candidate_persist_formal(
                context=context,
                env=env,
                payload=payload,
                result=result,
                overwrite=overwrite,
                output_name=output_name,
                artifacts=artifacts,
                progress=progress,
            )
            return payload, warnings, result
        except Exception as exc:
            last_error = exc
            current_payload = _payload_preserving_referenced_context_modules(
                payload,
                previous_payload=current_payload,
                context=context,
            )
            current_error = str(exc)
            artifacts.write_text(
                f"9{attempt + 2}_runtime_repair_error.txt",
                f"{type(exc).__name__}: {exc}\n",
            )

    raise AssertionError(f"运行验证修复失败: {last_error or failed_error}")


def _candidate_context(
    context: ProjectContext,
    artifacts: _GenerationArtifacts,
) -> ProjectContext:
    source_test_dir = Path(context.test_dir)
    candidate_test_dir = (artifacts.path / "candidate_test_dir").resolve()
    if candidate_test_dir.exists():
        shutil.rmtree(candidate_test_dir)
    if source_test_dir.exists():
        shutil.copytree(
            source_test_dir,
            candidate_test_dir,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
        )
    else:
        candidate_test_dir.mkdir(parents=True, exist_ok=True)
    for dirname in ("cases", "data", "elements", "modules", "vars"):
        (candidate_test_dir / dirname).mkdir(parents=True, exist_ok=True)
    return replace(context, test_dir=candidate_test_dir)


def _verify_generated_case(
    *,
    context: ProjectContext,
    env: str,
    result: dict[str, Any],
    progress: Callable[[str], None] | None,
    stage: str = "生成",
    artifacts: _GenerationArtifacts | None = None,
) -> None:
    case_file = Path(result["case_file"])
    _emit(progress, f"执行{stage}用例真实页面验证: {case_file}")
    previous_env = {name: os.environ.get(name) for name in _VERIFY_ENV_KEYS}
    os.environ["TEST_PROJECT"] = context.project
    os.environ["TEST_ENV"] = env
    os.environ["TEST_DIR"] = str(context.test_dir)
    os.environ["BASE_URL"] = str(context.base_url or "")
    os.environ.setdefault("BROWSER", "chromium")
    os.environ.setdefault("PWHEADED", "1")
    os.environ.setdefault("PWSLOWMO", "0")
    os.environ.setdefault("UI_AI_MODE", "strict")
    os.environ["UI_SELECTOR_CACHE_COMMIT_MODE"] = "deferred"
    _configure_runtime_for_verification(context=context, env=env)
    _clear_runtime_data_caches()
    output_buffer = StringIO()
    try:
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            exit_code = pytest.main(
                [
                    str(case_file),
                    "-v",
                    "--tb=line",
                    "-p",
                    "no:warnings",
                    "--skip-yaml-schema",
                    "-s",
                    "--alluredir=reports/allure-results",
                    "--clean-alluredir",
                ]
            )
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    output = output_buffer.getvalue()
    if artifacts:
        artifacts.write_text(
            f"{_safe_artifact_name(stage)}_pytest_output.txt",
            output,
        )
    if exit_code != 0:
        raise AssertionError(
            f"{stage}用例真实页面验证失败: exit_code={exit_code}\n"
            f"{_tail_text(output)}"
        )
    _emit(progress, f"{stage}用例真实页面验证通过")


def _regenerate_without_cache(
    *,
    context: ProjectContext,
    spec: dict[str, Any],
    env: str,
    output_name: str,
    use_ai: bool,
    progress: Callable[[str], None] | None,
    artifacts: _GenerationArtifacts,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    payload = _build_payload(
        context,
        spec,
        env=env,
        output_name=output_name,
        use_ai=use_ai,
        use_cache=False,
        progress=progress,
    )
    artifacts.write_json("92_regenerated_payload.json", payload)
    payload, warnings = _normalize_validate_payload(
        context=context,
        spec=spec,
        output_name=output_name,
        payload=payload,
        use_ai=use_ai,
        progress=progress,
        artifacts=artifacts,
    )
    result = _result_paths(context, payload, output_name=output_name)
    return payload, warnings, result


_VERIFY_ENV_KEYS = (
    "TEST_PROJECT",
    "TEST_ENV",
    "TEST_DIR",
    "BASE_URL",
    "BROWSER",
    "PWHEADED",
    "PWSLOWMO",
    "UI_AI_MODE",
    "UI_SELECTOR_CACHE_COMMIT_MODE",
)


def _configure_runtime_for_verification(*, context: ProjectContext, env: str) -> None:
    try:
        from ai_playwright.utils.config import Config

        Config(
            project=context.project,
            env=env,
            base_url=str(context.base_url or ""),
            test_dir=str(context.test_dir),
        )
    except Exception:
        # Environment variables above remain the source of truth for verification.
        pass


def _clear_runtime_data_caches() -> None:
    try:
        from ai_playwright.case_utils import _load_data_for

        _load_data_for.cache_clear()
    except Exception:
        pass


def _tail_text(text: str, *, max_chars: int = 4000) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


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


def _default_output_name(
    spec_path: str | Path, *, context: ProjectContext | None = None
) -> str:
    path = Path(spec_path)
    if context is not None:
        try:
            generation_dir = (context.test_dir / "generation").resolve()
            relative_path = path.resolve().relative_to(generation_dir)
        except Exception:
            relative_path = path
        if relative_path.suffix:
            relative_path = relative_path.with_suffix("")
        return relative_path.as_posix()
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
    env: str,
    output_name: str,
    use_ai: bool,
    cache_info: dict[str, Any] | None = None,
    use_cache: bool = True,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if _has_explicit_steps(spec):
        _emit(progress, "检测到结构化 action steps，跳过模型调用")
        return _payload_from_explicit_spec(spec)
    if not use_ai:
        raise ValueError("规格没有显式steps，关闭AI时无法从自然语言生成用例")

    navigation_context = _resolve_navigation_context(context, spec)
    if not navigation_context["resolved"]:
        raise ValueError(
            "生成规格、公共module、项目配置均未提供入口URL，无法生成可执行UI用例"
        )

    ai_config = load_ai_config()
    generation_cfg = ai_config.get("generation", {})
    prompts_cfg = ai_config.get("prompts", {})
    llm_cfg = ai_config.get("llm", {})
    max_items = int(generation_cfg.get("max_context_items", 160))
    prompt_version = str(prompts_cfg.get("generation_version", "generation-v1"))
    schema_version = str(llm_cfg.get("schema_version", "ui-ai-schema-v1"))
    runtime_compile = (
        spec.get("runtime_compile")
        if isinstance(spec.get("runtime_compile"), dict)
        else {}
    )
    system_extra = _runtime_compile_prompt_rules(runtime_compile)
    required_output_shape = {
        "cases": [{"name": "test_xxx"}],
        "data": {
            "test_xxx": {
                "mode": "smart",
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
    }
    if system_extra:
        required_output_shape["runtime_compile_contract"] = {
            "modules": {},
            "allowed_use_module": runtime_compile.get("allowed_modules") or [],
        }
    if cache_info is not None:
        cache_info.update({"cache_hit": False, "model_calls": 1})
    _emit(progress, "调用模型生成用例，等待模型响应...")
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
                    "cases层只用于组织用例顺序；description可省略；mode可省略，省略时默认smart。"
                    "用户的generation_spec.cases只描述业务场景，不要求用户指定module、element或变量。"
                    "你必须根据project_context自动选择可复用资产。"
                    "入口URL按就近原则解析：优先使用generation_spec的自然语言steps或description中的URL；"
                    "如果规格没有URL，优先复用project_context里已有module的goto/open/navigate；"
                    "如果module也没有入口，再使用project_context.base_url。禁止凭站点名称猜URL。"
                    "generation_spec中的自然语言用例可以是字符串，也可以是包含name、description、intent、steps、inputs、checkpoints、final的对象。"
                    "inputs不是必需字段，只是可选的结构化补充；值可能直接写在description、intent、steps、checkpoints或final里。"
                    "当steps是字符串列表时，必须逐条理解为业务步骤，再转换为框架action步骤。"
                    "checkpoints/final是验收标准，生成结果必须把它们落实为项目格式断言步骤。"
                    "优先复用已有module、element key、变量key；只有项目资产确实不存在且业务必须新增时，才输出新的elements、modules或vars。"
                    "如果已有module能覆盖当前目标所需的前置步骤或公共流程，优先用use_module而不是重复生成步骤。"
                    "可以生成原生AI步骤 action=ai_step，但只用于自然语言探索或无法稳定拆分的低确定性步骤；"
                    "稳定业务主链路优先生成明确 action、selector/target、value 和断言。"
                    "action必须来自valid_actions；框架会在页面操作后自动等待页面稳定，不要编造等待跳转类action。"
                    "Module output shape must be modules: {module_name: [step, ...]}, not {module_name: {steps: [...]}}. "
                    'Call modules as a step shaped exactly like {"use_module":"module_name","params":{...}}; do not use action=use_module or module_name fields. '
                    "If a module needs case-specific data, use ${param_name} inside the module and pass use_module.params in every call. "
                    "Do not put params or inputs on data.<case>; case-specific module data belongs on the use_module step as params. "
                    "Only reference ${name} when it is backed by project_context.variable_keys, generation_spec.inputs, use_module.params, or emitted vars; otherwise use the literal value from the natural language. "
                    "Do not invent variable names such as username_var unless they are also emitted under vars. "
                    "Do not output vars for keys that already exist in project_context.variable_keys; reference the existing variable key instead. "
                    "When similar cases differ by input values, roles, states, or expected outcomes, parameterize shared modules via use_module.params instead of hard-coding one fixed value. "
                    "每个用例必须至少包含一个断言步骤，断言必须使用项目格式："
                    "assert_visible需要selector；assert_text/assert_text_contains需要selector和value；"
                    "assert_url/assert_url_contains/assert_title/assert_title_contains需要value。"
                    "验证输入框/textarea/select当前值必须用assert_value，不要用assert_attribute attribute=value。"
                    "额外断言可以用于确认后续规划所需状态，但只能使用框架原生支持的断言能力，且必须有可执行selector/target和可观察期望。"
                    "不要为了证明click已经执行而额外生成无业务期望、无法稳定执行的断言；点击失败由执行器直接报错。"
                    "断言必须验证前置步骤实际造成或后续规划需要的页面状态，不允许用空value或泛化描述充当断言。"
                    "所有结果验证必须断言可观察的准确信息，例如页面文案、URL、标题、状态或数量。"
                    "使用target而没有selector时，该步骤或data用例层必须声明mode为smart。"
                    "找不到元素时用 target + mode: smart，不要编造selector。"
                    f"{system_extra}"
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
                        "navigation_context": navigation_context,
                        "prompt_version": prompt_version,
                        "schema_version": schema_version,
                        "valid_actions": _valid_step_actions(),
                        "generation_spec": spec,
                        "required_output_shape": required_output_shape,
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


def _generation_input_type(spec: dict[str, Any]) -> str:
    if spec.get("steps"):
        return "natural_steps"
    cases = spec.get("cases")
    if isinstance(cases, list):
        for case in cases:
            if isinstance(case, dict) and case.get("steps"):
                return "natural_steps"
    return "intent"


def _runtime_compile_prompt_rules(runtime_compile: dict[str, Any]) -> str:
    if not runtime_compile or runtime_compile.get("allow_new_modules") is not False:
        return ""
    allowed_modules = [
        str(name)
        for name in runtime_compile.get("allowed_modules") or []
        if str(name).strip()
    ]
    allowed = ", ".join(allowed_modules) if allowed_modules else "无"
    return (
        "运行时编译约束：当前请求来自 run_case agent_case，只能生成内存执行计划，"
        "不能创建、修改或输出任何module资产；输出的 modules 必须是空对象 {}。"
        "use_module 只能引用 project_context 中已存在且列在 runtime_compile.allowed_modules 的module；"
        f"当前允许引用的module为: {allowed}。"
        "如果已有module不能精确满足参数化需求，不要新建参数化module，也不要输出modules；"
        "应改为在 data.<case>.steps 中生成直接 action 步骤，引用已有 element key/target 和字面值/变量。"
    )


def _safe_generation_model_key() -> str:
    try:
        return load_llm_settings().model
    except Exception:
        return ""


def _valid_step_actions() -> list[str]:
    return sorted(
        {
            action.lower()
            for attr in dir(StepAction)
            if isinstance((items := getattr(StepAction, attr)), list)
            for action in items
        }
    )


def _payload_with_referenced_context_modules(
    payload: dict[str, Any],
    *,
    context: ProjectContext,
) -> dict[str, Any]:
    enriched = deepcopy(payload)
    modules = enriched.setdefault("modules", {})
    if not isinstance(modules, dict):
        modules = {}
        enriched["modules"] = modules
    for module_name in sorted(_referenced_module_names(enriched)):
        if module_name not in modules and module_name in context.modules:
            modules[module_name] = deepcopy(context.modules[module_name])
    return enriched


def _payload_preserving_referenced_context_modules(
    payload: dict[str, Any],
    *,
    previous_payload: dict[str, Any],
    context: ProjectContext,
) -> dict[str, Any]:
    preserved = deepcopy(payload)
    modules = preserved.setdefault("modules", {})
    if not isinstance(modules, dict):
        modules = {}
        preserved["modules"] = modules

    referenced = _referenced_module_names(preserved) | _referenced_module_names(
        previous_payload
    )
    previous_modules = previous_payload.get("modules") or {}
    for module_name in sorted(referenced):
        if module_name not in context.modules:
            continue
        context_steps = _module_steps_for_effective_check(context.modules[module_name])
        generated_steps = _module_steps_for_effective_check(modules.get(module_name))
        previous_steps = _module_steps_for_effective_check(
            previous_modules.get(module_name)
        )
        if not generated_steps:
            continue
        if context_steps and len(generated_steps) < len(context_steps):
            modules[module_name] = deepcopy(
                previous_modules.get(module_name) or context.modules[module_name]
            )
        elif previous_steps and len(generated_steps) < len(previous_steps):
            modules[module_name] = deepcopy(previous_modules[module_name])
    return preserved


def _referenced_module_names(payload: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for case_data in (payload.get("data") or {}).values():
        if isinstance(case_data, dict):
            names.update(_module_refs_from_steps(case_data.get("steps") or []))
    for module_steps in (payload.get("modules") or {}).values():
        names.update(_module_refs_from_steps(module_steps or []))
    return names


def _module_refs_from_steps(steps: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(steps, dict):
        steps = steps.get("steps") or []
    if not isinstance(steps, list):
        return names
    for step in steps:
        if not isinstance(step, dict):
            continue
        module_name = step.get("use_module")
        if module_name is None and step.get("module") and not step.get("action"):
            module_name = step.get("module")
        if isinstance(module_name, str) and module_name.strip():
            names.add(module_name.strip())
    return names


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
    navigation_context = _resolve_navigation_context(context, spec)
    result = provider.complete_model(
        [
            {
                "role": "system",
                "content": (
                    "你是UI自动化生成结果修正器。必须只返回当前框架格式JSON，不要解释。"
                    "修正 invalid_payload 中导致 Harness/schema/真实页面验证失败的问题，保留原业务意图。"
                    "cases层只允许name；description可省略；mode可省略，省略时默认smart；steps必须在data层；"
                    "data.<case>层不要输出params或inputs；模块入参必须放到对应use_module步骤的params字段；"
                    "inputs不是必需字段；如果规格把值直接写在description、intent、steps、checkpoints或final里，修正后应直接使用字面值。"
                    "selector、target、use_module、action、mode等字段必须是字符串，不能是对象。"
                    "modules必须是 {module_name: [step, ...]}。"
                    "每个用例必须至少有一个准确断言。"
                    "验证输入框/textarea/select当前值必须用assert_value，不要用assert_attribute attribute=value。"
                    "不要为了证明click已经执行而额外生成无业务期望的assert_visible；点击失败由执行器直接报错。"
                    "优先复用project_context已有 elements/modules/vars。"
                    "action必须来自valid_actions；框架会在页面操作后自动等待页面稳定，不要编造等待跳转类action。"
                    "不要输出或覆盖project_context.variable_keys中已存在的vars；直接引用已有变量key。"
                    "只有变量被project_context.variable_keys、generation_spec.inputs、use_module.params或vars支撑时才引用${name}；否则不要把字面值改写成变量。"
                    "invalid_payload.modules中可能包含本次失败用例引用的上下文module副本；"
                    "如果错误发生在某个module内部，必须修改该module或移除对它的复用，不能原样返回失败module。"
                    "如果真实页面验证提示selector匹配不到元素、fixture value缺失或测试数据未参数化，"
                    "必须针对错误点修复对应steps/elements/modules/data映射；不要保留导致失败的新增element selector。"
                ),
            },
            {
                "role": "user",
                "content": _json_payload(
                    {
                        "project_context": summarize_context(
                            context, max_items=max_items
                        ),
                        "navigation_context": navigation_context,
                        "prompt_version": prompt_version,
                        "schema_version": schema_version,
                        "valid_actions": _valid_step_actions(),
                        "generation_spec": spec,
                        "validation_error": error,
                        "invalid_payload": invalid_payload,
                        "required_output_shape": {
                            "cases": [{"name": "test_xxx"}],
                            "data": {
                                "test_xxx": {
                                    "mode": "smart",
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


def _resolve_navigation_context(
    context: ProjectContext,
    spec: dict[str, Any],
) -> dict[str, Any]:
    scope = resolve_entry_scope(
        spec=spec,
        modules=context.modules,
        base_url=str(context.base_url or ""),
        spec_source_name="generation_spec",
        priority=[
            "generation_spec.steps_or_description_url",
            "project_context.module_goto",
            "project_context.base_url",
        ],
    )
    scope["generation_spec_urls"] = scope.pop("spec_urls", [])
    return scope


def _extract_spec_urls(spec: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    cases = spec.get("cases")
    if isinstance(cases, list):
        for case in cases:
            if isinstance(case, dict):
                urls.extend(_extract_urls(case.get("steps")))
                urls.extend(_extract_urls(case.get("intent")))
    urls.extend(_extract_urls(spec.get("steps")))
    urls.extend(_extract_urls(spec.get("intent")))
    if isinstance(cases, list):
        for case in cases:
            if isinstance(case, dict):
                urls.extend(_extract_urls(case.get("description")))
            elif isinstance(case, str):
                urls.extend(_extract_urls(case))
    urls.extend(_extract_urls(spec.get("description")))
    return _dedupe_preserve_order(urls)


def _module_goto_entries(modules: dict[str, Any]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for module_name, raw_steps in modules.items():
        steps = raw_steps.get("steps") if isinstance(raw_steps, dict) else raw_steps
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            action = str(step.get("action") or "").lower()
            if action not in {"goto", "open", "navigate"}:
                continue
            value = step.get("value") or step.get("url")
            if not value:
                continue
            entries.append(
                {
                    "module": str(module_name),
                    "action": action,
                    "value": str(value),
                }
            )
    return entries


def _extract_urls(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [
            match.group(0).rstrip(".,;，。；、)")
            for match in re.finditer(r"https?://[^\s\"'<>）)]+", value)
        ]
    if isinstance(value, list):
        urls: list[str] = []
        for item in value:
            urls.extend(_extract_urls(item))
        return urls
    if isinstance(value, dict):
        urls: list[str] = []
        for item in value.values():
            urls.extend(_extract_urls(item))
        return urls
    return []


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


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
            case_data = {
                "mode": item.get("mode", spec.get("mode", "smart")),
                "steps": steps,
            }
            description = item.get("description") or spec.get("description")
            if description:
                case_data["description"] = description
            data[name] = case_data
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
    case_data = {
        "mode": spec.get("mode", "smart"),
        "steps": steps,
    }
    if spec.get("description"):
        case_data["description"] = spec["description"]
    return {
        "cases": [{"name": name}],
        "data": {name: case_data},
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
    post_verify: Callable[[], None] | None = None,
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
        files[result["vars_file"]] = _merged_vars_payload(
            result["vars_file"],
            payload.get("vars") or {},
        )

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
        if post_verify:
            post_verify()
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


def _merged_vars_payload(path: Path, generated_vars: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if path.exists():
        yaml = YAML(typ="safe")
        with path.open("r", encoding="utf-8") as fh:
            existing = yaml.load(fh) or {}
        if isinstance(existing, dict):
            merged.update(deepcopy(existing))
    merged.update(deepcopy(generated_vars))
    return merged


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
