from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.ai_runtime.playwright_selectors import looks_like_raw_selector
from src.case_utils import TestCaseFile, yaml_handler
from src.step_actions.action_types import StepAction


class StepPayload(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    action: str | None = None
    use_module: str | None = None
    if_: str | None = Field(default=None, alias="if")
    for_each: Any | None = None
    selector: Any | None = None
    target: Any | None = None
    value: Any | None = None
    mode: str | None = None
    timeout: Any | None = None
    description: str | None = None


class CaseDataPayload(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    description: str = ""
    case_type: str | None = Field(default=None, alias="type")
    mode: str | None = None
    intent: Any | None = None
    inputs: Any | None = None
    checkpoints: Any | None = None
    final: Any | None = None
    steps: Any | None = None


@dataclass(frozen=True)
class SchemaIssue:
    path: str
    message: str

    def format(self) -> str:
        return f"{self.path}: {self.message}"


class YamlSchemaValidationError(ValueError):
    def __init__(self, issues: list[SchemaIssue]):
        self.issues = issues
        message = "YAML schema 校验失败:\n" + "\n".join(
            f"- {issue.format()}" for issue in issues
        )
        super().__init__(message)


@dataclass
class ValidationContext:
    test_dir: Path
    project: str
    elements: dict[str, Any]
    test_datas: dict[str, Any]
    modules: dict[str, Any]
    issues: list[SchemaIssue] = field(default_factory=list)

    def add_issue(self, path: str, message: str) -> None:
        self.issues.append(SchemaIssue(path=path, message=message))


def _action_set(*groups: list[str]) -> set[str]:
    return {action.lower() for group in groups for action in group}


VALID_ACTIONS = {
    action.lower()
    for attr in dir(StepAction)
    if isinstance((group := getattr(StepAction, attr)), list)
    for action in group
}
NO_SELECTOR_ACTIONS = {action.lower() for action in StepAction.NO_SELECTOR_ACTIONS}

CLICK_ACTIONS = _action_set(
    StepAction.CLICK,
    StepAction.HOVER,
    StepAction.DOUBLE_CLICK,
    StepAction.RIGHT_CLICK,
    StepAction.CLEAR,
    StepAction.SCROLL_INTO_VIEW,
    StepAction.FOCUS,
    StepAction.BLUR,
    StepAction.ENTER_FRAME,
    StepAction.DISMISS_ALERT,
)
VALUE_ACTIONS = _action_set(
    StepAction.FILL,
    StepAction.TYPE,
    StepAction.SELECT,
    StepAction.UPLOAD,
)
TEXT_ASSERT_ACTIONS = _action_set(
    StepAction.ASSERT_TEXT,
    StepAction.HARD_ASSERT_TEXT,
    StepAction.ASSERT_TEXT_CONTAINS,
    StepAction.ASSERT_URL,
    StepAction.ASSERT_URL_CONTAINS,
    StepAction.ASSERT_TITLE,
    StepAction.ASSERT_TITLE_CONTAINS,
    StepAction.ASSERT_VALUE,
)
PRESENCE_ASSERT_ACTIONS = _action_set(
    StepAction.ASSERT_VISIBLE,
    StepAction.ASSERT_BE_HIDDEN,
    StepAction.ASSERT_EXISTS,
    StepAction.ASSERT_NOT_EXISTS,
    StepAction.ASSERT_ENABLED,
    StepAction.ASSERT_DISABLED,
)

COMMON_ALLOWED_FIELDS = {
    "action",
    "description",
    "selector",
    "target",
    "value",
    "expected",
    "timeout",
    "mode",
    "nth",
}

ACTION_ALLOWED_FIELDS: dict[str, set[str]] = {}


def _allow(actions: Iterable[str], fields: set[str]) -> None:
    for action in actions:
        ACTION_ALLOWED_FIELDS[action.lower()] = COMMON_ALLOWED_FIELDS | fields


_allow(CLICK_ACTIONS | PRESENCE_ASSERT_ACTIONS, set())
_allow(VALUE_ACTIONS, {"delay"})
_allow(_action_set(StepAction.PRESS_KEY), {"key"})
_allow(TEXT_ASSERT_ACTIONS, {"expected"})
_allow(_action_set(StepAction.ASSERT_ATTRIBUTE), {"attribute", "expected"})
_allow(_action_set(StepAction.ASSERT_ELEMENT_COUNT), {"expected", "expression"})
_allow(_action_set(StepAction.ASSERT_HAVE_VALUES), {"expected_values"})
_allow(
    _action_set(
        StepAction.NAVIGATE,
        StepAction.WAIT,
        StepAction.REFRESH,
        StepAction.PAUSE,
        StepAction.EXECUTE_PYTHON,
        StepAction.EXECUTE_SCRIPT,
        StepAction.ACCEPT_ALERT,
        StepAction.SWITCH_WINDOW,
        StepAction.TAB_SWITCH,
        StepAction.CLOSE_WINDOW,
    ),
    {"x", "y"},
)
_allow(
    _action_set(
        StepAction.STORE_TEXT,
        StepAction.STORE_INPUT_VALUE,
        StepAction.SAVE_ELEMENT_COUNT,
        StepAction.WAIT_FOR_NEW_WINDOW,
        StepAction.DOWNLOAD_FILE,
        StepAction.DOWNLOAD_VERIFY,
    ),
    {"variable_name", "scope", "save_path", "file_pattern"},
)
_allow(_action_set(StepAction.STORE_ATTRIBUTE), {"variable_name", "attribute", "scope"})
_allow(_action_set(StepAction.STORE_VARIABLE), {"name", "scope", "expression"})
_allow(
    _action_set(
        StepAction.WAIT_FOR_ELEMENT_HIDDEN,
        StepAction.WAIT_FOR_ELEMENT_CLICKABLE,
    ),
    set(),
)
_allow(_action_set(StepAction.WAIT_FOR_ELEMENT_TEXT), {"expected_text"})
_allow(_action_set(StepAction.WAIT_FOR_ELEMENT_COUNT), {"expected_count"})
_allow(
    _action_set(StepAction.EXPECT_POPUP),
    {"real_action", "variable_name"},
)
_allow(
    _action_set(StepAction.FAKER),
    {"data_type", "variable_name", "scope"},
)
_allow(_action_set(StepAction.KEYBOARD_SHORTCUT), {"key_combination"})
_allow(_action_set(StepAction.KEYBOARD_PRESS), {"key"})
_allow(_action_set(StepAction.KEYBOARD_TYPE), {"text", "delay"})
_allow(_action_set(StepAction.AI_STEP), {"instruction"})
_allow(
    _action_set(StepAction.MONITOR_REQUEST, StepAction.MONITOR_RESPONSE),
    {
        "url_pattern",
        "action_type",
        "assert_params",
        "save_params",
        "variable_name",
        "scope",
        "key",
    },
)


def validate_all_projects(root: str | Path = "test_data") -> None:
    root_path = Path(root)
    issues: list[SchemaIssue] = []
    for test_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        issues.extend(validate_project(test_dir, raise_on_error=False))
    if issues:
        raise YamlSchemaValidationError(issues)


def validate_pytest_targets(paths: Iterable[str | Path]) -> None:
    case_files = _pytest_case_files(paths)
    if not case_files:
        validate_all_projects()
        return

    contexts: dict[Path, ValidationContext] = {}
    issues: list[SchemaIssue] = []
    for case_file in case_files:
        project_dir = _project_dir_for_case_file(case_file)
        if project_dir is None:
            continue
        context = contexts.get(project_dir)
        if context is None:
            context = load_validation_context(project_dir)
            contexts[project_dir] = context
        validate_case_file(case_file, context)
    for context in contexts.values():
        issues.extend(context.issues)
    if issues:
        raise YamlSchemaValidationError(issues)


def _pytest_case_files(paths: Iterable[str | Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path_text = str(raw_path or "")
        if not path_text or path_text.startswith("-"):
            continue
        path_text = path_text.split("::", 1)[0]
        path = Path(path_text)
        if not path.exists():
            continue
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                result.append(path)
            continue
        if path.is_dir():
            for case_file in sorted(path.rglob("*.y*ml")):
                resolved = case_file.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    result.append(case_file)
    return result


def _project_dir_for_case_file(case_file: Path) -> Path | None:
    parts = case_file.parts
    if "cases" not in parts:
        return None
    cases_index = len(parts) - 1 - list(reversed(parts)).index("cases")
    if cases_index <= 0:
        return None
    return Path(*parts[:cases_index])


def validate_project(
    test_dir: str | Path, *, raise_on_error: bool = True
) -> list[SchemaIssue]:
    context = load_validation_context(test_dir)
    for case_file in sorted((context.test_dir / "cases").glob("*.y*ml")):
        validate_case_file(case_file, context)
    if raise_on_error and context.issues:
        raise YamlSchemaValidationError(context.issues)
    return context.issues


def validate_case_file(case_file: str | Path, context: ValidationContext) -> None:
    path = Path(case_file)
    payload = yaml_handler.load_yaml(path) or {}
    try:
        case_file_payload = TestCaseFile.model_validate(payload)
    except ValidationError as exc:
        context.add_issue(str(path), f"cases 文件结构不合法: {exc}")
        return

    for case in case_file_payload.test_cases:
        case_name = case.name
        data_name = case.data_name or case_name
        case_path = f"{path}:{case_name}"
        if data_name not in context.test_datas:
            context.add_issue(case_path, f"找不到 test_data: {data_name}")
            continue
        validate_case_data(
            context=context,
            case_name=case_name,
            data_name=data_name,
            raw_case_data=context.test_datas[data_name],
            inherited_mode=None,
            path=case_path,
        )


def validate_case_data(
    *,
    context: ValidationContext,
    case_name: str,
    data_name: str,
    raw_case_data: Any,
    inherited_mode: str | None,
    path: str,
) -> None:
    variants = raw_case_data if isinstance(raw_case_data, list) else [raw_case_data]
    if not variants:
        context.add_issue(path, f"test_data 为空: {data_name}")
        return

    for index, variant in enumerate(variants):
        variant_path = f"{path}.data[{index}]"
        try:
            payload = CaseDataPayload.model_validate(variant)
        except ValidationError as exc:
            context.add_issue(variant_path, f"test_data 结构不合法: {exc}")
            continue
        case_type = _normalize_case_type(payload.case_type)
        _validate_case_type(context, variant_path, case_type)
        if case_type == "agent_case":
            _validate_agent_case(
                context=context,
                payload=payload,
                raw_case_data=variant,
                path=variant_path,
            )
            continue

        mode = payload.mode or inherited_mode or "strict"
        _validate_mode(context, variant_path, mode)
        _validate_steps(
            context=context,
            steps=payload.steps or [],
            inherited_mode=mode,
            path=f"{variant_path}.steps",
            module_stack=(),
        )


def _validate_agent_case(
    *,
    context: ValidationContext,
    payload: CaseDataPayload,
    raw_case_data: dict[str, Any],
    path: str,
) -> None:
    if payload.mode is not None:
        context.add_issue(path, "agent_case 不需要声明 mode；type 已决定运行方式")

    allowed_fields = {
        "description",
        "type",
        "mode",
        "intent",
        "steps",
        "inputs",
        "checkpoints",
        "final",
    }
    extra = sorted(set(raw_case_data) - allowed_fields)
    if extra:
        context.add_issue(path, f"agent_case 包含未声明字段: {', '.join(extra)}")

    intent = payload.intent
    steps = payload.steps
    if not _has_non_empty_text(intent) and not steps:
        context.add_issue(path, "agent_case 必须提供 intent 或 steps")
    if intent is not None and not _has_non_empty_text(intent):
        context.add_issue(f"{path}.intent", "intent 必须是非空字符串")
    if steps is not None:
        if not isinstance(steps, list) or not steps:
            context.add_issue(f"{path}.steps", "steps 必须是非空字符串列表")
        else:
            for index, item in enumerate(steps):
                if not _has_non_empty_text(item):
                    context.add_issue(
                        f"{path}.steps[{index}]",
                        "自然语言步骤必须是非空字符串",
                    )

    if payload.inputs is not None and not isinstance(payload.inputs, dict):
        context.add_issue(f"{path}.inputs", "inputs 必须是对象")

    if payload.checkpoints is None and payload.final is None:
        context.add_issue(f"{path}.final", "agent_case 必须声明 checkpoints 或 final")
        return
    _validate_criteria(
        context,
        path,
        {
            "checkpoints": payload.checkpoints,
            "final": payload.final,
        },
        field_label="agent_case",
    )


def _validate_case_type(context: ValidationContext, path: str, case_type: str) -> None:
    if case_type and case_type not in {"standard", "agent_case"}:
        context.add_issue(path, f"type 只支持 standard/agent_case: {case_type}")


def _normalize_case_type(case_type: Any) -> str:
    return str(case_type or "").lower()


def _validate_criteria(
    context: ValidationContext,
    path: str,
    criteria: Any,
    *,
    field_label: str,
) -> None:
    if not isinstance(criteria, dict):
        context.add_issue(path, f"{field_label} 必须是包含 checkpoints/final 的对象")
        return
    allowed_fields = {"checkpoints", "final"}
    extra = sorted(set(criteria) - allowed_fields)
    if extra:
        context.add_issue(path, f"{field_label} 包含未声明字段: {', '.join(extra)}")
    if not criteria.get("checkpoints") and not criteria.get("final"):
        context.add_issue(path, f"{field_label} 必须提供 checkpoints 或 final")
    for field_name in ("checkpoints", "final"):
        value = criteria.get(field_name)
        if value is None:
            continue
        if not isinstance(value, list) or not value:
            context.add_issue(f"{path}.{field_name}", "必须是非空字符串列表")
            continue
        for index, item in enumerate(value):
            if not _has_non_empty_text(item):
                context.add_issue(
                    f"{path}.{field_name}[{index}]",
                    "成功标准必须是非空字符串",
                )


def _has_non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_steps(
    *,
    context: ValidationContext,
    steps: Any,
    inherited_mode: str,
    path: str,
    module_stack: tuple[str, ...],
) -> None:
    if not isinstance(steps, list):
        context.add_issue(path, "steps 必须是列表")
        return
    if not steps:
        context.add_issue(path, "steps 不能为空")
        return

    for index, raw_step in enumerate(steps):
        step_path = f"{path}[{index}]"
        if not isinstance(raw_step, dict):
            context.add_issue(step_path, "步骤必须是 YAML 对象")
            continue
        try:
            step = StepPayload.model_validate(raw_step)
        except ValidationError as exc:
            context.add_issue(step_path, f"步骤结构不合法: {exc}")
            continue
        _validate_step(
            context=context,
            raw_step=raw_step,
            step=step,
            inherited_mode=inherited_mode,
            path=step_path,
            module_stack=module_stack,
        )


def _validate_step(
    *,
    context: ValidationContext,
    raw_step: dict[str, Any],
    step: StepPayload,
    inherited_mode: str,
    path: str,
    module_stack: tuple[str, ...],
) -> None:
    special_fields = [
        bool(step.use_module),
        step.if_ is not None,
        step.for_each is not None,
    ]
    if sum(special_fields) > 1 or (step.action and any(special_fields)):
        context.add_issue(path, "action/use_module/if/for_each 只能声明一种")
        return
    if step.use_module:
        _validate_module_step(
            context, raw_step, step, inherited_mode, path, module_stack
        )
        return
    if step.if_ is not None:
        _validate_condition_step(context, raw_step, inherited_mode, path, module_stack)
        return
    if step.for_each is not None:
        _validate_loop_step(context, raw_step, inherited_mode, path, module_stack)
        return
    if not step.action:
        context.add_issue(path, "普通步骤缺少 action")
        return
    _validate_action_step(context, raw_step, step, inherited_mode, path)


def _validate_module_step(
    context: ValidationContext,
    raw_step: dict[str, Any],
    step: StepPayload,
    inherited_mode: str,
    path: str,
    module_stack: tuple[str, ...],
) -> None:
    _validate_allowed_keys(
        path, raw_step, {"use_module", "params", "description"}, context
    )
    if not isinstance(step.use_module, str) or not step.use_module.strip():
        context.add_issue(path, "use_module 必须是非空字符串")
        return
    if "params" in raw_step and not isinstance(raw_step["params"], dict):
        context.add_issue(path, "params 必须是对象")
    module_name = step.use_module
    if module_name not in context.modules:
        context.add_issue(path, f"找不到 module: {module_name}")
        return
    if module_name in module_stack:
        cycle = " -> ".join(module_stack + (module_name,))
        context.add_issue(path, f"module 存在循环引用: {cycle}")
        return
    module_steps = context.modules[module_name]
    _validate_steps(
        context=context,
        steps=module_steps,
        inherited_mode=inherited_mode,
        path=f"{path}.module[{module_name}]",
        module_stack=module_stack + (module_name,),
    )


def _validate_condition_step(
    context: ValidationContext,
    raw_step: dict[str, Any],
    inherited_mode: str,
    path: str,
    module_stack: tuple[str, ...],
) -> None:
    _validate_allowed_keys(
        path,
        raw_step,
        {"if", "then", "else", "description"},
        context,
    )
    if not isinstance(raw_step.get("if"), str) or not raw_step.get("if", "").strip():
        context.add_issue(path, "if 必须是非空表达式字符串")
    for branch in ("then", "else"):
        if branch in raw_step:
            _validate_steps(
                context=context,
                steps=raw_step.get(branch),
                inherited_mode=inherited_mode,
                path=f"{path}.{branch}",
                module_stack=module_stack,
            )


def _validate_loop_step(
    context: ValidationContext,
    raw_step: dict[str, Any],
    inherited_mode: str,
    path: str,
    module_stack: tuple[str, ...],
) -> None:
    _validate_allowed_keys(
        path,
        raw_step,
        {"for_each", "as", "do", "description"},
        context,
    )
    if "do" not in raw_step:
        context.add_issue(path, "for_each 步骤缺少 do")
        return
    if "as" in raw_step and not isinstance(raw_step["as"], str):
        context.add_issue(path, "as 必须是字符串")
    _validate_steps(
        context=context,
        steps=raw_step.get("do"),
        inherited_mode=inherited_mode,
        path=f"{path}.do",
        module_stack=module_stack,
    )


def _validate_action_step(
    context: ValidationContext,
    raw_step: dict[str, Any],
    step: StepPayload,
    inherited_mode: str,
    path: str,
) -> None:
    action = str(step.action).lower()
    if action not in VALID_ACTIONS:
        context.add_issue(path, f"不支持的 action: {step.action}")
        return

    _validate_allowed_keys(
        path,
        raw_step,
        ACTION_ALLOWED_FIELDS.get(action, COMMON_ALLOWED_FIELDS),
        context,
    )

    mode = str(step.mode or inherited_mode or "strict").lower()
    _validate_mode(context, path, mode)
    _validate_timeout(context, path, raw_step.get("timeout"))

    if action not in NO_SELECTOR_ACTIONS:
        _validate_selector_requirement(context, path, raw_step, mode)

    if action in VALUE_ACTIONS:
        _require_present(context, path, raw_step, "value")
    elif action in _action_set(StepAction.PRESS_KEY):
        _require_any(context, path, raw_step, ("key", "value"))
    elif action in TEXT_ASSERT_ACTIONS:
        _require_any(context, path, raw_step, ("expected", "value"))
    elif action in _action_set(StepAction.ASSERT_ATTRIBUTE):
        _require_present(context, path, raw_step, "attribute")
        _require_any(context, path, raw_step, ("expected", "value"))
    elif action in _action_set(StepAction.ASSERT_ELEMENT_COUNT):
        _require_any(context, path, raw_step, ("expected", "value", "expression"))
    elif action in _action_set(StepAction.ASSERT_HAVE_VALUES):
        _require_any(context, path, raw_step, ("expected_values", "value"))
    elif action in _action_set(StepAction.WAIT_FOR_ELEMENT_TEXT):
        _require_any(context, path, raw_step, ("expected_text", "value"))
    elif action in _action_set(StepAction.WAIT_FOR_ELEMENT_COUNT):
        _require_any(context, path, raw_step, ("expected_count", "value"))
    elif action in _action_set(StepAction.EXECUTE_PYTHON, StepAction.EXECUTE_SCRIPT):
        _require_present(context, path, raw_step, "value")
    elif action in _action_set(StepAction.FAKER):
        _require_present(context, path, raw_step, "data_type")
        _require_present(context, path, raw_step, "variable_name")
    elif action in _action_set(StepAction.KEYBOARD_SHORTCUT):
        _require_any(context, path, raw_step, ("key_combination", "value"))
    elif action in _action_set(StepAction.KEYBOARD_PRESS):
        _require_any(context, path, raw_step, ("key", "value"))
    elif action in _action_set(StepAction.KEYBOARD_TYPE):
        _require_any(context, path, raw_step, ("text", "value"))
    elif action in _action_set(StepAction.AI_STEP):
        _require_any(context, path, raw_step, ("instruction", "value", "target"))
    elif action in _action_set(StepAction.MONITOR_REQUEST, StepAction.MONITOR_RESPONSE):
        _require_any(context, path, raw_step, ("url_pattern", "value"))
    elif action in _action_set(StepAction.DOWNLOAD_VERIFY):
        _require_any(context, path, raw_step, ("file_pattern", "value"))
    elif action in _action_set(StepAction.STORE_TEXT, StepAction.STORE_INPUT_VALUE):
        _require_present(context, path, raw_step, "variable_name")
    elif action in _action_set(StepAction.STORE_ATTRIBUTE):
        _require_present(context, path, raw_step, "variable_name")
        _require_present(context, path, raw_step, "attribute")
    elif action in _action_set(StepAction.STORE_VARIABLE):
        _require_present(context, path, raw_step, "name")
        _require_any(context, path, raw_step, ("value", "expression"))


def _validate_selector_requirement(
    context: ValidationContext,
    path: str,
    raw_step: dict[str, Any],
    mode: str,
) -> None:
    selector = raw_step.get("selector")
    target = raw_step.get("target")
    has_selector = _has_non_empty_value(raw_step, "selector")
    has_target = _has_non_empty_value(raw_step, "target")

    if not has_selector:
        if mode == "smart" and has_target:
            return
        context.add_issue(
            path,
            "该 action 需要 selector；如果没有稳定 selector，请使用 target + mode: smart",
        )
        return

    if not isinstance(selector, str):
        context.add_issue(path, "selector 必须是字符串")
        return
    selector = selector.strip()
    if not selector:
        context.add_issue(path, "selector 不能为空")
        return
    if _is_variable_expression(selector):
        return
    if selector in context.elements:
        return
    if looks_like_raw_selector(selector):
        return
    if mode == "smart" and has_target:
        return
    target_hint = f"，target={target}" if target else ""
    context.add_issue(
        path,
        f"selector 未在 elements 中定义，也不像合法原始 selector: {selector}{target_hint}",
    )


def _validate_allowed_keys(
    path: str,
    raw_step: dict[str, Any],
    allowed: set[str],
    context: ValidationContext,
) -> None:
    unknown = sorted(set(raw_step) - allowed)
    if unknown:
        context.add_issue(path, f"包含未声明字段: {', '.join(unknown)}")


def _validate_mode(
    context: ValidationContext,
    path: str,
    mode: str | None,
) -> None:
    if mode is None:
        return
    normalized = str(mode).lower()
    if normalized not in {"strict", "smart"}:
        context.add_issue(path, f"mode 只支持 strict/smart: {mode}")


def _validate_timeout(context: ValidationContext, path: str, timeout: Any) -> None:
    if timeout is None or _is_variable_expression(str(timeout)):
        return
    try:
        if int(timeout) < 0:
            raise ValueError
    except (TypeError, ValueError):
        context.add_issue(path, f"timeout 必须是非负整数或变量引用: {timeout}")


def _require_present(
    context: ValidationContext, path: str, raw_step: dict[str, Any], field_name: str
) -> None:
    if not _has_required_value(raw_step, field_name):
        context.add_issue(path, f"缺少必要字段: {field_name}")


def _require_any(
    context: ValidationContext,
    path: str,
    raw_step: dict[str, Any],
    field_names: tuple[str, ...],
) -> None:
    if not any(_has_required_value(raw_step, field_name) for field_name in field_names):
        context.add_issue(path, f"至少需要字段之一: {', '.join(field_names)}")


def _has_non_empty_value(raw_step: dict[str, Any], field_name: str) -> bool:
    if field_name not in raw_step:
        return False
    value = raw_step[field_name]
    if value is None:
        return False
    if isinstance(value, str) and value == "":
        return False
    return True


def _has_required_value(raw_step: dict[str, Any], field_name: str) -> bool:
    if field_name not in raw_step:
        return False
    value = raw_step[field_name]
    if value is None:
        return False
    if field_name in {
        "value",
        "expected",
        "expected_text",
        "expected_count",
        "expected_values",
        "expression",
        "url_pattern",
        "file_pattern",
    }:
        return True
    if isinstance(value, str) and value == "":
        return False
    return True


def _is_variable_expression(value: str) -> bool:
    return "${" in value or "$<" in value or "$[[" in value


def load_validation_context(test_dir: str | Path) -> ValidationContext:
    test_dir = Path(test_dir)
    elements = _merge_yaml_section(test_dir / "elements", "elements")
    test_datas = _merge_yaml_section(test_dir / "data", "test_data")
    modules = (
        yaml_handler.merge_yaml_files(test_dir / "modules")
        if (test_dir / "modules").is_dir()
        else {}
    )
    return ValidationContext(
        test_dir=test_dir,
        project=test_dir.name,
        elements=elements if isinstance(elements, dict) else {},
        test_datas=test_datas if isinstance(test_datas, dict) else {},
        modules=modules if isinstance(modules, dict) else {},
    )


def _merge_yaml_section(directory: Path, section: str) -> dict[str, Any]:
    if not directory.is_dir():
        return {}
    merged = yaml_handler.merge_yaml_files(directory) or {}
    value = merged.get(section, {})
    return value if isinstance(value, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验 test_data YAML action schema")
    parser.add_argument(
        "paths",
        nargs="*",
        help="项目 test_data 目录；默认校验 test_data 下所有项目",
    )
    args = parser.parse_args(argv)

    try:
        if args.paths:
            issues: list[SchemaIssue] = []
            for path in args.paths:
                issues.extend(validate_project(path, raise_on_error=False))
            if issues:
                raise YamlSchemaValidationError(issues)
        else:
            validate_all_projects()
    except YamlSchemaValidationError as exc:
        print(exc)
        return 1
    print("[OK] YAML schema 校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
