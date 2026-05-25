from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.ai_generation.project_context import ProjectContext
from src.step_actions.action_types import StepAction


_VALID_MODES = {"strict", "smart", "ai"}


@dataclass
class GenerationHarness:
    context: ProjectContext
    spec: dict[str, Any]
    output_name: str | None = None

    def normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = self._normalize_top_level(payload)
        self._normalize_cases_and_data(payload)
        for case_data in payload["data"].values():
            if isinstance(case_data, dict):
                case_data["steps"] = self._normalize_steps(case_data.get("steps") or [])
        return payload

    def validate(self, payload: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        valid_actions = self._valid_actions()
        case_names = [case.get("name") for case in payload.get("cases", [])]

        for case in payload.get("cases", []):
            extra_keys = sorted(set(case.keys()) - {"name"})
            if extra_keys:
                raise ValueError(f"cases层只允许name字段，发现: {extra_keys}")

        for name in case_names:
            if not name:
                raise ValueError("生成结果存在缺少name的用例")
            if name not in payload.get("data", {}):
                raise ValueError(f"生成结果缺少data步骤: {name}")

        known_elements = set(self.context.elements) | set(payload.get("elements") or {})
        known_modules = set(self.context.modules) | set(payload.get("modules") or {})

        for case_name, case_data in (payload.get("data") or {}).items():
            if not isinstance(case_data, dict):
                raise ValueError(f"data.{case_name} 必须是对象")
            mode = str(case_data.get("mode", "strict")).lower()
            if mode not in _VALID_MODES:
                raise ValueError(f"data.{case_name}.mode 不合法: {mode}")
            steps = case_data.get("steps")
            if not isinstance(steps, list) or not steps:
                raise ValueError(f"data.{case_name} 缺少steps")
            has_assertion = False
            for index, step in enumerate(steps, start=1):
                if self._validate_step(
                    case_name=case_name,
                    index=index,
                    step=step,
                    valid_actions=valid_actions,
                    known_elements=known_elements,
                    known_modules=known_modules,
                    warnings=warnings,
                ):
                    has_assertion = True
            if not has_assertion:
                raise ValueError(
                    f"data.{case_name} 缺少断言步骤。每个生成用例必须至少包含一个项目格式断言。"
                )
        return warnings

    def _normalize_top_level(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "test_cases" in payload:
            payload = {
                "cases": payload.get("test_cases") or [],
                "data": payload.get("test_data") or {},
                "elements": payload.get("elements") or {},
                "modules": payload.get("modules") or {},
                "vars": payload.get("vars") or {},
            }
        payload.setdefault("cases", [])
        payload.setdefault("data", {})
        payload.setdefault("elements", {})
        payload.setdefault("modules", {})
        payload.setdefault("vars", {})
        return payload

    def _normalize_cases_and_data(self, payload: dict[str, Any]) -> None:
        if not payload["cases"]:
            name = self.spec.get("case_name") or self.output_name or "test_generated"
            payload["cases"] = [{"name": name}]

        normalized_cases: list[dict[str, str]] = []
        normalized_data: dict[str, Any] = {}
        for raw_case in payload["cases"]:
            if not isinstance(raw_case, dict):
                raise ValueError(f"cases条目必须是对象: {raw_case}")
            raw_name = raw_case.get("name")
            if not raw_name:
                raise ValueError(f"cases条目缺少name: {raw_case}")
            name = _safe_case_name(str(raw_name))
            raw_data = payload.get("data", {}).get(raw_name)
            case_data = dict(raw_data) if isinstance(raw_data, dict) else {}
            case_data.setdefault(
                "description",
                raw_case.get("description") or self.spec.get("description", ""),
            )
            case_data.setdefault("mode", self._case_mode(raw_case))
            normalized_cases.append({"name": name})
            normalized_data[name] = case_data
        payload["cases"] = normalized_cases
        payload["data"] = normalized_data

    def _case_mode(self, raw_case: dict[str, Any]) -> str:
        if self.spec.get("mode") and not _has_explicit_steps(self.spec):
            return str(self.spec["mode"]).lower()
        return str(raw_case.get("mode") or self.spec.get("mode") or "strict").lower()

    def _normalize_steps(self, steps: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError(f"生成结果中的step必须是对象: {step}")
            item = dict(step)
            module_name = self._module_name(item)
            if module_name:
                normalized.append({"use_module": module_name})
                continue

            action = str(item.get("action") or "").lower()
            if action:
                item["action"] = _ACTION_ALIASES.get(action, item.get("action"))
                item = _normalize_verify_step(item)
            normalized.append(item)
        return normalized

    @staticmethod
    def _module_name(step: dict[str, Any]) -> str | None:
        if "use_module" in step:
            return step.get("use_module")
        if "module" in step and "action" not in step:
            return step.get("module")
        action = str(step.get("action") or "").lower()
        if action in {"module", "use_module", "reuse_module", "call_module"}:
            return (
                step.get("use_module")
                or step.get("module")
                or step.get("value")
                or step.get("name")
            )
        return None

    @staticmethod
    def _valid_actions() -> set[str]:
        return {
            action.lower()
            for attr in dir(StepAction)
            if isinstance((items := getattr(StepAction, attr)), list)
            for action in items
        }

    @staticmethod
    def _validate_step(
        *,
        case_name: str,
        index: int,
        step: dict[str, Any],
        valid_actions: set[str],
        known_elements: set[str],
        known_modules: set[str],
        warnings: list[str],
    ) -> bool:
        if not isinstance(step, dict):
            raise ValueError(f"{case_name} step {index} 必须是对象")
        module_name = step.get("use_module")
        if module_name:
            if module_name not in known_modules:
                raise ValueError(f"{case_name} 引用了不存在的公共组件: {module_name}")
            return False

        action = str(step.get("action") or "").lower()
        if not action:
            raise ValueError(f"{case_name} step {index} 缺少action")
        if action not in valid_actions:
            raise ValueError(f"{case_name} step {index} 不支持的action: {action}")

        selector = step.get("selector")
        target = step.get("target")
        if selector and selector not in known_elements and not _looks_raw_selector(selector):
            warnings.append(
                f"{case_name}: selector未在元素库中找到，将按原始选择器处理: {selector}"
            )
        if not selector and not target and action not in _NO_SELECTOR_ACTIONS:
            warnings.append(f"{case_name}: step缺少selector/target: {step}")
        if action in _ASSERTION_ACTIONS:
            GenerationHarness._validate_assertion_fields(
                case_name=case_name, index=index, step=step, action=action
            )
            return True
        return False

    @staticmethod
    def _validate_assertion_fields(
        *, case_name: str, index: int, step: dict[str, Any], action: str
    ) -> None:
        has_selector = bool(step.get("selector") or step.get("target"))
        has_expected = step.get("value") is not None or step.get("expected") is not None
        if action in _TEXT_ASSERTIONS:
            if not has_selector or not has_expected:
                raise ValueError(
                    f"{case_name} step {index} 断言格式错误: {action} 需要 selector/target 和 value/expected"
                )
        elif action in _VISIBILITY_ASSERTIONS:
            if not has_selector:
                raise ValueError(
                    f"{case_name} step {index} 断言格式错误: {action} 需要 selector/target"
                )
        elif action in _URL_TITLE_ASSERTIONS:
            if not has_expected:
                raise ValueError(
                    f"{case_name} step {index} 断言格式错误: {action} 需要 value/expected"
                )
        elif action in _COUNT_ASSERTIONS:
            if not has_selector or not (has_expected or step.get("expression")):
                raise ValueError(
                    f"{case_name} step {index} 断言格式错误: {action} 需要 selector/target 和 value/expected/expression"
                )
        elif action in _ATTRIBUTE_ASSERTIONS:
            if not has_selector or not step.get("attribute") or not has_expected:
                raise ValueError(
                    f"{case_name} step {index} 断言格式错误: {action} 需要 selector/target、attribute 和 value/expected"
                )
        elif action in _MULTI_VALUE_ASSERTIONS:
            has_values = step.get("value") is not None or step.get("expected_values") is not None
            if not has_selector or not has_values:
                raise ValueError(
                    f"{case_name} step {index} 断言格式错误: {action} 需要 selector/target 和 value/expected_values"
                )


_ACTION_ALIASES = {
    "assert_contain_text": "assert_text_contains",
    "assert_contains_text": "assert_text_contains",
    "assert_text_contain": "assert_text_contains",
    "assert_contains": "assert_text_contains",
    "verify_text": "assert_text",
    "verify_contain_text": "assert_text_contains",
    "verify_contains_text": "assert_text_contains",
    "verify_text_contains": "assert_text_contains",
    "verify_visible": "assert_visible",
    "input": "fill",
    "type_text": "fill",
    "navigate": "goto",
    "open": "goto",
}

def _lower_actions(*groups: list[str]) -> set[str]:
    return {action.lower() for group in groups for action in group}


_NO_SELECTOR_ACTIONS = _lower_actions(StepAction.NO_SELECTOR_ACTIONS)
_TEXT_ASSERTIONS = _lower_actions(
    StepAction.HARD_ASSERT_TEXT,
    StepAction.ASSERT_TEXT,
    StepAction.ASSERT_TEXT_CONTAINS,
    StepAction.ASSERT_VALUE,
)
_VISIBILITY_ASSERTIONS = _lower_actions(
    StepAction.ASSERT_VISIBLE,
    StepAction.ASSERT_BE_HIDDEN,
    StepAction.ASSERT_EXISTS,
    StepAction.ASSERT_NOT_EXISTS,
    StepAction.ASSERT_ENABLED,
    StepAction.ASSERT_DISABLED,
)
_URL_TITLE_ASSERTIONS = _lower_actions(
    StepAction.ASSERT_URL,
    StepAction.ASSERT_URL_CONTAINS,
    StepAction.ASSERT_TITLE,
)
_COUNT_ASSERTIONS = _lower_actions(StepAction.ASSERT_ELEMENT_COUNT)
_ATTRIBUTE_ASSERTIONS = _lower_actions(StepAction.ASSERT_ATTRIBUTE)
_MULTI_VALUE_ASSERTIONS = _lower_actions(StepAction.ASSERT_HAVE_VALUES)
_ASSERTION_ACTIONS = (
    _TEXT_ASSERTIONS
    | _VISIBILITY_ASSERTIONS
    | _URL_TITLE_ASSERTIONS
    | _COUNT_ASSERTIONS
    | _ATTRIBUTE_ASSERTIONS
    | _MULTI_VALUE_ASSERTIONS
)


def _normalize_verify_step(step: dict[str, Any]) -> dict[str, Any]:
    action = str(step.get("action") or "").lower()
    if action not in {"verify", "验证"}:
        return step
    value = step.get("value")
    value_text = str(value).strip() if value is not None else ""
    if value_text.lower() in {"visible", "可见"}:
        step["action"] = "assert_visible"
        step.pop("value", None)
        return step
    for prefix in ("contain:", "contains:", "包含:"):
        if value_text.lower().startswith(prefix):
            step["action"] = "assert_text_contains"
            step["value"] = value_text[len(prefix) :]
            return step
    step["action"] = "assert_text"
    return step


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


def _safe_case_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_\u4e00-\u9fff]+", "_", value.strip()).strip("_")
    return name or "test_generated"


def _looks_raw_selector(value: str) -> bool:
    value = str(value).strip()
    return value.startswith(("#", ".", "//", "(//", "text=", "[", "css=", "xpath="))
