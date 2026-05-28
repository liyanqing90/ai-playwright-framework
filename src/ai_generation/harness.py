from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.ai_generation.project_context import ProjectContext
from src.step_actions.action_types import StepAction


_VALID_MODES = {"strict", "smart"}
_DEFAULT_GENERATED_MODE = "smart"


@dataclass
class GenerationHarness:
    context: ProjectContext
    spec: dict[str, Any]
    output_name: str | None = None

    def normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = self._normalize_top_level(payload)
        raw_elements = payload.get("elements") or {}
        semantic_element_aliases = _semantic_element_aliases(raw_elements)
        payload["elements"] = self._normalize_elements(raw_elements)
        selector_element_aliases = set(self.context.elements) | {
            key
            for key, value in payload["elements"].items()
            if isinstance(value, str) and value.strip()
        }
        payload["modules"] = self._normalize_modules(
            payload.get("modules") or {},
            selector_element_aliases=selector_element_aliases,
            semantic_element_aliases=semantic_element_aliases,
        )
        self._normalize_cases_and_data(payload)
        for case_data in payload["data"].values():
            if isinstance(case_data, dict):
                case_data["steps"] = self._normalize_steps(
                    case_data.get("steps") or [],
                    selector_element_aliases=selector_element_aliases,
                    semantic_element_aliases=semantic_element_aliases,
                )
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
        known_variables = set(self.context.variables) | set(payload.get("vars") or {})
        generated_module_refs = {
            module_name: _extract_variable_refs(module_steps)
            for module_name, module_steps in self.context.modules.items()
        }
        generated_module_refs.update(
            {
                module_name: _extract_variable_refs(module_steps)
                for module_name, module_steps in (payload.get("modules") or {}).items()
            }
        )

        for module_name, module_steps in (payload.get("modules") or {}).items():
            if not isinstance(module_steps, list) or not module_steps:
                raise ValueError(f"modules.{module_name} 必须是非空steps列表")
            for index, step in enumerate(module_steps, start=1):
                self._validate_step(
                    case_name=f"modules.{module_name}",
                    index=index,
                    step=step,
                    case_mode=str(
                        self.spec.get("mode") or _DEFAULT_GENERATED_MODE
                    ).lower(),
                    valid_actions=valid_actions,
                    known_elements=known_elements,
                    known_modules=known_modules,
                    warnings=warnings,
                    known_variables=known_variables
                    | generated_module_refs.get(module_name, set()),
                    generated_module_refs=generated_module_refs,
                )

        for case_name, case_data in (payload.get("data") or {}).items():
            if not isinstance(case_data, dict):
                raise ValueError(f"data.{case_name} 必须是对象")
            mode = str(case_data.get("mode", _DEFAULT_GENERATED_MODE)).lower()
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
                    case_mode=mode,
                    valid_actions=valid_actions,
                    known_elements=known_elements,
                    known_modules=known_modules,
                    warnings=warnings,
                    known_variables=known_variables,
                    generated_module_refs=generated_module_refs,
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
            original_name = raw_case.get("name")
            raw_name = _coerce_generated_string(
                original_name,
                field_path="cases[].name",
                preferred_keys=("name", "case_name", "id", "value"),
            )
            if not raw_name:
                raise ValueError(f"cases条目缺少name: {raw_case}")
            name = _safe_case_name(str(raw_name))
            raw_data = None
            if isinstance(original_name, str):
                raw_data = payload.get("data", {}).get(original_name)
            raw_data = raw_data or payload.get("data", {}).get(raw_name)
            case_data = dict(raw_data) if isinstance(raw_data, dict) else {}
            description = raw_case.get("description") or self.spec.get("description")
            if "description" not in case_data and description:
                case_data["description"] = description
            case_data.setdefault("mode", self._case_mode(raw_case))
            normalized_cases.append({"name": name})
            normalized_data[name] = case_data
        payload["cases"] = normalized_cases
        payload["data"] = normalized_data

    def _case_mode(self, raw_case: dict[str, Any]) -> str:
        if self.spec.get("mode") and not _has_explicit_steps(self.spec):
            return str(self.spec["mode"]).lower()
        return str(
            raw_case.get("mode") or self.spec.get("mode") or _DEFAULT_GENERATED_MODE
        ).lower()

    def _normalize_steps(
        self,
        steps: list[Any],
        *,
        selector_element_aliases: set[str] | None = None,
        semantic_element_aliases: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError(f"生成结果中的step必须是对象: {step}")
            item = dict(step)
            item = _normalize_step_scalar_fields(item)
            module_name = self._module_name(item)
            if module_name:
                module_step: dict[str, Any] = {"use_module": module_name}
                if "params" in item:
                    module_step["params"] = _normalize_variable_syntax(item["params"])
                if "description" in item:
                    module_step["description"] = item["description"]
                normalized.append(module_step)
                continue

            action = str(item.get("action") or "").lower()
            if action:
                item["action"] = _ACTION_ALIASES.get(action, item.get("action"))
                item = _normalize_action_fields(item)
                item = _normalize_verify_step(item)
            item = _normalize_variable_syntax(item)
            item = _normalize_step_element_references(
                item,
                selector_element_aliases=selector_element_aliases or set(),
                semantic_element_aliases=semantic_element_aliases or {},
            )
            normalized.append(item)
        return normalized

    @staticmethod
    def _normalize_elements(elements: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in elements.items():
            normalized_key = _coerce_generated_string(
                key,
                field_path="elements key",
                preferred_keys=("key", "name", "id", "value"),
            )
            if isinstance(value, dict) and value.get("selector"):
                normalized[normalized_key] = _coerce_generated_string(
                    value["selector"],
                    field_path=f"elements.{normalized_key}.selector",
                    preferred_keys=("selector", "css", "xpath", "value", "text"),
                )
            elif isinstance(value, dict):
                continue
            else:
                normalized[normalized_key] = value
        return normalized

    def _normalize_modules(
        self,
        modules: dict[str, Any],
        *,
        selector_element_aliases: set[str] | None = None,
        semantic_element_aliases: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for module_name, raw_module in modules.items():
            module_name = _coerce_generated_string(
                module_name,
                field_path="modules key",
                preferred_keys=("name", "module", "key", "value"),
            )
            if isinstance(raw_module, dict) and "steps" in raw_module:
                raw_steps = raw_module.get("steps") or []
            elif isinstance(raw_module, list):
                raw_steps = raw_module
            else:
                raise ValueError(
                    f"modules.{module_name} 必须是steps列表，或包含steps字段的对象"
                )
            normalized[module_name] = self._normalize_steps(
                raw_steps,
                selector_element_aliases=selector_element_aliases or set(),
                semantic_element_aliases=semantic_element_aliases or {},
            )
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
                or step.get("target")
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
        case_mode: str,
        valid_actions: set[str],
        known_elements: set[str],
        known_modules: set[str],
        warnings: list[str],
        known_variables: set[str],
        generated_module_refs: dict[str, set[str]],
    ) -> bool:
        if not isinstance(step, dict):
            raise ValueError(f"{case_name} step {index} 必须是对象")
        module_name = step.get("use_module")
        if module_name:
            if module_name not in known_modules:
                raise ValueError(f"{case_name} 引用了不存在的公共组件: {module_name}")
            GenerationHarness._validate_module_params(
                case_name=case_name,
                index=index,
                step=step,
                module_name=module_name,
                known_variables=known_variables,
                generated_module_refs=generated_module_refs,
            )
            return False

        action = str(step.get("action") or "").lower()
        if not action:
            raise ValueError(f"{case_name} step {index} 缺少action")
        if action not in valid_actions:
            raise ValueError(f"{case_name} step {index} 不支持的action: {action}")

        selector = step.get("selector")
        target = step.get("target")
        effective_mode = str(
            step.get("mode") or case_mode or _DEFAULT_GENERATED_MODE
        ).lower()
        if effective_mode not in _VALID_MODES:
            raise ValueError(f"{case_name} step {index} mode 不合法: {effective_mode}")
        if (
            selector
            and selector not in known_elements
            and not _looks_raw_selector(selector)
        ):
            warnings.append(
                f"{case_name}: selector未在元素库中找到，将按原始选择器处理: {selector}"
            )
        if not selector and not target and action not in _NO_SELECTOR_ACTIONS:
            raise ValueError(f"{case_name} step {index} 缺少selector/target: {step}")
        if (
            target
            and not selector
            and action not in _NO_SELECTOR_ACTIONS
            and effective_mode == "strict"
        ):
            raise ValueError(
                f"{case_name} step {index} 使用target时必须声明 mode: smart，或在data用例层声明 mode: smart"
            )
        if action in _ASSERTION_ACTIONS:
            GenerationHarness._validate_assertion_fields(
                case_name=case_name, index=index, step=step, action=action
            )
        GenerationHarness._validate_variable_refs(
            case_name=case_name,
            index=index,
            value=step,
            known_variables=known_variables,
        )
        if action in _ASSERTION_ACTIONS:
            return True
        return False

    @staticmethod
    def _validate_assertion_fields(
        *, case_name: str, index: int, step: dict[str, Any], action: str
    ) -> None:
        has_selector = bool(step.get("selector") or step.get("target"))
        has_expected = step.get("value") is not None or step.get("expected") is not None
        if _is_empty_expected(step.get("value")) or _is_empty_expected(
            step.get("expected")
        ):
            raise ValueError(f"{case_name} step {index} 断言期望值不能为空: {action}")
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
            has_values = (
                step.get("value") is not None or step.get("expected_values") is not None
            )
            if not has_selector or not has_values:
                raise ValueError(
                    f"{case_name} step {index} 断言格式错误: {action} 需要 selector/target 和 value/expected_values"
                )

    @staticmethod
    def _validate_module_params(
        *,
        case_name: str,
        index: int,
        step: dict[str, Any],
        module_name: str,
        known_variables: set[str],
        generated_module_refs: dict[str, set[str]],
    ) -> None:
        params = step.get("params") or {}
        if "params" in step and not isinstance(params, dict):
            raise ValueError(f"{case_name} step {index} params 必须是对象")
        for param_value in params.values():
            GenerationHarness._validate_variable_refs(
                case_name=case_name,
                index=index,
                value=param_value,
                known_variables=known_variables,
            )
        required_params = (
            generated_module_refs.get(module_name, set()) - known_variables
        )
        missing = sorted(required_params - set(params))
        if missing:
            raise ValueError(
                f"{case_name} step {index} 引用模块 {module_name} 缺少params: {', '.join(missing)}"
            )

    @staticmethod
    def _validate_variable_refs(
        *,
        case_name: str,
        index: int,
        value: Any,
        known_variables: set[str],
    ) -> None:
        refs = _extract_variable_refs(value)
        unknown = sorted(ref for ref in refs if ref not in known_variables)
        if unknown:
            raise ValueError(
                f"{case_name} step {index} 引用了未定义变量: {', '.join(unknown)}"
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


def _is_empty_expected(value: Any) -> bool:
    return value is not None and isinstance(value, str) and not value.strip()


def _normalize_step_scalar_fields(step: dict[str, Any]) -> dict[str, Any]:
    field_keys: dict[str, tuple[str, ...]] = {
        "action": ("action", "type", "name", "value"),
        "selector": (
            "key",
            "element_key",
            "name",
            "selector",
            "css",
            "xpath",
            "value",
            "text",
            "id",
        ),
        "target": ("target", "name", "text", "value", "description"),
        "mode": ("mode", "value", "name"),
        "attribute": ("attribute", "name", "value"),
        "use_module": ("use_module", "module", "name", "key", "value"),
        "module": ("module", "use_module", "name", "key", "value"),
    }
    for field, preferred_keys in field_keys.items():
        if field in step and step[field] is not None:
            step[field] = _coerce_generated_string(
                step[field],
                field_path=f"step.{field}",
                preferred_keys=preferred_keys,
            )
    return step


def _coerce_generated_string(
    value: Any, *, field_path: str, preferred_keys: tuple[str, ...]
) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for key in preferred_keys:
            candidate = value.get(key)
            if isinstance(candidate, (str, int, float, bool)):
                return str(candidate)
        raise ValueError(
            f"{field_path} 必须是字符串，模型返回对象缺少可用字段: {value}"
        )
    raise ValueError(f"{field_path} 必须是字符串，实际类型: {type(value).__name__}")


def _semantic_element_aliases(elements: dict[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for key, value in (elements or {}).items():
        if not isinstance(value, dict) or value.get("selector"):
            continue
        name = _coerce_generated_string(
            key,
            field_path="elements key",
            preferred_keys=("key", "name", "id", "value"),
        )
        target = _first_generated_string(
            value,
            preferred_keys=("target", "description", "desc", "label", "text", "name"),
        )
        if name and target:
            aliases[name] = target
    return aliases


def _normalize_step_element_references(
    step: dict[str, Any],
    *,
    selector_element_aliases: set[str],
    semantic_element_aliases: dict[str, str],
) -> dict[str, Any]:
    target = step.get("target")
    if not isinstance(target, str) or not target.strip():
        return step
    if target in selector_element_aliases and not step.get("selector"):
        step["selector"] = target
        step.pop("target", None)
    elif target in semantic_element_aliases:
        step["target"] = semantic_element_aliases[target]
    return step


def _first_generated_string(
    value: dict[str, Any],
    *,
    preferred_keys: tuple[str, ...],
) -> str:
    for key in preferred_keys:
        raw = value.get(key)
        if raw is None:
            continue
        text = _coerce_generated_string(
            raw,
            field_path=key,
            preferred_keys=preferred_keys,
        )
        text = str(text or "").strip()
        if text:
            return text
    return ""


def _extract_variable_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, str):
        refs.update(
            match.group(1) or match.group(2)
            for match in re.finditer(r"\$\{([^{}]+)\}|\$<([^<>]+)>", value)
        )
        return {ref.strip() for ref in refs if ref and ref.strip()}
    if isinstance(value, list):
        for item in value:
            refs.update(_extract_variable_refs(item))
    elif isinstance(value, dict):
        for item in value.values():
            refs.update(_extract_variable_refs(item))
    return refs


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
    StepAction.ASSERT_TITLE_CONTAINS,
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


def _normalize_action_fields(step: dict[str, Any]) -> dict[str, Any]:
    action = str(step.get("action") or "").lower()
    if action in _lower_actions(StepAction.NAVIGATE):
        _move_first_present(step, aliases=("url", "href", "link"), target="value")
    if action in _lower_actions(StepAction.FILL, StepAction.TYPE):
        _move_first_present(step, aliases=("text", "input"), target="value")
    return step


def _move_first_present(
    step: dict[str, Any], *, aliases: tuple[str, ...], target: str
) -> None:
    if step.get(target) is None:
        for alias in aliases:
            if step.get(alias) is not None:
                step[target] = step[alias]
                break
    for alias in aliases:
        step.pop(alias, None)


def _normalize_variable_syntax(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}", r"${\1}", value)
    if isinstance(value, list):
        return [_normalize_variable_syntax(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_variable_syntax(item) for key, item in value.items()}
    return value


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
    name = name or "generated"
    return name if name.startswith("test_") else f"test_{name}"


def _looks_raw_selector(value: str) -> bool:
    value = str(value).strip()
    return value.startswith(("#", ".", "//", "(//", "text=", "[", "css=", "xpath="))
