from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit
from typing import Any

from ai_playwright.ai_generation.project_context import ProjectContext
from ai_playwright.step_actions.action_registry import (
    NO_SELECTOR_ACTIONS,
    VALID_ACTIONS,
)
from ai_playwright.step_actions.action_types import StepAction


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
        element_scope_aliases = self._element_scope_aliases(payload, raw_elements)
        payload["elements"], element_key_aliases = self._normalize_elements(
            raw_elements,
            element_scope_aliases=element_scope_aliases,
        )
        element_target_aliases = _element_target_aliases(
            raw_elements=raw_elements,
            context_elements=self.context.elements,
            normalized_elements=payload["elements"],
            element_key_aliases=element_key_aliases,
        )
        semantic_element_aliases = {
            element_key_aliases.get(key, key): target
            for key, target in semantic_element_aliases.items()
        }
        selector_element_aliases = set(self.context.elements) | {
            key
            for key, value in payload["elements"].items()
            if isinstance(value, str) and value.strip()
        }
        payload["modules"] = self._normalize_modules(
            payload.get("modules") or {},
            selector_element_aliases=selector_element_aliases,
            semantic_element_aliases=semantic_element_aliases,
            element_key_aliases=element_key_aliases,
            element_target_aliases=element_target_aliases,
        )
        self._normalize_cases_and_data(payload)
        module_variable_refs = _module_variable_refs(
            self.context.modules, payload["modules"]
        )
        for case_name, case_data in payload["data"].items():
            if isinstance(case_data, dict):
                case_data["steps"] = self._normalize_steps(
                    case_data.get("steps") or [],
                    selector_element_aliases=selector_element_aliases,
                    semantic_element_aliases=semantic_element_aliases,
                    element_key_aliases=element_key_aliases,
                    element_target_aliases=element_target_aliases,
                )
                _align_module_params_with_spec_inputs(
                    case_data["steps"],
                    module_variable_refs=module_variable_refs,
                    spec_inputs=self._spec_inputs_for_case(case_name),
                )
                _inline_spec_input_values(
                    case_data["steps"],
                    spec_inputs=self._spec_inputs_for_case(case_name),
                    known_variables=set(self.context.variables)
                    | set(payload.get("vars") or {}),
                    module_param_names=_module_param_names(case_data["steps"]),
                )
                _drop_unsupported_url_assertions(
                    case_data["steps"],
                    spec=self.spec,
                    context=self.context,
                )
                _normalize_visible_title_assertions(
                    case_data["steps"],
                    element_selectors={**self.context.elements, **payload["elements"]},
                )
                _normalize_assertions_against_step_evidence(
                    case_data["steps"],
                    element_selectors={**self.context.elements, **payload["elements"]},
                )
                _ensure_step_semantic_targets(
                    case_data["steps"],
                    selector_element_aliases=selector_element_aliases,
                    element_target_aliases=element_target_aliases,
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
        for module_name in sorted(payload.get("modules") or {}):
            if _is_composite_module_name(module_name):
                raise ValueError(
                    f"modules.{module_name} 颗粒度过粗：module 只能表达单一业务意图，"
                    "不要用 and/并/和 组合多个流程"
                )
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
        generated_module_case_refs = _generated_module_case_reference_counts(payload)

        for module_name, module_steps in (payload.get("modules") or {}).items():
            if not isinstance(module_steps, list) or not module_steps:
                raise ValueError(f"modules.{module_name} 必须是非空steps列表")
            if (
                module_name not in self.context.modules
                and _is_single_step_generated_module(module_steps)
            ):
                raise ValueError(
                    f"modules.{module_name} single-step module is unnecessary; "
                    "inline one-step goto/click/fill/assert actions in data.<case>.steps"
                )
            if (
                module_name not in self.context.modules
                and generated_module_case_refs.get(module_name, 0) < 2
            ):
                raise ValueError(
                    f"modules.{module_name} has no actual reuse value; "
                    "generated modules must be referenced by at least two case steps, "
                    "otherwise inline the steps in data.<case>.steps"
                )
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
        payload["vars"] = {
            key: value
            for key, value in (payload.get("vars") or {}).items()
            if key not in self.context.variables
        }
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
            case_params = _pop_case_level_params(case_data)
            if case_params:
                _apply_case_params_to_module_steps(case_data, case_params)
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

    def _spec_inputs_for_case(self, case_name: str) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        if isinstance(self.spec.get("inputs"), dict):
            inputs.update(_normalize_variable_syntax(self.spec["inputs"]))
        for raw_case in self.spec.get("cases") or []:
            if not isinstance(raw_case, dict):
                continue
            raw_name = str(raw_case.get("name") or "").strip()
            if raw_name and raw_name not in {case_name, _safe_case_name(raw_name)}:
                continue
            if isinstance(raw_case.get("inputs"), dict):
                inputs.update(_normalize_variable_syntax(raw_case["inputs"]))
        return inputs

    def _normalize_steps(
        self,
        steps: list[Any],
        *,
        selector_element_aliases: set[str] | None = None,
        semantic_element_aliases: dict[str, str] | None = None,
        element_key_aliases: dict[str, str] | None = None,
        element_target_aliases: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError(f"生成结果中的step必须是对象: {step}")
            item = dict(step)
            item = _normalize_step_scalar_fields(item)
            module_name = self._module_name(item)
            if not module_name:
                module_name = self._single_module_name_for_action(item)
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
                element_key_aliases=element_key_aliases or {},
                element_target_aliases=element_target_aliases or {},
            )
            _ensure_step_semantic_target(
                item,
                selector_element_aliases=selector_element_aliases or set(),
                element_target_aliases=element_target_aliases or {},
            )
            normalized.append(item)
        return normalized

    def _normalize_elements(
        self,
        elements: dict[str, Any],
        *,
        element_scope_aliases: dict[str, str],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        normalized: dict[str, Any] = {}
        aliases: dict[str, str] = {}
        used_keys = set(self.context.elements)
        for key, value in elements.items():
            raw_key = _coerce_generated_string(
                key,
                field_path="elements key",
                preferred_keys=("key", "name", "id", "value"),
            )
            if isinstance(value, dict) and value.get("selector"):
                normalized_key = self._generated_element_key(
                    raw_key,
                    used_keys,
                    page_scope=element_scope_aliases.get(raw_key),
                )
                aliases[raw_key] = normalized_key
                normalized[normalized_key] = _coerce_generated_string(
                    value["selector"],
                    field_path=f"elements.{normalized_key}.selector",
                    preferred_keys=("selector", "css", "xpath", "value", "text"),
                )
                used_keys.add(normalized_key)
            elif isinstance(value, dict):
                continue
            else:
                normalized_key = self._generated_element_key(
                    raw_key,
                    used_keys,
                    page_scope=element_scope_aliases.get(raw_key),
                )
                aliases[raw_key] = normalized_key
                normalized[normalized_key] = value
                used_keys.add(normalized_key)
        return normalized, aliases

    def _generated_element_key(
        self,
        raw_key: str,
        used_keys: set[str],
        *,
        page_scope: str | None,
    ) -> str:
        base_key = _safe_element_key(raw_key)
        if base_key not in used_keys:
            return base_key
        scope = _safe_element_key(page_scope or "page")
        return _unique_element_key(f"{base_key}_{scope}", used_keys)

    def _element_scope_aliases(
        self,
        payload: dict[str, Any],
        raw_elements: dict[str, Any],
    ) -> dict[str, str]:
        scopes: dict[str, tuple[str, int]] = {}
        raw_keys = {
            _coerce_generated_string(
                key,
                field_path="elements key",
                preferred_keys=("key", "name", "id", "value"),
            )
            for key in raw_elements
        }
        for raw_key, value in raw_elements.items():
            normalized_key = _coerce_generated_string(
                raw_key,
                field_path="elements key",
                preferred_keys=("key", "name", "id", "value"),
            )
            explicit_scope = _scope_from_element_definition(value)
            if explicit_scope:
                _record_element_scope(scopes, normalized_key, explicit_scope, 50)

        for step_list in _payload_step_lists(payload):
            self._collect_step_scopes(step_list, raw_keys=raw_keys, scopes=scopes)
        return {key: value for key, (value, _priority) in scopes.items()}

    def _collect_step_scopes(
        self,
        steps: list[Any],
        *,
        raw_keys: set[str],
        scopes: dict[str, tuple[str, int]],
    ) -> None:
        current_scope: tuple[str, int] | None = None
        for step in steps:
            if not isinstance(step, dict):
                continue
            marker = _scope_marker_from_step(step)
            if marker:
                current_scope = marker
            if current_scope:
                for key in _step_element_references(step, raw_keys):
                    _record_element_scope(
                        scopes,
                        key,
                        current_scope[0],
                        current_scope[1],
                    )

        next_scope: tuple[str, int] | None = None
        for step in reversed(steps):
            if not isinstance(step, dict):
                continue
            marker = _scope_marker_from_step(step)
            if marker:
                next_scope = marker
            if next_scope:
                for key in _step_element_references(step, raw_keys):
                    _record_element_scope(
                        scopes,
                        key,
                        next_scope[0],
                        next_scope[1],
                    )

    def _normalize_modules(
        self,
        modules: dict[str, Any],
        *,
        selector_element_aliases: set[str] | None = None,
        semantic_element_aliases: dict[str, str] | None = None,
        element_key_aliases: dict[str, str] | None = None,
        element_target_aliases: dict[str, str] | None = None,
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
                element_key_aliases=element_key_aliases or {},
                element_target_aliases=element_target_aliases or {},
            )
        return normalized

    @staticmethod
    def _module_name(step: dict[str, Any]) -> str | None:
        if "use_module" in step:
            return step.get("use_module")
        if "module" in step and "action" not in step:
            return step.get("module")
        if "module_name" in step:
            return step.get("module_name")
        action = str(step.get("action") or "").lower()
        if action in {"module", "use_module", "reuse_module", "call_module"}:
            return (
                step.get("use_module")
                or step.get("module_name")
                or step.get("module")
                or step.get("target")
                or step.get("value")
                or step.get("name")
            )
        return None

    def _single_module_name_for_action(self, step: dict[str, Any]) -> str | None:
        action = str(step.get("action") or "").lower()
        if action not in {"module", "use_module", "reuse_module", "call_module"}:
            return None
        module_names = list((self.context.modules or {}).keys())
        if len(module_names) != 1:
            return None
        return str(module_names[0])

    @staticmethod
    def _valid_actions() -> set[str]:
        return set(VALID_ACTIONS)

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
            if _is_composite_module_name(module_name):
                raise ValueError(
                    f"{case_name} step {index} module颗粒度过粗: {module_name}；"
                    "请拆成多个单一业务意图module或直接action步骤"
                )
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
            and not isinstance(selector, dict)
            and selector not in known_elements
            and not _looks_raw_selector(selector)
        ):
            raise ValueError(
                f"{case_name} step {index} selector 未在 elements 中定义，"
                f"也不像合法原始 selector: {selector}"
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
        elif action in _VALUE_ASSERTIONS:
            if not has_selector or not has_expected:
                raise ValueError(
                    f"{case_name} step {index} 断言格式错误: {action} 需要 selector/target 和 value/expected"
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


def _is_composite_module_name(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    normalized = re.sub(r"[\s\-]+", "_", text.lower())
    if re.search(r"(?:^|_)and(?:_|$)|(?:^|_)then(?:_|$)", normalized):
        return True
    return any(token in text for token in ("并", "和", "及", "与"))


def _is_single_step_generated_module(steps: Any) -> bool:
    if not isinstance(steps, list):
        return False
    executable_steps = [
        step
        for step in steps
        if isinstance(step, dict)
        and (step.get("action") or step.get("selector") or step.get("target"))
    ]
    return len(executable_steps) <= 1


def _generated_module_case_reference_counts(payload: dict[str, Any]) -> dict[str, int]:
    generated_modules = {
        str(name) for name in (payload.get("modules") or {}) if str(name).strip()
    }
    counts = {name: 0 for name in generated_modules}
    for case_data in (payload.get("data") or {}).values():
        if not isinstance(case_data, dict):
            continue
        steps = case_data.get("steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            module_name = step.get("use_module")
            if isinstance(module_name, str) and module_name in counts:
                counts[module_name] += 1
    return counts


def _element_target_aliases(
    *,
    raw_elements: dict[str, Any],
    context_elements: dict[str, Any],
    normalized_elements: dict[str, Any],
    element_key_aliases: dict[str, str],
) -> dict[str, str]:
    aliases: dict[str, str] = {
        str(key): _target_from_element_key(str(key))
        for key in (context_elements or {})
        if str(key).strip()
    }
    aliases.update(
        {
            str(key): _target_from_element_key(str(key))
            for key in (normalized_elements or {})
            if str(key).strip()
        }
    )
    for raw_key, value in (raw_elements or {}).items():
        key = _coerce_generated_string(
            raw_key,
            field_path="elements key",
            preferred_keys=("key", "name", "id", "value"),
        )
        if not key:
            continue
        normalized_key = element_key_aliases.get(key, key)
        target = ""
        if isinstance(value, dict):
            preferred_keys = (
                ("target", "description", "label", "text", "name")
                if value.get("selector")
                else ("target", "description", "desc", "label", "text", "name")
            )
            target = _first_generated_string(
                value,
                preferred_keys=preferred_keys,
            )
        if not target:
            target = _target_from_element_key(normalized_key)
        aliases[normalized_key] = target
    return {key: value for key, value in aliases.items() if value}


def _normalize_step_element_references(
    step: dict[str, Any],
    *,
    selector_element_aliases: set[str],
    semantic_element_aliases: dict[str, str],
    element_key_aliases: dict[str, str],
    element_target_aliases: dict[str, str],
) -> dict[str, Any]:
    selector = step.get("selector")
    if isinstance(selector, str) and selector in element_key_aliases:
        step["selector"] = element_key_aliases[selector]

    target = step.get("target")
    if not isinstance(target, str) or not target.strip():
        return step
    if target in element_key_aliases:
        normalized_key = element_key_aliases[target]
        if not step.get("selector"):
            step["selector"] = normalized_key
        step["target"] = element_target_aliases.get(
            normalized_key
        ) or _target_from_element_key(normalized_key)
    elif target in semantic_element_aliases:
        step["target"] = semantic_element_aliases[target]
    elif target in selector_element_aliases:
        if not step.get("selector"):
            step["selector"] = target
        step["target"] = element_target_aliases.get(target) or _target_from_element_key(
            target
        )
    return step


def _ensure_step_semantic_targets(
    steps: list[dict[str, Any]],
    *,
    selector_element_aliases: set[str],
    element_target_aliases: dict[str, str],
) -> None:
    for step in steps:
        if isinstance(step, dict):
            _ensure_step_semantic_target(
                step,
                selector_element_aliases=selector_element_aliases,
                element_target_aliases=element_target_aliases,
            )


def _ensure_step_semantic_target(
    step: dict[str, Any],
    *,
    selector_element_aliases: set[str],
    element_target_aliases: dict[str, str],
) -> None:
    action = str(step.get("action") or "").lower()
    if not action or action in _NO_SELECTOR_ACTIONS:
        return
    selector = step.get("selector")
    if not selector or isinstance(selector, dict):
        return
    target = step.get("target")
    if _is_usable_semantic_target(target, selector_element_aliases):
        return
    candidate = _target_from_step_description(step)
    if not candidate:
        candidate = element_target_aliases.get(str(selector).strip())
    if not candidate:
        candidate = _target_from_selector(selector)
    if not candidate:
        candidate = _target_from_element_key(str(selector))
    if candidate:
        step["target"] = candidate


def _is_usable_semantic_target(
    target: Any,
    selector_element_aliases: set[str],
) -> bool:
    text = str(target or "").strip()
    if not text:
        return False
    if text in selector_element_aliases:
        return False
    if _looks_raw_selector(text):
        return False
    return True


def _target_from_step_description(step: dict[str, Any]) -> str:
    description = str(step.get("description") or "").strip()
    if not description or _looks_raw_selector(description):
        return ""
    if len(description) > 120:
        return ""
    return description


def _target_from_selector(value: Any) -> str:
    selector = str(value or "").strip()
    if not selector:
        return ""
    text = _selector_text_query(selector, element_selectors={})
    if text:
        return text
    attr_match = re.search(
        r"\[(?:aria-label|title|placeholder|name|data-test|data-testid|data-qa|"
        r"data-cy|data-ui|id)[^\]]*=\s*(['\"])(.*?)\1",
        selector,
        flags=re.IGNORECASE,
    )
    if attr_match:
        return _target_from_identifier(attr_match.group(2))
    if selector.startswith("#"):
        return _target_from_identifier(selector[1:])
    return ""


def _target_from_element_key(value: str) -> str:
    return _target_from_identifier(value)


def _target_from_identifier(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"^(?:css|xpath|text)=", "", text, flags=re.IGNORECASE)
    text = text.strip(" \"'[]()（）:：,，.。;；#.")
    if not text:
        return ""
    replacements = {
        "btn": "button",
        "cta": "button",
        "ipt": "input",
        "pwd": "password",
        "spu": "SPU",
        "sku": "SKU",
        "id": "ID",
    }
    chunks = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", text.replace("-", "_"))
    words: list[str] = []
    for chunk in chunks:
        for part in re.split(r"_+", chunk):
            if not part:
                continue
            camel_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", part)
            for item in camel_parts or [part]:
                lowered = item.lower()
                words.append(replacements.get(lowered, item))
    result = " ".join(words).strip()
    return result or text


def _payload_step_lists(payload: dict[str, Any]) -> list[list[Any]]:
    result: list[list[Any]] = []
    for raw_module in (payload.get("modules") or {}).values():
        if isinstance(raw_module, dict) and isinstance(raw_module.get("steps"), list):
            result.append(raw_module["steps"])
        elif isinstance(raw_module, list):
            result.append(raw_module)
    for case_data in (payload.get("data") or {}).values():
        if isinstance(case_data, dict) and isinstance(case_data.get("steps"), list):
            result.append(case_data["steps"])
    return result


def _pop_case_level_params(case_data: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key in ("inputs", "params"):
        raw = case_data.pop(key, None)
        if isinstance(raw, dict):
            params.update(_normalize_variable_syntax(raw))
    return params


def _apply_case_params_to_module_steps(
    case_data: dict[str, Any],
    params: dict[str, Any],
) -> None:
    steps = case_data.get("steps")
    if not isinstance(steps, list):
        return
    for step in steps:
        if not isinstance(step, dict) or "params" in step:
            continue
        action = str(step.get("action") or "").lower()
        if step.get("use_module") or (step.get("module") and not action):
            step["params"] = dict(params)


def _module_variable_refs(
    context_modules: dict[str, Any],
    generated_modules: dict[str, Any],
) -> dict[str, set[str]]:
    refs = {
        module_name: _extract_variable_refs(module_steps)
        for module_name, module_steps in (context_modules or {}).items()
    }
    refs.update(
        {
            module_name: _extract_variable_refs(module_steps)
            for module_name, module_steps in (generated_modules or {}).items()
        }
    )
    return refs


def _align_module_params_with_spec_inputs(
    steps: list[dict[str, Any]],
    *,
    module_variable_refs: dict[str, set[str]],
    spec_inputs: dict[str, Any],
) -> None:
    if not isinstance(spec_inputs, dict) or not spec_inputs:
        return
    for step in steps:
        if not isinstance(step, dict) or not step.get("use_module"):
            continue
        module_name = str(step.get("use_module") or "")
        required = module_variable_refs.get(module_name, set())
        if not required:
            continue
        params = step.get("params")
        if not isinstance(params, dict):
            params = {}
        params = dict(params)
        for name in sorted(required):
            if name in spec_inputs:
                params[name] = spec_inputs[name]
        if params:
            step["params"] = params


def _module_param_names(steps: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("params"), dict):
            continue
        names.update(str(key) for key in step["params"])
    return names


def _inline_spec_input_values(
    value: Any,
    *,
    spec_inputs: dict[str, Any],
    known_variables: set[str],
    module_param_names: set[str],
) -> Any:
    if not isinstance(spec_inputs, dict) or not spec_inputs:
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _inline_spec_input_values(
                item,
                spec_inputs=spec_inputs,
                known_variables=known_variables,
                module_param_names=module_param_names,
            )
        return value
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key == "params":
                continue
            value[key] = _inline_spec_input_values(
                item,
                spec_inputs=spec_inputs,
                known_variables=known_variables,
                module_param_names=module_param_names,
            )
        return value
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"\$\{([^{}]+)\}|\$<([^<>]+)>", value.strip())
    if not match:
        return value
    name = (match.group(1) or match.group(2) or "").strip()
    if not name or name in known_variables or name in module_param_names:
        return value
    input_value = _spec_input_value(spec_inputs, name)
    if input_value is _MISSING:
        return value
    return input_value


_MISSING = object()


def _spec_input_value(spec_inputs: dict[str, Any], name: str) -> Any:
    if name in spec_inputs:
        return spec_inputs[name]
    current: Any = spec_inputs
    for part in name.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _drop_unsupported_url_assertions(
    steps: list[dict[str, Any]],
    *,
    spec: dict[str, Any],
    context: ProjectContext,
) -> None:
    allowed_terms = _url_assertion_terms_from_spec(spec=spec, context=context)
    kept: list[dict[str, Any]] = []
    for step in steps:
        action = str((step or {}).get("action") or "").lower()
        if action not in {"assert_url", "assert_url_contains"}:
            kept.append(step)
            continue
        expected = str(step.get("value") or step.get("expected") or "").strip()
        if _looks_like_display_text_url_assertion(expected):
            step["action"] = "assert_title_contains"
            if "expected" in step and "value" not in step:
                step["value"] = step.pop("expected")
            kept.append(step)
            continue
        if expected and any(expected.lower() in term for term in allowed_terms):
            kept.append(step)
            continue
    steps[:] = kept


def _normalize_assertions_against_step_evidence(
    steps: list[dict[str, Any]],
    *,
    element_selectors: dict[str, Any] | None = None,
) -> None:
    fills: list[dict[str, Any]] = []
    element_selectors = element_selectors or {}
    for step in steps:
        action = str((step or {}).get("action") or "").lower()
        if action in _lower_actions(StepAction.FILL, StepAction.TYPE):
            expected = _step_expected_value(step)
            if expected:
                fills.append(
                    {
                        "selector": step.get("selector"),
                        "target": step.get("target"),
                        "value": expected,
                    }
                )
            continue
        if action not in {"assert_text", "assert_text_contains", "hard_assert"}:
            continue
        expected = _step_expected_value(step)
        if not expected:
            continue
        fill = _matching_fill_for_expected(fills, expected)
        if not fill:
            continue
        if not _text_assertion_should_use_previous_fill_value(
            step,
            fill=fill,
            expected=expected,
            element_selectors=element_selectors,
        ):
            continue
        step["action"] = "assert_value"
        step["value"] = expected
        step.pop("expected", None)
        fill_selector = fill.get("selector")
        fill_target = fill.get("target")
        if fill_selector:
            step["selector"] = fill_selector
            if fill_target:
                step["target"] = fill_target
        elif fill_target:
            step["target"] = fill_target
            step.pop("selector", None)


def _normalize_visible_title_assertions(
    steps: list[dict[str, Any]],
    *,
    element_selectors: dict[str, Any] | None = None,
) -> None:
    title_selector = _visible_title_element_key(element_selectors or {})
    if not title_selector:
        return
    for step in steps:
        action = str((step or {}).get("action") or "").lower()
        if action not in {"assert_title", "assert_title_contains"}:
            continue
        expected = _step_expected_value(step)
        if not expected:
            continue
        step["action"] = (
            "assert_text_contains"
            if action == "assert_title_contains"
            else "assert_text"
        )
        step["selector"] = title_selector
        step["value"] = expected
        step.pop("expected", None)


def _visible_title_element_key(element_selectors: dict[str, Any]) -> str | None:
    for key, value in element_selectors.items():
        if not isinstance(value, str) or not value.strip():
            continue
        normalized = str(key or "").lower()
        if any(term in normalized for term in ("page_title", "title", "heading")):
            return str(key)
    return None


def _looks_like_display_text_url_assertion(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if _urls_from_value(text):
        return False
    if re.search(r"[\u4e00-\u9fff]", text):
        return not re.search(r"[/?#=&%]", text)
    return False


def _step_expected_value(step: dict[str, Any]) -> str:
    expected = step.get("expected")
    if expected is None:
        expected = step.get("value")
    return str(expected).strip() if expected is not None else ""


def _matching_fill_for_expected(
    fills: list[dict[str, Any]],
    expected: str,
) -> dict[str, Any] | None:
    normalized_expected = _assertion_value_key(expected)
    if not normalized_expected:
        return None
    for fill in reversed(fills):
        if _assertion_value_key(fill.get("value")) == normalized_expected:
            return fill
    return None


def _text_assertion_should_use_previous_fill_value(
    step: dict[str, Any],
    *,
    fill: dict[str, Any],
    expected: str,
    element_selectors: dict[str, Any],
) -> bool:
    if _same_locator_reference(step, fill, element_selectors=element_selectors):
        return True
    selector_text = _selector_text_query(
        step.get("selector"),
        element_selectors=element_selectors,
    )
    if not selector_text:
        selector_text = _selector_text_query(
            step.get("target"),
            element_selectors=element_selectors,
        )
    if not selector_text:
        return False
    return _assertion_value_key(selector_text) != _assertion_value_key(expected)


def _same_locator_reference(
    step: dict[str, Any],
    fill: dict[str, Any],
    *,
    element_selectors: dict[str, Any],
) -> bool:
    step_selector = _effective_selector_value(
        step.get("selector"),
        element_selectors=element_selectors,
    )
    fill_selector = _effective_selector_value(
        fill.get("selector"),
        element_selectors=element_selectors,
    )
    if step_selector and fill_selector and step_selector == fill_selector:
        return True
    step_target = str(step.get("target") or "").strip()
    fill_target = str(fill.get("target") or "").strip()
    return bool(step_target and fill_target and step_target == fill_target)


def _effective_selector_value(
    value: Any,
    *,
    element_selectors: dict[str, Any],
) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    resolved = element_selectors.get(text)
    if isinstance(resolved, str) and resolved.strip():
        return resolved.strip()
    return text


def _selector_text_query(
    value: Any,
    *,
    element_selectors: dict[str, Any],
) -> str:
    selector = _effective_selector_value(value, element_selectors=element_selectors)
    if not selector:
        return ""
    if selector.lower().startswith("text="):
        return selector[5:].strip(" \"'")
    for pattern in (
        r"has-text\(\s*(['\"])(.*?)\1\s*\)",
        r"text\(\s*(['\"])(.*?)\1\s*\)",
        r"normalize-space\(\)\s*=\s*(['\"])(.*?)\1",
    ):
        match = re.search(pattern, selector)
        if match:
            return str(match.group(2) or "").strip()
    return ""


def _assertion_value_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if re.search(r"[\u4e00-\u9fff]", text):
        text = re.sub(r"\s+", "", text)
    else:
        text = re.sub(r"\s+", " ", text)
    return text


def _url_assertion_terms_from_spec(
    *,
    spec: dict[str, Any],
    context: ProjectContext,
) -> set[str]:
    terms: set[str] = set()
    for text in _strings_from_value(spec):
        normalized = " ".join(str(text or "").lower().split())
        if normalized:
            terms.add(normalized)
    for url in _urls_from_value(spec):
        terms.add(url.lower())
        parts = urlsplit(url)
        for item in (parts.netloc, parts.path, parts.fragment):
            item = str(item or "").strip("/").lower()
            if item:
                terms.add(item)
    base_url = str(context.base_url or "").strip()
    if base_url:
        terms.add(base_url.lower())
        parts = urlsplit(base_url)
        for item in (parts.netloc, parts.path, parts.fragment):
            item = str(item or "").strip("/").lower()
            if item:
                terms.add(item)
    return terms


def _strings_from_value(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    strings: list[str] = []
    if isinstance(value, list):
        for item in value:
            strings.extend(_strings_from_value(item))
    elif isinstance(value, dict):
        for item in value.values():
            strings.extend(_strings_from_value(item))
    return strings


def _urls_from_value(value: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(value, str):
        urls.extend(
            match.group(0)
            for match in re.finditer(r"https?://[^\s\"'<>，,。；;、)）\]】]+", value)
        )
        return urls
    if isinstance(value, list):
        for item in value:
            urls.extend(_urls_from_value(item))
    elif isinstance(value, dict):
        for item in value.values():
            urls.extend(_urls_from_value(item))
    return urls


def _record_element_scope(
    scopes: dict[str, tuple[str, int]],
    key: str,
    scope: str,
    priority: int,
) -> None:
    normalized_scope = _safe_element_key(scope)
    if not normalized_scope:
        return
    current = scopes.get(key)
    if current is None or priority >= current[1]:
        scopes[key] = (normalized_scope, priority)


def _scope_from_element_definition(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    for key in (
        "page",
        "page_name",
        "page_title",
        "screen",
        "view",
        "route",
        "section",
        "context",
    ):
        if value.get(key):
            return _scope_from_business_text(value[key])
    for key in ("url", "page_url", "href"):
        if value.get(key):
            return _scope_from_url(value[key])
    return None


def _scope_marker_from_step(step: dict[str, Any]) -> tuple[str, int] | None:
    for key in ("page", "page_name", "page_title", "screen", "view"):
        if step.get(key):
            scope = _scope_from_business_text(step[key])
            if scope:
                return scope, 40

    action = str(step.get("action") or step.get("type") or "").lower()
    if action in {"goto", "open", "navigate"}:
        scope = _scope_from_url(
            step.get("value") or step.get("url") or step.get("href") or step.get("link")
        )
        if scope:
            return scope, 10

    if action in {
        "assert_title",
        "assert_title_contains",
        "title_should_be",
        "title_contains",
    }:
        scope = _scope_from_business_text(
            step.get("value") or step.get("expected") or step.get("target")
        )
        if scope:
            return scope, 30
    return None


def _step_element_references(step: dict[str, Any], raw_keys: set[str]) -> set[str]:
    refs: set[str] = set()
    for field in ("selector", "target"):
        value = step.get(field)
        if isinstance(value, dict):
            value = _coerce_generated_string(
                value,
                field_path=f"step.{field}",
                preferred_keys=("key", "element_key", "name", "selector", "value"),
            )
        if isinstance(value, str) and value in raw_keys:
            refs.add(value)
    return refs


def _scope_from_url(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = urlsplit(text)
    route = parsed.fragment or parsed.path
    if route.startswith("/"):
        route = route[1:]
    route = route.split("?", 1)[0].split("#", 1)[0]
    route = unquote(route)
    segments = [
        segment
        for segment in re.split(r"[/\\._\-\s]+", route)
        if segment and not segment.isdigit()
    ]
    ignored = {
        "admin",
        "api",
        "app",
        "v",
        "www",
        "html",
        "index",
    }
    meaningful = [segment for segment in segments if segment.lower() not in ignored]
    if not meaningful:
        host = parsed.hostname or ""
        meaningful = [
            segment
            for segment in re.split(r"[._\-]+", host)
            if segment and segment.lower() not in ignored
        ]
    if not meaningful:
        return None
    route_text = "_".join(meaningful[-3:])
    aliases = {"home": "首页", "login": "登录页"}
    return aliases.get(route_text.lower(), route_text)


def _scope_from_business_text(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = re.sub(r"^(?:text|css|xpath)=", "", text, flags=re.IGNORECASE).strip()
    text = text.strip(" \"'[]()（）:：,，.。;；")
    text = re.sub(r"\s+", "_", text)
    if not text:
        return None
    if re.search(r"[\u4e00-\u9fff]", text) and not text.endswith(("页", "页面")):
        return f"{text}页"
    return text


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


_NO_SELECTOR_ACTIONS = set(NO_SELECTOR_ACTIONS)
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
_VALUE_ASSERTIONS = _lower_actions(StepAction.ASSERT_VALUE)
_MULTI_VALUE_ASSERTIONS = _lower_actions(StepAction.ASSERT_HAVE_VALUES)
_ASSERTION_ACTIONS = (
    _TEXT_ASSERTIONS
    | _VISIBILITY_ASSERTIONS
    | _URL_TITLE_ASSERTIONS
    | _COUNT_ASSERTIONS
    | _ATTRIBUTE_ASSERTIONS
    | _VALUE_ASSERTIONS
    | _MULTI_VALUE_ASSERTIONS
)


def _normalize_verify_step(step: dict[str, Any]) -> dict[str, Any]:
    action = str(step.get("action") or "").lower()
    if action in _ATTRIBUTE_ASSERTIONS and str(step.get("attribute") or "").lower() in {
        "value",
        "input_value",
        "input-value",
    }:
        step["action"] = "assert_value"
        step.pop("attribute", None)
        return step
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


def _safe_element_key(value: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_\u4e00-\u9fff]+", "_", str(value).strip()).strip("_")
    return key or "element"


def _unique_element_key(candidate: str, used_keys: set[str]) -> str:
    base = _safe_element_key(candidate)
    if base not in used_keys:
        return base
    index = 2
    while f"{base}_{index}" in used_keys:
        index += 1
    return f"{base}_{index}"


def _looks_raw_selector(value: str) -> bool:
    value = str(value).strip()
    return value.startswith(("#", ".", "//", "(//", "text=", "[", "css=", "xpath="))
