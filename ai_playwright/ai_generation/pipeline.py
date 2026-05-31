from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ai_playwright.ai_generation.harness import GenerationHarness
from ai_playwright.ai_generation.project_context import ProjectContext
from ai_playwright.step_actions.step_executor import StepExecutor
from ai_playwright.utils.logger import logger


@dataclass(frozen=True)
class CompiledCasePayload:
    payload: dict[str, Any]
    raw_payload: dict[str, Any]
    case_name: str
    steps: list[dict[str, Any]]
    model_calls: int
    warnings: list[str] = field(default_factory=list)
    cache_hit: bool = False
    cache_key: str = ""


@dataclass
class PayloadExecutionResult:
    case_name: str
    history: list[dict[str, Any]]


class PayloadExecutionError(AssertionError):
    def __init__(
        self,
        message: str,
        *,
        history: list[dict[str, Any]],
        failed_step: dict[str, Any],
        step_index: int,
        original_error: Exception,
    ):
        super().__init__(message)
        self.history = history
        self.failed_step = failed_step
        self.step_index = step_index
        self.original_error = original_error


def compile_case_payload(
    *,
    context: ProjectContext,
    spec: dict[str, Any],
    env: str,
    output_name: str,
    build_payload,
    normalize_payload=None,
    use_ai: bool = True,
    use_cache: bool = True,
    cache_output_name: str | None = None,
    artifacts=None,
    progress=None,
) -> CompiledCasePayload:
    cache_info: dict[str, Any] = {"cache_hit": False, "model_calls": 0}
    payload = build_payload(
        context,
        spec,
        env=env,
        output_name=cache_output_name or output_name,
        use_ai=use_ai,
        use_cache=use_cache,
        cache_info=cache_info,
        progress=progress,
    )
    if normalize_payload is not None:
        normalized, warnings = normalize_payload(
            context=context,
            spec=spec,
            output_name=output_name,
            payload=payload,
            use_ai=use_ai,
            progress=progress,
            artifacts=artifacts,
        )
    else:
        harness = GenerationHarness(
            context=context,
            spec=spec,
            output_name=output_name,
        )
        normalized = harness.normalize(payload)
        warnings = harness.validate(normalized)
    case_name, steps = first_payload_case_steps(normalized, preferred_name=output_name)
    return CompiledCasePayload(
        payload=normalized,
        raw_payload=payload,
        case_name=case_name,
        steps=steps,
        model_calls=int(cache_info.get("model_calls") or 0),
        warnings=warnings,
        cache_hit=bool(cache_info.get("cache_hit") or cache_info.get("hit")),
        cache_key=str(cache_info.get("key") or ""),
    )


def first_payload_case_steps(
    payload: dict[str, Any],
    *,
    preferred_name: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    data = payload.get("data") if isinstance(payload, dict) else {}
    if not isinstance(data, dict):
        raise ValueError("compiled payload data must be an object")
    case_data = data.get(preferred_name) if preferred_name else None
    case_name = str(preferred_name or "")
    if not isinstance(case_data, dict):
        for item in payload.get("cases") or []:
            if not isinstance(item, dict):
                continue
            candidate = str(item.get("name") or "")
            candidate_data = data.get(candidate)
            if isinstance(candidate_data, dict):
                case_name = candidate
                case_data = candidate_data
                break
    if not isinstance(case_data, dict):
        raise ValueError(f"compiled payload missing data for case: {preferred_name}")
    steps = case_data.get("steps") or []
    if not isinstance(steps, list):
        raise ValueError("compiled payload data.steps must be a list")
    return case_name, [step for step in steps if isinstance(step, dict)]


def install_payload_assets(
    step_executor: StepExecutor,
    payload: dict[str, Any],
    *,
    elements: dict[str, Any] | None = None,
) -> dict[str, Any]:
    previous = {
        "elements": dict(getattr(step_executor, "elements", {}) or {}),
        "modules_cache": dict(getattr(step_executor, "modules_cache", {}) or {}),
    }
    payload_elements = payload.get("elements") if isinstance(payload, dict) else {}
    payload_modules = payload.get("modules") if isinstance(payload, dict) else {}
    payload_vars = payload.get("vars") if isinstance(payload, dict) else {}
    if isinstance(payload_elements, dict) and payload_elements:
        merged_elements = {**previous["elements"], **payload_elements}
        step_executor.elements = merged_elements
        if elements is not None:
            elements.clear()
            elements.update(merged_elements)
    if isinstance(payload_modules, dict) and payload_modules:
        modules_cache = getattr(step_executor, "modules_cache", None)
        if modules_cache is None:
            step_executor.modules_cache = {}
            modules_cache = step_executor.modules_cache
        modules_cache.update(
            {name: {name: steps} for name, steps in payload_modules.items()}
        )
    if isinstance(payload_vars, dict) and payload_vars:
        variable_manager = getattr(step_executor, "variable_manager", None)
        if variable_manager is not None:
            variable_manager.import_variables(
                payload_vars,
                scope="temp",
                overwrite=True,
            )
    return previous


def restore_payload_assets(
    step_executor: StepExecutor,
    previous: dict[str, Any],
    *,
    elements: dict[str, Any] | None = None,
) -> None:
    previous_elements = dict(previous.get("elements") or {})
    previous_modules = dict(previous.get("modules_cache") or {})
    if hasattr(step_executor, "elements"):
        step_executor.elements = previous_elements
    if hasattr(step_executor, "modules_cache"):
        step_executor.modules_cache = previous_modules
    if elements is not None:
        elements.clear()
        elements.update(previous_elements)


def execute_compiled_payload_steps(
    *,
    step_executor: StepExecutor,
    payload: dict[str, Any],
    case_name: str,
    steps: list[dict[str, Any]],
    elements: dict[str, Any] | None = None,
    execute_step=None,
    history_item=None,
    source: str = "compiled_case",
    max_steps: int | None = None,
) -> PayloadExecutionResult:
    install_payload_assets(step_executor, payload, elements=elements)
    history: list[dict[str, Any]] = []
    for index, raw_step in enumerate(steps, start=1):
        if max_steps is not None and len(history) >= max_steps:
            raise AssertionError(
                f"compiled payload exceeded max steps: case={case_name} | max_steps={max_steps}"
            )
        step = copy.deepcopy(raw_step)
        _reuse_previous_fill_selector(step, history)
        logger.info(
            "编译步骤执行: "
            f"case={case_name} | step={index}/{len(steps)} | {_format_step(step)}"
        )
        try:
            if execute_step is not None:
                execute_step(step)
            else:
                step_executor.execute_step(step)
        except Exception as exc:
            failed_step = copy.deepcopy(step)
            failed_step["_execution_error"] = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "编译步骤失败: "
                f"case={case_name} | step={index}/{len(steps)} "
                f"| {_format_step(step)} | error={exc}"
            )
            if history_item is not None:
                history.append(
                    history_item(step=failed_step, source=source, result="failed")
                )
            else:
                history.append(
                    {
                        "index": None,
                        "source": source,
                        "step": dict(failed_step),
                        "decision": None,
                        "result": "failed",
                        "url_after": getattr(
                            getattr(step_executor, "page", None), "url", ""
                        ),
                    }
                )
            raise PayloadExecutionError(
                f"compiled payload step failed: case={case_name} | step={index}",
                history=history,
                failed_step=failed_step,
                step_index=index,
                original_error=exc,
            ) from exc
        logger.info("编译步骤完成: " f"case={case_name} | step={index}/{len(steps)}")
        if history_item is not None:
            history.append(history_item(step=step, source=source, result="passed"))
        else:
            history.append(
                {
                    "index": None,
                    "source": source,
                    "step": dict(step),
                    "decision": None,
                    "result": "passed",
                    "url_after": getattr(
                        getattr(step_executor, "page", None), "url", ""
                    ),
                }
            )
    return PayloadExecutionResult(case_name=case_name, history=history)


def _reuse_previous_fill_selector(
    step: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    action = str(step.get("action") or "").lower()
    if action not in {"assert_value", "assert_have_values"}:
        return
    if step.get("selector"):
        return
    target = _normalize_text(step.get("target"))
    value = _normalize_text(step.get("value") or step.get("expected"))
    if not target and not value:
        return
    for item in reversed(history):
        previous_step = item.get("step") if isinstance(item, dict) else None
        if not isinstance(previous_step, dict):
            continue
        if str(previous_step.get("action") or "").lower() != "fill":
            continue
        previous_target = _normalize_text(previous_step.get("target"))
        previous_value = _normalize_text(previous_step.get("value"))
        target_matches = bool(target and previous_target and target == previous_target)
        value_matches = bool(value and previous_value and value == previous_value)
        if target and previous_target and not target_matches:
            continue
        if not target_matches and value and previous_value and not value_matches:
            continue
        if not target_matches and not value_matches:
            continue
        selector = (
            previous_step.get("_action_executed_selector")
            or previous_step.get("_resolved_selector")
            or previous_step.get("selector")
        )
        if selector:
            step["selector"] = selector
            step.setdefault("mode", previous_step.get("mode") or "smart")
            return


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _format_step(step: dict[str, Any]) -> str:
    action = step.get("action") or ("use_module" if step.get("use_module") else "")
    parts = [f"action={action}"]
    for key in ("use_module", "selector", "target", "value", "key"):
        if step.get(key) is not None:
            parts.append(f"{key}={step.get(key)}")
    return " | ".join(parts)
