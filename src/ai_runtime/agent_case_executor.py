from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.ai_generation.case_generator import (
    _build_payload,
)
from src.ai_generation.harness import GenerationHarness
from src.ai_generation.pipeline import (
    compile_case_payload,
    execute_compiled_payload_steps,
    PayloadExecutionError,
    restore_payload_assets,
)
from src.ai_generation.project_context import (
    ProjectContext,
    load_project_context,
)
from src.ai_runtime.ai_cache_store import AiCacheStore, ai_cache_path_from_config
from src.ai_runtime.cache_scope import context_asset_fingerprint, resolve_entry_scope
from src.ai_runtime.config import load_ai_config
from src.ai_runtime.contracts import AgentCaseDecision, AgentCaseRuntimeDecision
from src.ai_runtime.native_observe import NativeObserveSettings
from src.ai_runtime.payload_compactor import (
    build_dom_context,
    compact_model_dom_context,
    compact_dom_candidates,
    compact_history,
    compact_project_context,
    compressed_decision_summary,
    looks_like_internal_element_id,
    normalize_model_text,
    selector_for_element_id,
    selectors_for_element_id,
)
from src.ai_runtime.playwright_selectors import (
    collect_candidates,
    heuristic_selectors,
    semantic_selectors,
)
from src.ai_runtime.semantic_terms import semantic_text_variants
from src.ai_runtime.provider import (
    ChatCompletionProvider,
    load_llm_settings,
    parse_json_object,
)
from src.step_actions.step_executor import StepExecutor
from utils.logger import logger
from utils.token_usage import get_token_usage_tracker


_URL_RE = re.compile(r"https?://[^\s\"'<>，,。；;、)）\]】]+", re.IGNORECASE)
_PAYMENT_KEYWORDS = ("支付", "付款", "真实支付", "payment", "pay now", "credit card")
_ACTION_CATEGORIES: dict[str, tuple[str, ...]] = {
    "navigation": ("goto",),
    "interaction": ("click", "press"),
    "input": ("fill",),
    "assertion": (
        "assert_visible",
        "assert_text",
        "assert_url_contains",
        "assert_title",
        "assert_title_contains",
    ),
    "wait": ("wait",),
    "completion": ("finish", "fail"),
}
_DEFAULT_GUARDRAILS = {
    "no_runtime_registry_write": True,
    "stop_on_external_domain": True,
    "stop_on_unexpected_payment": True,
    "require_checkpoints_or_final": True,
}
_AGENT_CASE_SYSTEM_PROMPT = (
    "你是UI自动化运行时Agent。每次只返回一个JSON动作。"
    "stable_context是完整用例规格，realtime_state是唯一实时事实来源。"
    "必须按natural_steps顺序推进，把realtime_state.current_step作为本轮主任务。"
    "每轮只处理current_step/remaining_steps[0]；completed_steps和execution_evidence中已完成的节点不要重复执行。"
    "如果current_step目标在当前DOM不可见，只能返回让current_step可达的必要前置动作，reason要说明服务于current_step。"
    "runtime_harness只提供事实反馈，不替你选择动作；target_candidates只是候选事实，不能无视current_step直接点击。"
    "优先返回dom_context中的element_id；selector只能是真实选择器，不能写e1/e2。"
    "fill必须带value；inputs只是可选结构化补充；有inputs时${name}可作为运行时变量value。"
    "没有inputs时，直接使用intent/steps/criteria里的字面值，不要改写成未定义${name}。"
    "press必须带key和element_id/selector/target。"
    "页面可见标题用assert_text/assert_visible，不用assert_title。"
    "不要把外链、下载、退出、删除、重置、清空当作普通前置动作；除非current_step明确要求，否则应选择能推进当前步骤的页面动作。"
    "如果runtime_harness.feedback显示可见错误或页面错误，返回可处理该状态的动作或fail。"
    "禁止输出thought/analysis/steps/mode等契约外字段。"
    "验收满足后finish；无法继续fail；信息不足need_more_context。"
)
_AGENT_CASE_ACTION_CONTRACT = (
    "click/fill/assert_* use element_id|selector|target; "
    "fill also needs value; assert_text uses value as expected visible text; "
    "literal values from intent/natural_steps/criteria are valid fill/assert values; "
    "press uses key plus element_id|selector|target; "
    "goto/assert_url_contains/assert_title_contains use value; "
    "wait uses wait_ms; finish/fail use reason"
)
_AGENT_DECISION_FIELDS = set(AgentCaseDecision.model_fields)
_AGENT_RUNTIME_DECISION_FIELDS = set(AgentCaseRuntimeDecision.model_fields)
_AGENT_RUNTIME_ACTIONS = tuple(
    action
    for actions in _ACTION_CATEGORIES.values()
    for action in actions
)
_ASSERTION_TEXT_ELEMENT_KEYWORDS = (
    "title",
    "heading",
    "header",
    "badge",
    "message",
    "text",
    "name",
)
_COMMON_ASSERTION_TEXT_SELECTORS = (
    '[data-test="title"]',
    "[data-test='title']",
    '[data-testid="title"]',
    "[data-testid='title']",
    ".title",
    "[role='heading']",
    '[role="heading"]',
    "h1",
    "h2",
    "h3",
)
class AgentDecisionRejected(ValueError):
    """Recoverable Agent decision rejection; the next loop asks the model again."""


class AgentCaseCompileContractError(ValueError):
    """Non-recoverable compile contract violation for runtime agent_case."""


@dataclass
class AgentCaseRunResult:
    case_name: str
    steps_executed: int
    model_calls: int
    decisions: list[dict[str, Any]] = field(default_factory=list)
    final_reason: str = ""


@dataclass(frozen=True)
class CompiledAgentCase:
    generation_spec: dict[str, Any]
    payload: dict[str, Any]
    steps: list[dict[str, Any]]
    model_calls: int
    cache_hit: bool


class AgentCasePlanCache:
    """Compiled agent_case plan cache. The plan references runtime YAML assets."""

    def __init__(self, path: str | Path):
        self.store = AiCacheStore(path)
        self.namespace = "agent_case_plan"

    def load_plan(self, key: str) -> dict[str, Any] | None:
        try:
            return self.store.get_payload(namespace=self.namespace, key=key)
        except Exception as exc:
            logger.warning(f"agent_case plan缓存读取失败，忽略缓存: {exc}")
            return None

    def save_plan(
        self,
        *,
        key: str,
        project: str,
        env: str,
        case_name: str,
        entry_scope: str,
        spec: dict[str, Any],
        payload: dict[str, Any],
        case_payload_name: str,
        steps: list[dict[str, Any]],
        prompt_version: str,
        schema_version: str,
        model: str,
        asset_hash: str,
    ) -> None:
        try:
            self.store.put_payload(
                namespace=self.namespace,
                key=key,
                project=project,
                env=env,
                entry_scope=entry_scope,
                case_name=case_name,
                input_type=str(spec.get("input_type") or ""),
                model=model,
                prompt_version=prompt_version,
                schema_version=schema_version,
                spec_hash=_hash_payload(_plan_cache_spec(spec)),
                asset_hash=asset_hash,
                payload={
                    "case_name": case_payload_name,
                    "steps": _cacheable_plan_steps(steps),
                    "payload": _cacheable_plan_payload(payload),
                },
                metadata={
                    "intent": spec.get("intent"),
                    "steps": spec.get("steps"),
                    "inputs": spec.get("inputs"),
                    "criteria": spec.get("criteria"),
                    "updated_at": int(time.time()),
                },
                status="verified",
            )
        except Exception as exc:
            logger.warning(f"agent_case plan缓存写入失败，不阻塞执行: {exc}")


class AgentCaseExecutor:
    """Runtime AI case runner: observe current UI, decide one action, execute, repeat."""

    def __init__(
        self,
        *,
        page,
        ui_helper,
        elements: dict[str, Any],
        context: ProjectContext | None = None,
    ):
        self.page = page
        self.ui_helper = ui_helper
        self.elements = elements or {}
        self.project = os.environ.get("TEST_PROJECT", "demo")
        self.env = os.environ.get("TEST_ENV", "prod")
        self.context = context or load_project_context(self.project, env=self.env)
        self.ai_config = load_ai_config()
        self.native_observe = NativeObserveSettings.from_config(self.ai_config)
        runtime_cfg = self.ai_config.get("runtime", {})
        agent_policy = self.ai_config.get("agent_policy", {})
        policy_limits = agent_policy.get("limits") or {}
        self.default_guardrails = {
            **_DEFAULT_GUARDRAILS,
            **(agent_policy.get("guardrails") or {}),
        }
        generation_cfg = self.ai_config.get("generation", {})
        self.max_steps_default = int(policy_limits.get("max_steps") or 20)
        self.max_model_calls_default = int(policy_limits.get("max_model_calls") or 12)
        self.max_duration_default = int(
            policy_limits.get("max_duration_seconds") or 180
        )
        self.candidate_limit = int(runtime_cfg.get("candidate_limit", 120))
        self.agent_candidate_scan_limit = int(
            runtime_cfg.get("agent_candidate_scan_limit", self.candidate_limit)
        )
        self.agent_candidate_limit = int(
            runtime_cfg.get(
                "agent_candidate_limit",
                min(self.agent_candidate_scan_limit, 40),
            )
        )
        self.agent_reasoning_effort = runtime_cfg.get("agent_reasoning_effort")
        self.agent_timeout_seconds = runtime_cfg.get("agent_timeout_seconds")
        try:
            self.agent_completion_wait_seconds = float(
                runtime_cfg.get("agent_completion_wait_seconds", 12)
            )
        except (TypeError, ValueError):
            self.agent_completion_wait_seconds = 12.0
        self.agent_execution_retry_limit = int(
            runtime_cfg.get("agent_execution_retry_limit", 2)
        )
        self.plan_cache_enabled = bool(
            runtime_cfg.get("agent_case_plan_cache_enabled", True)
        )
        cache_path = ai_cache_path_from_config(self.ai_config)
        self.plan_cache = AgentCasePlanCache(cache_path)
        self.max_context_items = int(
            runtime_cfg.get(
                "agent_context_items",
                min(int(generation_cfg.get("max_context_items", 160)), 40),
            )
        )
        self.history_limit = int(runtime_cfg.get("agent_history_limit", 10))
        self.agent_model_candidate_limit = int(
            runtime_cfg.get(
                "agent_model_candidate_limit",
                min(self.agent_candidate_limit, 10),
            )
        )
        self.agent_model_form_limit = int(
            runtime_cfg.get(
                "agent_model_form_limit",
                min(self.agent_candidate_limit, 6),
            )
        )
        self.agent_model_assertion_limit = int(
            runtime_cfg.get(
                "agent_model_assertion_limit",
                min(self.agent_candidate_limit, 6),
            )
        )
        self.agent_prompt_project_items = int(
            runtime_cfg.get("agent_prompt_project_items", 6)
        )
        self.agent_prompt_modules = int(runtime_cfg.get("agent_prompt_modules", 2))
        self.agent_prompt_module_steps = int(
            runtime_cfg.get("agent_prompt_module_steps", 3)
        )
        prompts_cfg = self.ai_config.get("prompts", {})
        llm_cfg = self.ai_config.get("llm", {})
        self.agent_model = str(runtime_cfg.get("agent_model") or "").strip()
        self.agent_harness_lookahead = int(
            runtime_cfg.get("agent_harness_lookahead", 4)
        )
        self.prompt_version = str(
            prompts_cfg.get("agent_case_version", "agent-case-v1")
        )
        self.schema_version = str(llm_cfg.get("schema_version", "ui-ai-schema-v1"))
        self.step_executor = StepExecutor(
            page,
            ui_helper,
            self.elements,
            default_mode="smart",
        )
        self.current_dom_context: dict[str, Any] = {}
        self.last_decision_used_model = False
        self._last_agent_prompt_metrics: dict[str, Any] = {}

    def run(self, *, case_name: str, case_data: dict[str, Any]) -> AgentCaseRunResult:
        spec = self._agent_spec(case_name=case_name, case_data=case_data)
        history: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        started = time.monotonic()
        cache_key = self._cache_key(case_name=case_name, spec=spec)

        logger.info(
            "Agent用例执行开始: "
            f"case={case_name} | max_steps={spec['max_steps']} "
            f"| max_model_calls={spec['max_model_calls']} "
            f"| input_type={spec['input_type']} | intent={spec['intent']}"
        )
        logger.info(
            "Agent验收标准: "
            f"case={case_name} | {_criteria_summary(spec['criteria'])}"
        )

        self._maybe_open_start_url(
            case_name=case_name,
            case_data=case_data,
            spec=spec,
            intent=spec["intent"],
            steps=spec["steps"],
            history=history,
        )

        model_calls = 0
        execution_failures = 0

        local_result = self._local_completion_decision(
            case_name=case_name,
            spec=spec,
            history=history,
            cache_key=cache_key,
            model_calls=model_calls,
        )
        if local_result is not None:
            return local_result

        compiled_result = self._try_run_compiled_agent_case(
            case_name=case_name,
            spec=spec,
            history=history,
            cache_key=cache_key,
        )
        if compiled_result is not None:
            return compiled_result

        while len(history) < spec["max_steps"]:
            if time.monotonic() - started > spec["max_duration_seconds"]:
                raise AssertionError(
                    "Agent用例超过最大执行时长: "
                    f"case={case_name} | max_duration_seconds={spec['max_duration_seconds']}"
            )
            if model_calls >= spec["max_model_calls"]:
                raise AssertionError(
                    "Agent用例超过最大模型调用次数: "
                    f"case={case_name} | max_model_calls={spec['max_model_calls']}"
                )

            runtime_harness = self._runtime_harness(
                spec=spec,
                history=history,
                dom_context=self.current_dom_context,
                candidate_limit=self.agent_model_candidate_limit,
            )

            decision = self._decide_next_action(
                case_name=case_name,
                spec=spec,
                history=history,
                step_index=len(history) + 1,
                runtime_harness=runtime_harness,
            )
            if self.last_decision_used_model:
                model_calls += 1
            decision_dict = decision.model_dump(exclude_none=True)
            decisions.append(decision_dict)
            logger.info(
                "Agent用例决策: "
                f"case={case_name} | step={len(history) + 1}/{spec['max_steps']} "
                f"| status={decision.status} | action={decision.action} "
                f"| element_id={decision.element_id} | target={decision.target} "
                f"| selector={decision.selector} | value={decision.value} "
                f"| reason={normalize_model_text(decision.reason)} "
                f"| expected={normalize_model_text(decision.expected)} "
                f"| confidence={decision.confidence}"
            )

            if decision.status == "need_more_context":
                logger.warning(
                    "Agent请求更多上下文: "
                    f"case={case_name} | reason={normalize_model_text(decision.reason)} "
                    f"| requested_level={decision.context_level}"
                )
                history.append(
                    self._history_item(
                        step={"action": "wait", "value": "0"},
                        source="need_more_context",
                        decision=decision,
                        result="need_more_context",
                    )
                )
                continue

            if decision.status in {"blocked", "failed"}:
                raise AssertionError(
                    "Agent用例判定失败: "
                    f"case={case_name} | status={decision.status} "
                    f"| reason={normalize_model_text(decision.reason)}"
                )

            if decision.action in {"done", "finish"}:
                logger.info(
                    "Agent模型结束执行: "
                    f"case={case_name} "
                    f"| evidence={normalize_model_text(decision.reason)}"
                )
                result = AgentCaseRunResult(
                    case_name=case_name,
                    steps_executed=len(history),
                    model_calls=model_calls,
                    decisions=decisions,
                    final_reason=normalize_model_text(decision.reason),
                )
                logger.info(
                    "Agent用例执行完成: "
                    f"case={case_name} | steps_executed={result.steps_executed} "
                    f"| model_calls={model_calls} "
                    f"| reason={result.final_reason}"
                )
                return result

            if decision.action == "fail":
                raise AssertionError(
                    "Agent用例判定失败: "
                    f"case={case_name} | reason={normalize_model_text(decision.reason)}"
                )

            try:
                self._guard_decision(decision, spec=spec, history=history)
                step = self._decision_to_step(decision, spec=spec)
            except AgentDecisionRejected as exc:
                logger.warning(
                    "Agent决策被通用约束拒绝，重新观察: "
                    f"case={case_name} | reason={exc}"
                )
                history.append(
                    self._history_item(
                        step={
                            "action": "wait",
                            "value": "0",
                            "target": f"decision rejected: {exc}",
                        },
                        source="decision_guard",
                        decision=decision,
                        result="rejected",
                    )
                )
                continue
            logger.info(
                "Agent执行动作: "
                f"case={case_name} | step={len(history) + 1}/{spec['max_steps']} "
                f"| {_format_step(step)}"
            )
            try:
                self._execute_step(step, spec=spec)
            except Exception as exc:
                failed_step = copy.deepcopy(step)
                failed_step["_execution_error"] = f"{type(exc).__name__}: {exc}"
                history.append(
                    self._history_item(
                        step=failed_step,
                        source="agent_decision",
                        decision=decision,
                        result="failed",
                    )
                )
                if _is_assertion_step(step):
                    raise AssertionError(
                        "Agent断言执行失败: "
                        f"case={case_name} | {_format_step(step)} | error={exc}"
                    ) from exc
                execution_failures += 1
                if execution_failures > self.agent_execution_retry_limit:
                    raise AssertionError(
                        "Agent动作执行失败且超过重试次数: "
                        f"case={case_name} | retries={execution_failures}/"
                        f"{self.agent_execution_retry_limit} | error={exc}"
                    ) from exc
                logger.warning(
                    "Agent动作执行失败，交给模型重新规划: "
                    f"case={case_name} | retry={execution_failures}/"
                    f"{self.agent_execution_retry_limit} | error={exc}"
                )
                continue
            execution_failures = 0
            logger.info(
                "Agent动作完成: "
                f"case={case_name} | step={len(history) + 1}/{spec['max_steps']} "
                f"| url={getattr(self.page, 'url', '')}"
            )
            history.append(
                self._history_item(
                    step=step,
                    source="agent_decision",
                    decision=decision,
                    result="passed",
                )
            )
            failure_reason = self._post_action_failure_boundary(
                spec=spec,
                history=history,
            )
            if failure_reason:
                raise AssertionError(
                    "Agent用例运行态错误未恢复: "
                    f"case={case_name} | reason={failure_reason}"
                )

        raise AssertionError(
            f"Agent用例超过最大步骤数仍未完成: case={case_name} | max_steps={spec['max_steps']}"
        )

    def _agent_spec(
        self, *, case_name: str, case_data: dict[str, Any]
    ) -> dict[str, Any]:
        agent_case = normalize_agent_case(case_data)
        if not agent_case:
            raise ValueError(
                f"Agent用例缺少 intent/checkpoints/final 配置: {case_name}"
            )
        intent = str(agent_case.get("intent") or "").strip()
        steps = _normalize_text_list(agent_case.get("steps"))
        if not intent and not steps:
            raise ValueError(f"Agent用例必须提供 intent 或 steps: {case_name}")
        if not intent:
            intent = "按顺序完成自然语言步骤：" + "；".join(steps)
        criteria = agent_case.get("criteria")
        if not _has_criteria(criteria):
            raise ValueError(f"Agent用例必须声明 checkpoints 或 final: {case_name}")
        return {
            "description": str(case_data.get("description") or "").strip(),
            "intent": intent,
            "steps": steps,
            "input_type": "steps" if steps else "intent",
            "inputs": agent_case.get("inputs") or {},
            "criteria": criteria,
            "module_entry_allowed": _has_explicit_module_entry(
                case_data=case_data,
                intent=intent,
                steps=steps,
                modules=self.context.modules,
            ),
            "guardrails": self.default_guardrails,
            "max_steps": self.max_steps_default,
            "max_model_calls": self.max_model_calls_default,
            "max_duration_seconds": self.max_duration_default,
        }

    def _try_run_compiled_agent_case(
        self,
        *,
        case_name: str,
        spec: dict[str, Any],
        history: list[dict[str, Any]],
        cache_key: str,
    ) -> AgentCaseRunResult | None:
        if history and _history_action_satisfies(
            history,
            actions={"click", "fill", "press", "press_key"},
            terms=[],
        ):
            return None
        try:
            compiled = self._compile_agent_case_steps(
                case_name=case_name,
                spec=spec,
            )
        except AgentCaseCompileContractError:
            raise
        except Exception as exc:
            logger.warning(
                "Agent内存编译失败，切换逐步实时规划: "
                f"case={case_name} | error={exc}"
            )
            return None
        payload = compiled.payload
        steps = compiled.steps
        if not steps:
            logger.info(
                "Agent内存编译未生成steps，切换逐步实时规划: "
                f"case={case_name}"
            )
            return None
        previous_assets = {
            "elements": dict(getattr(self.step_executor, "elements", {}) or {}),
            "modules_cache": dict(getattr(self.step_executor, "modules_cache", {}) or {}),
        }
        try:
            execution = execute_compiled_payload_steps(
                step_executor=self.step_executor,
                payload=payload,
                case_name=case_name,
                steps=steps,
                elements=self.elements,
                execute_step=lambda step: self._execute_step(step, spec=spec),
                history_item=self._history_item,
                source="compiled_agent_case",
                max_steps=spec["max_steps"],
            )
            history.extend(execution.history)
            unmet = self._wait_for_compiled_completion_criteria(
                spec=spec,
                history=history,
                timeout_seconds=self.agent_completion_wait_seconds,
            )
            if unmet:
                raise AssertionError(
                    "Agent编译步骤执行完成但验收标准未满足: "
                    f"case={case_name} | unmet={unmet}"
                )
            if not compiled.cache_hit:
                self._save_compiled_plan_cache(
                    key=cache_key,
                    case_name=case_name,
                    spec=spec,
                    compiled=compiled,
                )
            return self._compiled_success_result(
                case_name=case_name,
                spec=spec,
                history=history,
                cache_key=cache_key,
                model_calls=compiled.model_calls,
                final_reason="compiled steps executed",
            )
        except PayloadExecutionError as exc:
            history.extend(exc.history)
            restore_payload_assets(
                self.step_executor,
                previous_assets,
                elements=self.elements,
            )
            failure_kind = "断言" if _is_assertion_step(exc.failed_step) else "步骤"
            raise AssertionError(
                f"Agent编译{failure_kind}执行失败: "
                f"case={case_name} | step={exc.step_index} "
                f"| {_format_step(exc.failed_step)} | error={exc.original_error}"
            ) from exc
        except Exception as exc:
            logger.warning(
                "Agent编译步骤执行失败: "
                f"case={case_name} | executed={len(history)} | error={exc}"
            )
            history.clear()
            restore_payload_assets(
                self.step_executor,
                previous_assets,
                elements=self.elements,
            )
            raise AssertionError(
                "Agent编译步骤执行失败: "
                f"case={case_name} | error={exc}"
            ) from exc

    def _compiled_success_result(
        self,
        *,
        case_name: str,
        spec: dict[str, Any],
        history: list[dict[str, Any]],
        cache_key: str,
        model_calls: int,
        final_reason: str,
    ) -> AgentCaseRunResult:
        result = AgentCaseRunResult(
            case_name=case_name,
            steps_executed=len(history),
            model_calls=model_calls,
            decisions=[],
            final_reason=final_reason,
        )
        logger.info(
            "Agent编译执行完成: "
            f"case={case_name} | steps_executed={result.steps_executed} "
            f"| model_calls={model_calls} "
            f"| reason={final_reason}"
        )
        return result

    def _compile_agent_case_steps(
        self,
        *,
        case_name: str,
        spec: dict[str, Any],
    ) -> CompiledAgentCase:
        cached = self._load_compiled_plan_cache(
            case_name=case_name,
            spec=spec,
        )
        if cached is not None:
            return cached

        allowed_modules = set(self.context.modules or {})
        generation_spec = _agent_spec_to_generation_spec(
            case_name=case_name,
            spec=spec,
            allowed_modules=sorted(allowed_modules),
        )
        compiled = compile_case_payload(
            context=self.context,
            spec=generation_spec,
            env=self.env,
            output_name=case_name,
            build_payload=_build_payload,
            normalize_payload=_normalize_runtime_compile_payload,
            use_ai=True,
            use_cache=False,
            progress=None,
        )
        _validate_runtime_compiled_payload(
            compiled.payload,
            allowed_modules=allowed_modules,
        )
        logger.info(
            "Agent内存编译完成: "
            f"case={case_name} | steps={len(compiled.steps)} "
            f"| model_calls={compiled.model_calls} "
            f"| cache_hit={compiled.cache_hit} "
            f"| elements={len(compiled.payload.get('elements') or {})} "
            f"| modules={len(compiled.payload.get('modules') or {})}"
        )
        return CompiledAgentCase(
            generation_spec=generation_spec,
            payload=compiled.payload,
            steps=compiled.steps,
            model_calls=compiled.model_calls,
            cache_hit=False,
        )

    def _load_compiled_plan_cache(
        self,
        *,
        case_name: str,
        spec: dict[str, Any],
    ) -> CompiledAgentCase | None:
        if not self.plan_cache_enabled:
            return None
        key = self._cache_key(case_name=case_name, spec=spec)
        record = self.plan_cache.load_plan(key)
        if not record:
            logger.info(
                "Agent plan缓存未命中: " f"case={case_name} | key_prefix={key[:12]}"
            )
            return None
        payload = record.get("payload")
        steps = record.get("steps")
        if not isinstance(payload, dict) or not isinstance(steps, list):
            logger.warning(f"Agent plan缓存格式无效，忽略: case={case_name}")
            return None
        logger.info(
            "Agent plan缓存命中: "
            f"case={case_name} | steps={len(steps)} | key_prefix={key[:12]}"
        )
        return CompiledAgentCase(
            generation_spec=_agent_spec_to_generation_spec(
                case_name=case_name,
                spec=spec,
                allowed_modules=sorted(self.context.modules or {}),
            ),
            payload=payload,
            steps=[copy.deepcopy(step) for step in steps if isinstance(step, dict)],
            model_calls=0,
            cache_hit=True,
        )

    def _save_compiled_plan_cache(
        self,
        *,
        key: str,
        case_name: str,
        spec: dict[str, Any],
        compiled: CompiledAgentCase,
    ) -> None:
        if not self.plan_cache_enabled:
            return
        if not _compiled_payload_safe_for_plan_cache(compiled.payload):
            logger.info(
                "Agent plan缓存跳过: compiled payload contains generated assets "
                f"| case={case_name}"
            )
            return
        self.plan_cache.save_plan(
            key=key,
            project=self.context.project,
            env=self.env,
            case_name=case_name,
            entry_scope=self._entry_scope(spec),
            spec=spec,
            payload=compiled.payload,
            case_payload_name=_first_payload_case_name(compiled.payload, case_name),
            steps=compiled.steps,
            prompt_version=self.prompt_version,
            schema_version=self.schema_version,
            model=self._model_cache_key(),
            asset_hash=_hash_payload(context_asset_fingerprint(self.context)),
        )

    def _refresh_dom_context(self, *, spec: dict[str, Any]) -> None:
        candidates = self._safe_collect_candidates()
        context_limit = self._dom_context_limit(candidates)
        self.current_dom_context = build_dom_context(
            candidates,
            url=getattr(self.page, "url", ""),
            title=_safe_page_title(self.page),
            context_level=2,
            limit=context_limit,
            hints=[spec["intent"], spec["steps"], spec["criteria"], spec["inputs"]],
        )

    def _wait_for_completion_criteria(
        self,
        *,
        spec: dict[str, Any],
        history: list[dict[str, Any]],
        timeout_seconds: float = 0,
    ) -> list[str]:
        criteria = spec.get("criteria")
        if not _has_completion_criteria(criteria):
            return []
        timeout_seconds = max(float(timeout_seconds or 0), 0)
        intent_unmet = _unmet_intent_action_requirements(
            spec=spec,
            history=history,
            current_url=getattr(self.page, "url", ""),
            dom_context=self.current_dom_context,
        )
        if intent_unmet:
            self._refresh_dom_context(spec=spec)
            intent_unmet = _unmet_intent_action_requirements(
                spec=spec,
                history=history,
                current_url=getattr(self.page, "url", ""),
                dom_context=self.current_dom_context,
            )
            criteria_unmet = _unmet_completion_criteria(
                criteria=criteria,
                history=history,
                current_url=getattr(self.page, "url", ""),
                dom_context=self.current_dom_context,
            )
            return _dedupe_texts(criteria_unmet + intent_unmet)

        deadline = time.monotonic() + timeout_seconds
        if timeout_seconds > 0:
            self._wait_for_observable_completion_terms(
                criteria=criteria,
                timeout_ms=int(timeout_seconds * 1000),
            )
        while True:
            self._refresh_dom_context(spec=spec)
            unmet = _unmet_completion_criteria(
                criteria=criteria,
                history=history,
                current_url=getattr(self.page, "url", ""),
                dom_context=self.current_dom_context,
            )
            unmet.extend(
                _unmet_intent_action_requirements(
                    spec=spec,
                    history=history,
                    current_url=getattr(self.page, "url", ""),
                    dom_context=self.current_dom_context,
                )
            )
            unmet = _dedupe_texts(unmet)
            if not unmet or time.monotonic() >= deadline:
                return unmet
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))

    def _wait_for_compiled_completion_criteria(
        self,
        *,
        spec: dict[str, Any],
        history: list[dict[str, Any]],
        timeout_seconds: float = 0,
    ) -> list[str]:
        criteria = spec.get("criteria")
        if not _has_completion_criteria(criteria):
            return []
        timeout_seconds = max(float(timeout_seconds or 0), 0)
        deadline = time.monotonic() + timeout_seconds
        if timeout_seconds > 0:
            self._wait_for_observable_completion_terms(
                criteria=criteria,
                timeout_ms=int(timeout_seconds * 1000),
            )
        while True:
            self._refresh_dom_context(spec=spec)
            unmet = _unmet_completion_criteria(
                criteria=criteria,
                history=history,
                current_url=getattr(self.page, "url", ""),
                dom_context=self.current_dom_context,
            )
            unmet = _dedupe_texts(unmet)
            if not unmet or time.monotonic() >= deadline:
                return unmet
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))

    def _wait_for_observable_completion_terms(
        self, *, criteria: Any, timeout_ms: int
    ) -> None:
        if timeout_ms <= 0 or not hasattr(self.page, "wait_for_function"):
            return
        title_terms, url_terms = _observable_completion_terms(criteria)
        if not title_terms and not url_terms:
            return
        try:
            self.page.wait_for_function(
                """(payload) => {
                    const title = String(document.title || '').toLowerCase();
                    const href = String(window.location.href || '').toLowerCase();
                    const titleTerms = payload.title_terms || [];
                    const urlTerms = payload.url_terms || [];
                    const titleOk = !titleTerms.length
                        || titleTerms.some((term) => title.includes(String(term).toLowerCase()));
                    const urlOk = !urlTerms.length
                        || urlTerms.some((term) => href.includes(String(term).toLowerCase()));
                    return titleOk && urlOk;
                }""",
                {"title_terms": title_terms, "url_terms": url_terms},
                timeout=timeout_ms,
            )
        except Exception:
            return

    def _local_completion_decision(
        self,
        *,
        case_name: str,
        spec: dict[str, Any],
        history: list[dict[str, Any]],
        cache_key: str,
        model_calls: int,
        runtime_harness: dict[str, Any] | None = None,
    ) -> AgentCaseRunResult | None:
        if not _has_completion_criteria(spec.get("criteria")):
            return None
        if not history and self._is_blank_page():
            return None
        self._refresh_dom_context(spec=spec)
        runtime_harness = runtime_harness or self._runtime_harness(
            spec=spec,
            history=history,
            dom_context=self.current_dom_context,
        )
        unmet = _unmet_completion_criteria(
            criteria=spec.get("criteria"),
            history=history,
            current_url=getattr(self.page, "url", ""),
            dom_context=self.current_dom_context,
        )
        unmet.extend(
            _unmet_intent_action_requirements(
                spec=spec,
                history=history,
                current_url=getattr(self.page, "url", ""),
                dom_context=self.current_dom_context,
            )
        )
        unmet = _dedupe_texts(unmet)
        if unmet:
            logger.info(
                "Agent本地验收未满足，继续实时规划: "
                f"case={case_name} | unmet={unmet}"
            )
            return None
        final_reason = "local completion criteria satisfied before model call"
        result = AgentCaseRunResult(
            case_name=case_name,
            steps_executed=len(history),
            model_calls=model_calls,
            decisions=[],
            final_reason=final_reason,
        )
        logger.info(
            "Agent本地验收通过，跳过模型调用: "
            f"case={case_name} | steps_executed={len(history)}"
        )
        return result

    def _maybe_open_start_url(
        self,
        *,
        case_name: str,
        case_data: dict[str, Any],
        spec: dict[str, Any],
        intent: str,
        steps: list[str],
        history: list[dict[str, Any]],
        force: bool = False,
    ) -> None:
        if not force and not self._is_blank_page():
            return
        module_decision, explicit_module_reference = self._first_entry_module_decision(
            spec=spec,
            case_data=case_data,
            intent=intent,
            steps=steps,
        )
        if module_decision is not None:
            step = self._decision_to_step(module_decision, spec=spec)
            logger.info(
                "Agent用例启动入口: "
                f"case={case_name} | source=first_use_module | "
                f"module={module_decision.module}"
            )
            self._execute_step(step, spec=spec)
            history.append(
                self._history_item(
                    step=step,
                    source="bootstrap",
                    decision=module_decision,
                )
            )
            return
        start_url = (
            _first_url(case_data.get("description"))
            or _first_url(intent)
            or _first_url(steps)
            or ("" if explicit_module_reference else _first_module_url(self.context.modules))
            or self.context.base_url
        )
        if not start_url:
            return
        step = {"action": "goto", "value": start_url}
        logger.info(
            "Agent用例启动入口: "
            f"case={case_name} | source=description_or_intent_or_project | url={start_url}"
        )
        self._execute_step(step, spec={"guardrails": {}})
        history.append(self._history_item(step=step, source="bootstrap"))

    def _first_entry_module_decision(
        self,
        *,
        spec: dict[str, Any],
        case_data: dict[str, Any],
        intent: str,
        steps: list[str],
    ) -> tuple[AgentCaseDecision | None, bool]:
        module_name, explicit_reference = _first_entry_module_name(
            case_data=case_data,
            intent=intent,
            steps=steps,
            modules=self.context.modules,
        )
        if not explicit_reference or not module_name:
            return None, explicit_reference
        if module_name not in (self.context.modules or {}):
            return None, True
        params, missing = self._infer_module_params(
            module_name,
            spec=spec,
            provided_params={},
        )
        if missing:
            return None, True
        payload: dict[str, Any] = {
            "action": "use_module",
            "module": module_name,
            "reason": "首步使用项目模块",
        }
        if params:
            payload["params"] = params
        return AgentCaseDecision.model_validate(payload), True

    def _execute_step(self, step: dict[str, Any], *, spec: dict[str, Any]) -> None:
        before_dom_signature = self._step_dom_signature(step)
        before_url = str(getattr(self.page, "url", "") or "")
        previous_pages = (
            _page_context_pages(self.page) if _step_action(step) == "click" else []
        )
        should_wait_for_new_page = _click_may_open_new_page(
            self.page,
            step.get("selector"),
        )
        new_page_url_hint = (
            _new_page_navigation_hint(self.page, step.get("selector"))
            if should_wait_for_new_page
            else None
        )
        self.step_executor.execute_step(step)
        if getattr(self.step_executor, "step_has_error", False):
            raise AssertionError(f"Agent动作执行失败: {_format_step(step)}")
        self._sync_runtime_page_from_step_executor()
        self._enrich_executed_step_state(step)
        self._enrich_action_observation(
            step,
            before_signature=before_dom_signature,
            before_url=before_url,
        )
        if previous_pages:
            new_page_wait_ms = (
                500
                if should_wait_for_new_page and new_page_url_hint
                else 2000 if should_wait_for_new_page else 0
            )
            adopted = self._adopt_new_page_if_opened(
                previous_pages,
                wait_ms=new_page_wait_ms,
            )
            if not adopted and new_page_url_hint:
                self._open_new_page_from_hint(new_page_url_hint)

    def _sync_runtime_page_from_step_executor(self) -> None:
        ui_page = getattr(getattr(self.step_executor, "ui_helper", None), "page", None)
        step_page = getattr(self.step_executor, "page", None)
        page = ui_page or step_page
        if page is not None and page is not self.page:
            self._adopt_runtime_page(page, reason="step executor page changed")

    def _enrich_executed_step_state(self, step: dict[str, Any]) -> None:
        if _step_action(step).lower() != "fill":
            return
        selector = str(step.get("_resolved_selector") or step.get("selector") or "")
        if not hasattr(self.page, "locator"):
            return
        try:
            locator = self.page.locator(selector).first
            if callable(locator):
                locator = locator()
            actual = _locator_text(locator)
        except Exception:
            actual = ""
        if actual:
            step["_resolved_value_after"] = actual

    def _step_dom_signature(self, step: dict[str, Any]) -> str:
        if _step_action(step).lower() not in {"click", "press", "press_key", "wait"}:
            return ""
        return _page_observation_signature(self.page)

    def _enrich_action_observation(
        self,
        step: dict[str, Any],
        *,
        before_signature: str,
        before_url: str,
    ) -> None:
        if _step_action(step).lower() not in {"click", "press", "press_key", "wait"}:
            return
        step["_action_before_url"] = str(before_url or "")
        step["_action_after_url"] = str(getattr(self.page, "url", "") or "")
        resolved_selector = str(step.get("_resolved_selector") or "").strip()
        if resolved_selector:
            step["_action_executed_selector"] = resolved_selector
        after_signature = _page_observation_signature(self.page)
        if before_signature or after_signature:
            step["_action_dom_changed"] = bool(
                before_signature and after_signature and before_signature != after_signature
            )
        selector = resolved_selector or str(step.get("selector") or "").strip()
        if selector:
            text_query = _text_query_from_selector(selector)
            if text_query and not step.get("_action_target_text"):
                step["_action_target_text"] = normalize_model_text(text_query, limit=80)
            if text_query:
                step["_action_target_visible_after"] = _page_contains_text(
                    self.page,
                    text_query,
                )

    def _post_action_failure_boundary(
        self,
        *,
        spec: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> str:
        self._refresh_dom_context(spec=spec)
        feedback = _runtime_feedback(
            history=history,
            dom_context=self.current_dom_context,
            current_url=getattr(self.page, "url", ""),
        )
        visible_errors = feedback.get("visible_errors")
        last_action = feedback.get("last_action") if isinstance(feedback, dict) else {}
        last_page_errors = (
            last_action.get("page_errors") if isinstance(last_action, dict) else None
        )
        last_url_changed = (
            bool(last_action.get("url_changed")) if isinstance(last_action, dict) else False
        )
        stalled_count = int(feedback.get("stalled_on_url") or 0)
        if visible_errors and last_page_errors and not last_url_changed:
            return "页面提交后显示错误: " + "; ".join(
                str(item) for item in visible_errors
            )
        if visible_errors and stalled_count >= 6:
            return (
                "页面持续显示错误且多次动作未使URL变化: "
                + "; ".join(str(item) for item in visible_errors)
            )
        return ""

    def _adopt_new_page_if_opened(
        self,
        previous_pages: list[Any],
        *,
        wait_ms: int = 0,
    ) -> bool:
        context = getattr(self.page, "context", None)
        if context is None and previous_pages:
            context = getattr(previous_pages[0], "context", None)
        if context is None:
            return False
        deadline = time.monotonic() + max(wait_ms, 0) / 1000
        while True:
            current_pages = list(getattr(context, "pages", []) or [])
            new_pages = [page for page in current_pages if page not in previous_pages]
            if new_pages:
                new_page = new_pages[-1]
                try:
                    new_page.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass
                self._adopt_runtime_page(new_page, reason="click opened new page")
                logger.info(
                    "Agent点击打开新页面，已切换观察页面: "
                    f"url={getattr(new_page, 'url', '')}"
                )
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def _open_new_page_from_hint(self, url: str) -> None:
        context = getattr(self.page, "context", None)
        if context is None or not hasattr(context, "new_page"):
            return
        new_page = context.new_page()
        new_page.goto(url, wait_until="domcontentloaded")
        try:
            new_page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass
        self._adopt_runtime_page(new_page, reason="target blank navigation fallback")
        logger.info("Agent按target=_blank语义打开新页面: " f"url={url}")

    def _adopt_runtime_page(self, page: Any, *, reason: str) -> None:
        self.page = page
        self.step_executor.page = page
        if hasattr(self.ui_helper, "page"):
            self.ui_helper.page = page
        pages = getattr(self.ui_helper, "pages", None)
        if isinstance(pages, list) and page not in pages:
            pages.append(page)
        resolver = getattr(self.step_executor, "smart_resolver", None)
        if resolver is not None:
            resolver.page = page
        logger.debug(
            "Agent运行时页面引用已同步: "
            f"reason={reason} | url={getattr(page, 'url', '')}"
        )

    def _decide_next_action(
        self,
        *,
        case_name: str,
        spec: dict[str, Any],
        history: list[dict[str, Any]],
        step_index: int,
        runtime_harness: dict[str, Any] | None = None,
    ) -> AgentCaseDecision:
        self.last_decision_used_model = True
        candidates = self._safe_collect_candidates()
        context_limit = self._dom_context_limit(candidates)
        criteria_prompt = _criteria_prompt_summary(
            spec["criteria"],
            checkpoint_limit=4 if spec["input_type"] == "intent" else 3,
        )
        history_prompt = compact_history(
            history,
            limit=self.history_limit,
        )
        page_title = _safe_page_title(self.page)
        current_goal = _current_agent_goal(
            spec=spec,
            history=history,
            runtime_harness=runtime_harness,
        )
        unmet_requirements: list[str] = []
        context_hints = [
            current_goal,
            runtime_harness,
            unmet_requirements,
            spec["intent"],
            spec["steps"],
            spec["criteria"],
            spec["inputs"],
            history_prompt,
        ]
        compact_candidates = compact_dom_candidates(
            candidates,
            limit=context_limit,
            hints=context_hints,
        )
        dom_context = build_dom_context(
            candidates,
            url=getattr(self.page, "url", ""),
            title=page_title,
            context_level=2,
            limit=context_limit,
            hints=context_hints,
        )
        self.current_dom_context = dom_context
        runtime_harness = self._runtime_harness(
            spec=spec,
            history=history,
            dom_context=dom_context,
            candidate_limit=self.agent_model_candidate_limit,
        )
        unmet_requirements = _runtime_unmet_requirements(
            spec=spec,
            history=history,
            current_url=getattr(self.page, "url", ""),
            dom_context=dom_context,
            phase=runtime_harness.get("phase"),
        )
        current_goal = _current_agent_goal(
            spec=spec,
            history=history,
            runtime_harness=runtime_harness,
        )
        context_hints[0] = current_goal
        context_hints[1] = runtime_harness
        context_hints[2] = unmet_requirements
        logger.info(
            "Agent Harness状态: "
            f"case={case_name} | phase={runtime_harness.get('phase')} "
            f"| phase_categories={runtime_harness.get('phase_categories')}"
        )
        logger.info(
            "Agent页面观察: "
            f"case={case_name} | step={step_index}/{spec['max_steps']} "
            f"| url={getattr(self.page, 'url', '')} "
            f"| title={page_title} "
            f"| candidates={len(candidates)} | compact_candidates={len(compact_candidates)} "
            f"| visible_text={_candidate_text_summary(candidates)}"
        )
        logger.debug(
            "Agent用例观察: "
            f"case={case_name} | step={step_index}/{spec['max_steps']} "
            f"| url={getattr(self.page, 'url', '')} | candidates={len(candidates)}"
        )
        model_dom_context = compact_model_dom_context(
            dom_context,
            candidate_limit=max(0, self.agent_model_candidate_limit),
            selector_limit=1,
            form_limit=max(0, self.agent_model_form_limit),
            assertion_limit=max(0, self.agent_model_assertion_limit),
            hints=context_hints,
            include_business_objects=False,
            include_compression=False,
        )
        project_context = self._agent_project_context_for_prompt(
            spec=spec,
            current_goal=current_goal,
            criteria_prompt=criteria_prompt,
            history=history,
        )
        provider = self._agent_provider()
        stable_prompt_payload = self._stable_agent_prompt_payload(
            case_name=case_name,
            spec=spec,
            criteria_prompt=criteria_prompt,
            project_context=project_context,
        )
        realtime_prompt_payload = self._realtime_agent_prompt_payload(
            step_index=step_index,
            spec=spec,
            current_goal=current_goal,
            history_prompt=history_prompt,
            unmet_requirements=unmet_requirements,
            runtime_harness=runtime_harness,
            model_dom_context=model_dom_context,
        )
        self._last_agent_prompt_metrics = _agent_prompt_metrics(
            stable_prompt_payload=stable_prompt_payload,
            realtime_prompt_payload=realtime_prompt_payload,
            dom_context=model_dom_context,
            history_prompt=history_prompt,
            project_context=project_context,
        )
        logger.info(
            "Agent提示词压缩: "
            f"case={case_name} | stable_chars={self._last_agent_prompt_metrics['stable_chars']} "
            f"| realtime_chars={self._last_agent_prompt_metrics['realtime_chars']} "
            f"| dom_items={self._last_agent_prompt_metrics['dom_items']} "
            f"| history_items={self._last_agent_prompt_metrics['history_items']} "
            f"| harness_phase_categories={self._last_agent_prompt_metrics['harness_phase_categories']}"
        )
        messages = [
            {
                "role": "system",
                "content": _AGENT_CASE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": _json_payload(stable_prompt_payload),
            },
            {
                "role": "user",
                "content": _json_payload(realtime_prompt_payload),
            },
        ]
        try:
            self.last_decision_used_model = True
            return _complete_agent_decision(
                provider,
                messages,
                spec=spec,
                dom_context=dom_context,
                schema_name="AgentCaseDecision",
                usage_operation="runtime.agent_case",
                usage_metadata={
                    "project": self.context.project,
                    "schema_name": "AgentCaseDecision",
                    "prompt_version": self.prompt_version,
                    "page_url": getattr(self.page, "url", ""),
                    "step_index": step_index,
                    **self._last_agent_prompt_metrics,
                },
            )
        except Exception as exc:
            raise

    def _agent_provider(self) -> ChatCompletionProvider:
        if (
            not self.agent_model
            and self.agent_reasoning_effort is None
            and self.agent_timeout_seconds is None
        ):
            return ChatCompletionProvider()
        try:
            settings = load_llm_settings()
            reasoning_effort = (
                None
                if str(self.agent_reasoning_effort or "").lower()
                in {"", "none", "false"}
                else str(self.agent_reasoning_effort)
            )
            timeout_seconds = (
                int(self.agent_timeout_seconds)
                if self.agent_timeout_seconds
                else settings.timeout_seconds
            )
            return ChatCompletionProvider(
                replace(
                    settings,
                    model=self.agent_model or settings.model,
                    reasoning_effort=reasoning_effort,
                    timeout_seconds=timeout_seconds,
                )
            )
        except Exception:
            return ChatCompletionProvider()

    def _guard_decision(
        self,
        decision: AgentCaseDecision,
        *,
        spec: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
    ) -> None:
        if decision.action == "use_module" and not spec.get("module_entry_allowed"):
            raise AgentDecisionRejected(
                "use_module仅允许在用例显式声明项目模块时使用"
            )
        guardrails = spec["guardrails"]
        text = " ".join(
            str(value or "")
            for value in (
                decision.target,
                decision.selector,
                decision.value,
                decision.reason,
            )
        ).lower()
        if (
            guardrails.get("stop_on_unexpected_payment", True)
            and not guardrails.get("allow_test_order_submit", False)
            and any(keyword in text for keyword in _PAYMENT_KEYWORDS)
        ):
            raise ValueError(f"Agent动作疑似触发支付，已拦截: {decision.action}")
        if (
            guardrails.get("stop_on_external_domain", True)
            and decision.action == "goto"
            and decision.value
        ):
            if _is_truncated_current_url_prefix(
                current_url=getattr(self.page, "url", ""),
                next_url=decision.value,
            ):
                raise AgentDecisionRejected(
                    f"goto URL is an unsafe truncated current URL prefix: {decision.value}"
                )
            try:
                fallback_url = self._entry_scope(spec)
            except Exception:
                fallback_url = ""
            if not fallback_url:
                fallback_url = self.context.base_url
            if not _is_external_url(
                current_url=getattr(self.page, "url", ""),
                next_url=decision.value,
                fallback_url=fallback_url,
            ):
                return
            raise AgentDecisionRejected(
                f"Agent动作跳转外部域名，已拦截: {decision.value}"
            )

    def _decision_to_step(
        self, decision: AgentCaseDecision, *, spec: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if decision.action == "use_module":
            step: dict[str, Any] = {"use_module": decision.module}
            module_params = self._module_params_for_decision(decision, spec=spec or {})
            if module_params:
                step["params"] = module_params
            return step
        if decision.action == "wait":
            return {
                "action": "wait",
                "value": str((decision.wait_ms or 1000) / 1000),
            }
        action = "press_key" if decision.action == "press" else decision.action
        if action == "assert_title":
            visible_title_selector = _selector_for_misrouted_page_title_assertion(
                page=self.page,
                dom_context=self.current_dom_context,
                elements=self.elements,
                expected_text=decision.value,
            )
            if visible_title_selector:
                return {
                    "action": "assert_text",
                    "mode": "smart",
                    "selector": visible_title_selector,
                    "value": decision.value,
                }
        elif action == "assert_title_contains":
            visible_title_selector = _selector_for_visible_text(
                self.current_dom_context,
                decision.value,
            )
            if visible_title_selector:
                return {
                    "action": "assert_text_contains",
                    "mode": "smart",
                    "selector": visible_title_selector,
                    "value": decision.value,
                }
        step = {
            "action": action,
            "mode": "smart",
        }
        target_as_element_id = (
            decision.target
            if decision.target
            and selector_for_element_id(self.current_dom_context, decision.target)
            else None
        )
        explicit_selector = (
            decision.selector.strip()
            if isinstance(decision.selector, str) and decision.selector.strip()
            else None
        )
        selected_element_id = decision.element_id or target_as_element_id
        if (
            selected_element_id
            and explicit_selector
            and not looks_like_internal_element_id(explicit_selector)
            and not _selector_matches_element(
                explicit_selector,
                selectors_for_element_id(self.current_dom_context, selected_element_id),
            )
        ):
            logger.warning(
                "Agent返回的element_id与selector不一致，优先使用显式selector: "
                f"element_id={selected_element_id} | selector={explicit_selector}"
            )
            selected_element_id = None
        element_selector = selector_for_element_id(
            self.current_dom_context,
            selected_element_id
            or (
                explicit_selector
                if looks_like_internal_element_id(explicit_selector)
                else None
            ),
        )
        if element_selector:
            step["selector"] = element_selector
        elif selected_element_id:
            raise ValueError(f"Agent返回未知element_id: {selected_element_id}")
        elif looks_like_internal_element_id(decision.target) and not decision.selector:
            raise ValueError(
                f"Agent返回未解析的内部element_id target: {decision.target}"
            )
        if explicit_selector and not element_selector:
            step["selector"] = explicit_selector
        if action == "assert_text":
            if _is_title_metadata_selector(step.get("selector")):
                raise AgentDecisionRejected(
                    "browser title is metadata; do not assert it as visible text"
                )
            assertion_text = decision.value or decision.target
            assertion_selector = _selector_for_assertion_text(
                page=self.page,
                dom_context=self.current_dom_context,
                elements=self.elements,
                expected_text=assertion_text,
                current_selector=step.get("selector"),
            )
            if assertion_selector:
                step["selector"] = assertion_selector
        elif action == "assert_visible" and not step.get("selector"):
            assertion_selector = _selector_for_visible_text(
                self.current_dom_context,
                decision.value or decision.target,
            )
            if assertion_selector:
                step["selector"] = assertion_selector
        if action == "assert_text" and _assert_text_should_use_contains(
            criteria=(spec or {}).get("criteria"),
            expected=decision.value,
        ):
            step["action"] = "assert_text_contains"
        if (
            decision.target
            and decision.target != target_as_element_id
            and not step.get("selector")
        ):
            step["target"] = decision.target
        if decision.value is not None:
            step["value"] = decision.value
        if decision.key:
            step["key"] = decision.key
            step.setdefault("value", decision.key)
        if action in {"click", "press_key"}:
            self._repair_risky_click_step(step, spec=spec or {})
            target_text = _dom_target_text_for_step(
                self.current_dom_context,
                step,
                element_id=selected_element_id,
            )
            if target_text:
                step["_action_target_text"] = normalize_model_text(
                    target_text,
                    limit=80,
                )
        return step

    def _repair_risky_click_step(
        self, step: dict[str, Any], *, spec: dict[str, Any]
    ) -> None:
        selector = str(step.get("selector") or "").strip()
        current = _dom_element_for_step(self.current_dom_context, step)
        current_blob = _normalized_criterion_match_text(_element_primary_blob(current or {}))
        intent_blob = _normalized_criterion_match_text(
            " ".join(
                str(value or "")
                for value in (
                    spec.get("description"),
                    spec.get("intent"),
                    spec.get("steps"),
                    spec.get("criteria"),
                )
            )
        )
        if not _risky_click_mismatch(current_blob=current_blob, intent_blob=intent_blob):
            return
        target = _repair_click_target_from_spec(spec, current_blob=current_blob)
        if not target:
            return
        _, repaired_selector = _best_dom_target_reference(
            self.current_dom_context,
            target,
            actions={"click"},
        )
        if repaired_selector and repaired_selector != selector:
            logger.warning(
                "Agent点击目标与当前业务目标不匹配，已改用更相关候选: "
                f"old={selector} | new={repaired_selector} | target={target}"
            )
            step["selector"] = repaired_selector


    def _safe_collect_candidates(self) -> list[dict[str, Any]]:
        if self._is_blank_page():
            return []
        if not self.native_observe.enabled:
            return []
        try:
            return collect_candidates(
                self.page,
                limit=min(
                    self.agent_candidate_scan_limit,
                    self.native_observe.max_candidates,
                ),
                ignore_selectors=self.native_observe.ignore_selectors,
                include_open_shadow_dom=self.native_observe.include_open_shadow_dom,
            )
        except Exception as exc:
            logger.warning(f"Agent DOM观察失败，继续基于上下文决策: {exc}")
            return []

    def _dom_context_limit(self, candidates: list[dict[str, Any]]) -> int:
        return min(
            len(candidates),
            max(self.agent_candidate_limit * 8, self.agent_candidate_limit),
        )

    def _history_item(
        self,
        *,
        step: dict[str, Any],
        source: str,
        decision: AgentCaseDecision | None = None,
        result: str = "passed",
    ) -> dict[str, Any]:
        return {
            "index": None,
            "source": source,
            "step": _clean_history_step(step),
            "decision": (
                compressed_decision_summary(decision, result=result)
                if decision is not None
                else None
            ),
            "result": result,
            "url_after": getattr(self.page, "url", ""),
        }

    def _cache_key(self, *, case_name: str, spec: dict[str, Any]) -> str:
        return _hash_payload(
            {
                "project": self.context.project,
                "env": self.env,
                "case_name": case_name,
                "entry_scope": self._entry_scope(spec),
                "intent": spec["intent"],
                "steps": spec["steps"],
                "inputs": spec["inputs"],
                "criteria": spec["criteria"],
                "context": context_asset_fingerprint(self.context),
                "prompt_version": self.prompt_version,
                "schema_version": self.schema_version,
            }
        )

    def _entry_scope(self, spec: dict[str, Any]) -> str:
        scope = resolve_entry_scope(
            spec={
                "intent": spec.get("intent"),
                "steps": spec.get("steps"),
                "description": spec.get("description"),
            },
            modules=self.context.modules,
            base_url=str(self.context.base_url or ""),
            spec_source_name="agent_spec",
            priority=[
                "agent_spec.steps_or_intent_url",
                "project_context.module_goto",
                "project_context.base_url",
            ],
        )
        return str(scope.get("normalized_entry_url") or "")

    def _agent_project_context_for_prompt(
        self,
        *,
        spec: dict[str, Any],
        current_goal: str,
        criteria_prompt: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return compact_project_context(
            self.context,
            max_items=min(self.max_context_items, self.agent_prompt_project_items),
            max_modules=max(0, self.agent_prompt_modules),
            max_module_steps=max(0, self.agent_prompt_module_steps),
            hints=[
                spec["intent"],
                current_goal,
                criteria_prompt,
                spec["inputs"],
            ],
            include_modules=False,
            include_elements=False,
        )

    def _runtime_harness(
        self,
        *,
        spec: dict[str, Any],
        history: list[dict[str, Any]],
        dom_context: dict[str, Any],
        candidate_limit: int = 3,
    ) -> dict[str, Any]:
        state = _runtime_harness_state(
            spec=spec,
            history=history,
            dom_context=dom_context,
            current_url=getattr(self.page, "url", ""),
            lookahead=self.agent_harness_lookahead,
        )
        _attach_current_target_candidates(
            state,
            page=self.page,
            dom_context=dom_context,
            limit=max(1, min(int(candidate_limit or 3), 5)),
        )
        return state

    def _stable_agent_prompt_payload(
        self,
        *,
        case_name: str,
        spec: dict[str, Any],
        criteria_prompt: dict[str, Any],
        project_context: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "prompt_section": "stable_context",
            "case": case_name,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "input_type": spec["input_type"],
            "intent": spec["intent"],
            "natural_steps": _agent_natural_plan(spec)[:12],
            "inputs": spec["inputs"],
            "input_contract": (
                "inputs are optional structured supplements; literal values in "
                "intent/natural_steps/criteria are executable values."
            ),
            "criteria": criteria_prompt,
            "limits": {
                "max_steps": spec["max_steps"],
                "max_model_calls": spec["max_model_calls"],
            },
            "project_context": project_context,
            "action_categories": _action_categories_for_prompt(),
            "action_contract": _AGENT_CASE_ACTION_CONTRACT,
            "cache_policy": (
                "This section is stable across loop iterations; current DOM and "
                "execution evidence are only in realtime_state and must be treated as source of truth."
            ),
        }

    def _realtime_agent_prompt_payload(
        self,
        *,
        step_index: int,
        spec: dict[str, Any],
        current_goal: str,
        history_prompt: list[dict[str, Any]],
        unmet_requirements: list[str],
        runtime_harness: dict[str, Any],
        model_dom_context: dict[str, Any],
    ) -> dict[str, Any]:
        plan_status = (
            runtime_harness.get("plan_status")
            if isinstance(runtime_harness, dict)
            else {}
        )
        completed_steps = (
            plan_status.get("completed")
            if isinstance(plan_status, dict)
            else []
        )
        remaining_steps = (
            plan_status.get("remaining")
            if isinstance(plan_status, dict)
            else _remaining_step_hints(spec=spec, history=[], limit=4)
        )
        current_step = (
            plan_status.get("current")
            if isinstance(plan_status, dict)
            else (remaining_steps[0] if remaining_steps else "verify completion criteria")
        )
        return {
            "prompt_section": "realtime_state",
            "url": getattr(self.page, "url", ""),
            "current_step": current_step,
            "completed_steps": completed_steps,
            "remaining_steps": remaining_steps,
            "execution_evidence": _execution_evidence(history_prompt),
            "current_goal": current_goal,
            "step_index": step_index,
            "runtime_harness": _runtime_harness_prompt_view(runtime_harness),
            "unmet_requirements": unmet_requirements,
            "dom_context": model_dom_context,
        }

    def _module_params_for_decision(
        self,
        decision: AgentCaseDecision,
        *,
        spec: dict[str, Any],
    ) -> dict[str, Any]:
        provided = decision.params if isinstance(decision.params, dict) else {}
        params, missing = self._infer_module_params(
            str(decision.module or ""),
            spec=spec,
            provided_params=provided,
        )
        if missing:
            raise AgentDecisionRejected("模块缺少params: " + ", ".join(sorted(missing)))
        if params != provided:
            logger.warning(
                "Agent模块参数已根据模块占位符和inputs补齐: "
                f"module={decision.module} | params={sorted(params)}"
            )
        return params

    def _infer_module_params(
        self,
        module_name: str,
        *,
        spec: dict[str, Any],
        provided_params: dict[str, Any],
    ) -> tuple[dict[str, Any], set[str]]:
        params = dict(provided_params or {})
        module_steps = _module_steps_from_value(
            (self.context.modules or {}).get(module_name)
        )
        if not module_steps:
            return params, set()
        refs = _module_placeholder_refs(module_steps)
        known_variables = set((self.context.variables or {}).keys())
        missing = set(refs) - known_variables - set(params.keys())
        if not missing:
            return params, set()
        input_values = _flatten_input_values(spec.get("inputs") or {})
        for name in sorted(missing):
            matched = _input_value_for_param(name, input_values)
            if matched is not None:
                params[name] = matched
        unresolved = set(refs) - known_variables - set(params.keys())
        return params, unresolved

    @staticmethod
    def _model_cache_key() -> str:
        try:
            return load_llm_settings().model
        except Exception:
            return os.environ.get("LLM_MODEL", "")

    def _is_blank_page(self) -> bool:
        return (getattr(self.page, "url", "") or "about:blank") == "about:blank"


def _has_criteria(criteria: Any) -> bool:
    if not isinstance(criteria, dict):
        return False
    for field_name in ("checkpoints", "final"):
        value = criteria.get(field_name)
        if isinstance(value, list) and any(str(item).strip() for item in value):
            return True
    return False


def _has_final_criteria(criteria: Any) -> bool:
    if not isinstance(criteria, dict):
        return False
    value = criteria.get("final")
    return isinstance(value, list) and any(str(item).strip() for item in value)


def _has_completion_criteria(criteria: Any) -> bool:
    return bool(_completion_criteria_items(criteria))


def _completion_criteria_items(criteria: Any) -> list[str]:
    if not isinstance(criteria, dict):
        return []
    return _dedupe_texts(
        _normalize_text_list(criteria.get("checkpoints"))
        + _normalize_text_list(criteria.get("final"))
    )


def _observable_completion_terms(criteria: Any) -> tuple[list[str], list[str]]:
    title_terms: list[str] = []
    url_terms: list[str] = []
    for criterion in _completion_criteria_items(criteria):
        text = str(criterion or "").lower()
        quoted_terms = _quoted_evidence_terms(criterion)
        terms = quoted_terms or _criterion_evidence_terms(criterion)
        if _criterion_requires_title(text):
            title_terms.extend(terms)
        if _criterion_requires_url(text):
            url_terms.extend(terms)
    return _dedupe_texts(title_terms), _dedupe_texts(url_terms)


def _quoted_evidence_terms(criterion: str) -> list[str]:
    return [
        term.strip()
        for term in re.findall(
            r"[\"'“”‘’]([^\"'“”‘’]{2,80})[\"'“”‘’]", str(criterion or "")
        )
        if term.strip()
    ]


def _dedupe_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _criteria_summary(criteria: Any) -> str:
    if not isinstance(criteria, dict):
        return "checkpoints=[] | final=[]"
    checkpoints = _short_list(criteria.get("checkpoints"))
    final = _short_list(criteria.get("final"))
    return f"checkpoints={checkpoints} | final={final}"


def _criteria_prompt_summary(
    criteria: Any,
    *,
    checkpoint_limit: int = 4,
    final_limit: int = 4,
) -> dict[str, Any]:
    if not isinstance(criteria, dict):
        return {"checkpoints": [], "final": []}
    return {
        "checkpoints": _short_list(criteria.get("checkpoints"), limit=checkpoint_limit),
        "final": _short_list(criteria.get("final"), limit=final_limit),
    }


def _history_used_module(history: list[dict[str, Any]], module_name: str) -> bool:
    expected = str(module_name or "").strip()
    if not expected:
        return False
    for item in _iter_history_items(history):
        if not isinstance(item, dict) or item.get("result") != "passed":
            continue
        step = item.get("step")
        if not isinstance(step, dict):
            continue
        used = step.get("use_module")
        if str(used or "").strip() == expected:
            return True
    return False


def _module_steps_from_value(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        steps = value.get("steps")
        if isinstance(steps, list):
            return [step for step in steps if isinstance(step, dict)]
        if len(value) == 1:
            return _module_steps_from_value(next(iter(value.values())))
        return []
    if isinstance(value, list):
        return [step for step in value if isinstance(step, dict)]
    return []


def _module_placeholder_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        for item in value.values():
            refs.update(_module_placeholder_refs(item))
        return refs
    if isinstance(value, list):
        for item in value:
            refs.update(_module_placeholder_refs(item))
        return refs
    if isinstance(value, str):
        refs.update(
            match.strip()
            for match in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_.-]*)\}", value)
            if match.strip()
        )
    return refs


def _input_value_for_param(
    param_name: str,
    input_values: list[tuple[str, Any]],
) -> Any:
    normalized_param = _normalize_identifier_words(param_name)
    best: tuple[int, Any] | None = None
    tied = False
    for path, value in input_values:
        leaf = str(path or "").split(".")[-1]
        score = 0
        if path == param_name:
            score = 100
        elif leaf == param_name:
            score = 90
        else:
            score = _score_input_match(param_name, path)
            if normalized_param and normalized_param == _normalize_identifier_words(
                leaf
            ):
                score = max(score, 80)
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, value)
            tied = False
        elif score == best[0]:
            tied = True
    return None if best is None or tied else best[1]


def _runtime_unmet_requirements(
    *,
    spec: dict[str, Any],
    history: list[dict[str, Any]],
    current_url: str,
    dom_context: dict[str, Any],
    phase: str | None = None,
) -> list[str]:
    phase_text = str(phase or "").strip()
    if phase_text:
        return _dedupe_texts(
            _unmet_phase_requirements(
                phase_text,
                spec=spec,
                history=history,
                current_url=current_url,
                dom_context=dom_context,
            )
        )
    return _dedupe_texts(
        _unmet_completion_criteria(
            criteria=spec.get("criteria"),
            history=history,
            current_url=current_url,
            dom_context=dom_context,
        )
        + _unmet_intent_action_requirements(
            spec=spec,
            history=history,
            current_url=current_url,
            dom_context=dom_context,
        )
    )


def _unmet_phase_requirements(
    phase: str,
    *,
    spec: dict[str, Any],
    history: list[dict[str, Any]],
    current_url: str,
    dom_context: dict[str, Any],
) -> list[str]:
    phase_text = str(phase or "").strip()
    if not phase_text or phase_text == "verify completion criteria":
        return _unmet_completion_criteria(
            criteria=spec.get("criteria"),
            history=history,
            current_url=current_url,
            dom_context=dom_context,
        )
    if _natural_step_satisfied(
        phase_text,
        spec=spec,
        history=history,
        dom_context={
            **(dom_context or {}),
            "meta": {
                **(
                    ((dom_context or {}).get("meta") or {})
                    if isinstance((dom_context or {}).get("meta"), dict)
                    else {}
                ),
                "url": current_url,
            },
        },
    ):
        if not _next_plan_target_observable(
            last_satisfied=phase_text,
            spec=spec,
            dom_context=dom_context,
            current_url=current_url,
        ):
            return [phase_text]
        return []
    requirements = _intent_action_requirements(
        {"steps": [phase_text], "inputs": spec.get("inputs")}
    )
    if requirements:
        return [
            label
            for action, terms, label, source_text in requirements
            if label
            in _unmet_intent_action_requirements(
                spec={"steps": [source_text], "inputs": spec.get("inputs")},
                history=history,
                current_url=current_url,
                dom_context=dom_context,
            )
        ]
    return [phase_text]


def _runtime_harness_state(
    *,
    spec: dict[str, Any],
    history: list[dict[str, Any]],
    dom_context: dict[str, Any],
    current_url: str,
    lookahead: int = 4,
) -> dict[str, Any]:
    phase = _runtime_harness_phase(
        spec=spec,
        history=history,
        dom_context=dom_context,
        current_url=current_url,
    )
    plan_status = _runtime_plan_status(
        spec=spec,
        history=history,
        dom_context=dom_context,
        current_url=current_url,
        phase=phase,
        lookahead=lookahead,
    )
    return {
        "phase": phase,
        "plan_status": plan_status,
        "phase_categories": _phase_action_categories(phase),
        "phase_observation": _phase_observation(
            phase=phase,
            spec=spec,
            history=history,
            dom_context=dom_context,
            current_url=current_url,
        ),
        "feedback": _runtime_feedback(
            history=history,
            dom_context=dom_context,
            current_url=current_url,
        ),
        "rule": (
            "phase_categories are informational only; choose any schema-valid "
            "action that fits realtime DOM/history and safety guardrails."
        ),
    }


def _runtime_feedback(
    *,
    history: list[dict[str, Any]],
    dom_context: dict[str, Any],
    current_url: str,
) -> dict[str, Any]:
    return {
        "last_action": _last_action_feedback(history),
        "filled_fields": _filled_field_feedback(history),
        "visible_errors": _visible_error_texts(dom_context),
        "stalled_on_url": _stalled_action_count(history, current_url=current_url),
    }


def _last_action_feedback(history: list[dict[str, Any]]) -> dict[str, Any]:
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        step = item.get("step")
        if not isinstance(step, dict):
            continue
        feedback: dict[str, Any] = {
            "action": _step_action(step),
            "target": step.get("selector") or step.get("target") or step.get("use_module"),
            "result": item.get("result", "passed"),
        }
        page_errors = step.get("_action_page_errors")
        if isinstance(page_errors, list) and page_errors:
            feedback["page_errors"] = _short_list(page_errors, limit=2)
        before_url = str(step.get("_action_before_url") or "")
        after_url = str(step.get("_action_after_url") or item.get("url_after") or "")
        if before_url or after_url:
            feedback["url_changed"] = _url_state_changed(before_url, after_url)
        if step.get("_action_dom_changed") is not None:
            feedback["dom_changed"] = bool(step.get("_action_dom_changed"))
        if step.get("_action_target_text"):
            feedback["target_text"] = step.get("_action_target_text")
        if step.get("_action_executed_selector"):
            feedback["executed_selector"] = step.get("_action_executed_selector")
        if step.get("_action_target_visible_after") is not None:
            feedback["target_visible_after"] = bool(
                step.get("_action_target_visible_after")
            )
        return feedback
    return {}


def _phase_observation(
    *,
    phase: str,
    spec: dict[str, Any],
    history: list[dict[str, Any]],
    dom_context: dict[str, Any],
    current_url: str,
) -> dict[str, Any]:
    phase_text = str(phase or "").strip()
    payload = _natural_step_decision_payload(
        phase_text,
        spec=spec,
        dom_context=dom_context,
        history=history,
    )
    target = _phase_payload_target(payload, phase_text)
    result: dict[str, Any] = {
        "target": normalize_model_text(target, limit=80) if target else "",
        "expected_action": str(payload.get("action") or "").strip(),
        "target_observable": _natural_step_target_observable(
            phase_text,
            spec=spec,
            dom_context=dom_context,
            current_url=current_url,
        ),
        "satisfied_by_history": _natural_step_satisfied(
            phase_text,
            spec=spec,
            history=history,
            dom_context={
                **(dom_context or {}),
                "meta": {
                    **(
                        ((dom_context or {}).get("meta") or {})
                        if isinstance((dom_context or {}).get("meta"), dict)
                        else {}
                    ),
                    "url": current_url,
                },
            },
        ),
    }
    next_step = _next_natural_step_after(phase_text, spec)
    if next_step:
        result["next_phase"] = normalize_model_text(next_step, limit=80)
        result["next_target_observable"] = _natural_step_target_observable(
            next_step,
            spec=spec,
            dom_context=dom_context,
            current_url=current_url,
        )
    return {key: value for key, value in result.items() if value not in (None, "")}


def _runtime_plan_status(
    *,
    spec: dict[str, Any],
    history: list[dict[str, Any]],
    dom_context: dict[str, Any],
    current_url: str,
    phase: str,
    lookahead: int,
) -> dict[str, Any]:
    plan = _agent_natural_plan(spec)
    if not plan:
        return {"completed_count": len(history), "current": phase}
    dom_with_url = {
        **(dom_context or {}),
        "meta": {
            **(
                ((dom_context or {}).get("meta") or {})
                if isinstance((dom_context or {}).get("meta"), dict)
                else {}
            ),
            "url": current_url,
        },
    }
    completed_count = 0
    for step_text in plan:
        if not _natural_step_satisfied(
            step_text,
            spec=spec,
            history=history,
            dom_context=dom_with_url,
        ):
            break
        completed_count += 1
    current_index = min(completed_count + 1, len(plan)) if completed_count < len(plan) else None
    remaining_start = completed_count
    remaining_end = min(len(plan), remaining_start + max(1, int(lookahead or 1)))
    completed_steps = [
        normalize_model_text(item, limit=80)
        for item in plan[:completed_count]
    ]
    remaining_steps = [
        normalize_model_text(item, limit=80)
        for item in plan[remaining_start:remaining_end]
    ]
    status: dict[str, Any] = {
        "total": len(plan),
        "completed_count": completed_count,
        "completed": completed_steps,
        "remaining": remaining_steps,
    }
    if current_index is not None:
        status["current_index"] = current_index
        status["current"] = normalize_model_text(plan[current_index - 1], limit=80)
    else:
        status["current"] = "verify completion criteria"
    return status


def _execution_evidence(history_prompt: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    if not isinstance(history_prompt, list):
        return evidence
    for item in history_prompt[-4:]:
        if not isinstance(item, dict):
            continue
        step = item.get("step")
        if not isinstance(step, dict):
            continue
        row: dict[str, Any] = {
            "action": _step_action(step),
            "result": item.get("result", "passed"),
        }
        for source_key, target_key in (
            ("selector", "selector"),
            ("target", "target"),
            ("value", "value"),
            ("_action_executed_selector", "executed_selector"),
            ("_action_target_text", "target_text"),
            ("_action_before_url", "before_url"),
            ("_action_after_url", "after_url"),
            ("_resolved_value", "resolved_value"),
            ("_resolved_value_after", "resolved_value"),
        ):
            value = step.get(source_key)
            if value not in (None, "", []):
                row[target_key] = normalize_model_text(value, limit=100)
        if step.get("_action_dom_changed") is not None:
            row["dom_changed"] = bool(step.get("_action_dom_changed"))
        evidence.append(row)
    return evidence


def _runtime_harness_prompt_view(runtime_harness: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(runtime_harness, dict):
        return {}
    result: dict[str, Any] = {}
    for key in ("phase", "phase_categories", "phase_observation", "feedback"):
        value = runtime_harness.get(key)
        if value not in (None, "", [], {}):
            result[key] = value
    return result


def _attach_current_target_candidates(
    runtime_harness: dict[str, Any],
    *,
    page: Any,
    dom_context: dict[str, Any],
    limit: int = 3,
) -> None:
    if not isinstance(runtime_harness, dict):
        return
    observation = runtime_harness.get("phase_observation")
    if not isinstance(observation, dict):
        return
    target = str(observation.get("target") or "").strip()
    if not target:
        return
    categories = runtime_harness.get("phase_categories")
    action = (
        "fill"
        if isinstance(categories, list) and "input" in categories
        else "click"
    )
    candidates = _verified_target_candidates(
        page=page,
        dom_context=dom_context,
        target=target,
        action=action,
        limit=max(1, limit),
    )
    if candidates:
        observation["target_candidates"] = candidates
        if observation.get("target_observable") is False:
            observation["target_observable"] = True


def _verified_target_candidates(
    *,
    page: Any,
    dom_context: dict[str, Any],
    target: str,
    action: str,
    limit: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(
        selector: str,
        *,
        source: str,
        element_id: str = "",
        text: str = "",
        score: int = 0,
    ) -> None:
        clean_selector = str(selector or "").strip()
        if not clean_selector or clean_selector in seen:
            return
        if hasattr(page, "locator") and not _selector_is_visible_enabled(
            page,
            clean_selector,
        ):
            return
        seen.add(clean_selector)
        item: dict[str, Any] = {"selector": clean_selector, "source": source}
        if element_id:
            item["element_id"] = element_id
        if text:
            item["text"] = normalize_model_text(text, limit=80)
        if score:
            item["match_score"] = score
        result.append(item)

    terms = _intent_action_terms(target)
    dom_matches: list[tuple[int, str, str, str]] = []
    for item in _iter_dom_elements(dom_context):
        item_id = str(item.get("id") or "")
        selector = selector_for_element_id(dom_context, item_id)
        if not selector:
            continue
        blob = _normalized_criterion_match_text(_element_primary_blob(item))
        score = _target_candidate_match_score(terms, blob)
        if score <= 0:
            continue
        dom_matches.append((score, selector, item_id, _element_display_text(item)))
    for score, selector, item_id, text in sorted(
        dom_matches,
        key=lambda match: match[0],
        reverse=True,
    ):
        if len(result) >= limit:
            break
        add(
            selector,
            source="dom_context",
            element_id=item_id,
            text=text,
            score=score,
        )
    if len(result) >= limit:
        return result[:limit]

    selector_candidates: list[str] = []
    if hasattr(page, "locator"):
        try:
            selector_candidates.extend(
                semantic_selectors(
                    page,
                    target,
                    action,
                    limit=max(limit * 4, 12),
                )
            )
        except Exception:
            pass
    selector_candidates.extend(heuristic_selectors(target, action))
    for selector in _dedupe_strings(selector_candidates):
        if len(result) >= limit:
            break
        selector_text = _text_query_from_selector(selector)
        add(
            selector,
            source="verified_selector",
            text=selector_text,
            score=_target_candidate_match_score(
                terms,
                _normalized_criterion_match_text(selector_text or selector),
            ),
        )
    return result[:limit]


def _target_candidate_match_score(terms: list[str], blob: str) -> int:
    normalized_blob = _normalized_criterion_match_text(blob)
    if not normalized_blob:
        return 0
    score = 0
    for index, term in enumerate(terms):
        normalized_term = _normalized_criterion_match_text(term)
        if not normalized_term:
            continue
        compact_term = normalized_term.replace(" ", "")
        compact_blob = normalized_blob.replace(" ", "")
        if normalized_term in normalized_blob:
            score += (80 if index == 0 else 30) + min(len(normalized_term), 20)
        elif compact_term and compact_term in compact_blob:
            score += (70 if index == 0 else 24) + min(len(compact_term), 20)
    return score


def _phase_payload_target(payload: dict[str, Any], fallback: str) -> str:
    if not isinstance(payload, dict):
        return fallback
    return str(
        payload.get("target")
        or payload.get("value")
        or payload.get("module")
        or fallback
        or ""
    ).strip()


def _next_natural_step_after(phase: str, spec: dict[str, Any]) -> str:
    plan = _agent_natural_plan(spec)
    try:
        index = plan.index(phase)
    except ValueError:
        return ""
    for item in plan[index + 1 :]:
        text = str(item or "").strip()
        if text:
            return text
    return ""


def _filled_field_feedback(
    history: list[dict[str, Any]],
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    filled: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in reversed(history):
        if not isinstance(item, dict) or item.get("result") not in {None, "passed"}:
            continue
        step = item.get("step")
        if not isinstance(step, dict) or _step_action(step).lower() != "fill":
            continue
        target = str(step.get("selector") or step.get("target") or "").strip()
        value = str(
            step.get("_resolved_value")
            or step.get("_resolved_value_after")
            or step.get("value")
            or ""
        ).strip()
        if not target:
            continue
        key = (_normalize_selector_for_compare(target), value)
        if key in seen:
            continue
        seen.add(key)
        filled.append(
            {
                "target": normalize_model_text(target, limit=80),
                "value": normalize_model_text(value, limit=80),
            }
        )
        if len(filled) >= limit:
            break
    filled.reverse()
    return filled


def _visible_error_texts(dom_context: dict[str, Any]) -> list[str]:
    if not isinstance(dom_context, dict):
        return []
    errors: list[str] = []
    for item in _iter_dom_elements(dom_context):
        if not isinstance(item, dict):
            continue
        text = _direct_error_text(item)
        blob = _normalized_goal_text(text or _element_primary_blob(item))
        if not _blob_has_any(
            blob,
            (
                "错误",
                "失败",
                "异常",
                "不能为空",
                "必须",
                "账号或密码错误",
                "error",
                "failed",
                "invalid",
                "required",
            ),
        ):
            continue
        if text:
            errors.append(str(text))
    return _short_list(_dedupe_texts(errors), limit=3)


def _direct_error_text(item: dict[str, Any]) -> str:
    parts = [
        str(item.get(key) or "").strip()
        for key in ("text", "name", "label", "placeholder")
        if str(item.get(key) or "").strip()
    ]
    attributes = item.get("attributes")
    if isinstance(attributes, dict):
        parts.extend(
            str(attributes.get(key) or "").strip()
            for key in ("aria-label", "title")
            if str(attributes.get(key) or "").strip()
        )
    return " ".join(_dedupe_texts(parts))


def _stalled_action_count(
    history: list[dict[str, Any]],
    *,
    current_url: str,
) -> int:
    normalized_current = _normalized_url_for_stall(current_url)
    if not normalized_current:
        return 0
    count = 0
    for item in reversed(history):
        if not isinstance(item, dict) or item.get("result") != "passed":
            continue
        step = item.get("step")
        if not isinstance(step, dict):
            continue
        action = _step_action(step).lower()
        if action not in {"click", "fill", "press", "press_key", "wait"}:
            break
        after_url = _normalized_url_for_stall(item.get("url_after"))
        if after_url != normalized_current:
            break
        count += 1
    return count


def _normalized_url_for_stall(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
    return text.split("?")[0].rstrip("/")


def _runtime_harness_phase(
    *,
    spec: dict[str, Any],
    history: list[dict[str, Any]],
    dom_context: dict[str, Any],
    current_url: str,
) -> str:
    last_satisfied = ""
    for text in _agent_natural_plan(spec):
        step_text = str(text or "").strip()
        if not step_text:
            continue
        if not _natural_step_satisfied(
            step_text,
            spec=spec,
            history=history,
            dom_context={
                **(dom_context or {}),
                "meta": {
                    **(((dom_context or {}).get("meta") or {}) if isinstance((dom_context or {}).get("meta"), dict) else {}),
                    "url": current_url,
                },
            },
        ):
            if last_satisfied and not _natural_step_target_observable(
                step_text,
                spec=spec,
                dom_context=dom_context,
                current_url=current_url,
            ) and _has_pending_observable_spec_input(
                spec=spec,
                history=history,
                dom_context=dom_context,
            ):
                return last_satisfied
            return step_text
        last_satisfied = step_text
    if (
        last_satisfied
        and not _next_plan_target_observable(
            last_satisfied=last_satisfied,
            spec=spec,
            dom_context=dom_context,
            current_url=current_url,
        )
        and _has_pending_observable_spec_input(
            spec=spec,
            history=history,
            dom_context=dom_context,
        )
    ):
        return last_satisfied
    unmet = _runtime_unmet_requirements(
        spec=spec,
        history=history,
        current_url=current_url,
        dom_context=dom_context,
    )
    return unmet[0] if unmet else "verify completion criteria"


def _has_pending_observable_spec_input(
    *,
    spec: dict[str, Any],
    history: list[dict[str, Any]],
    dom_context: dict[str, Any],
) -> bool:
    for key, value in _flatten_input_values(spec.get("inputs") or {}):
        if _fill_history_has_input_value(history, key=key, value=value):
            continue
        element_id, _ = _best_fill_reference_for_input_value(
            dom_context,
            key=key,
            value=str(value),
            history=history,
        )
        if element_id:
            return True
    return False


def _next_plan_target_observable(
    *,
    last_satisfied: str,
    spec: dict[str, Any],
    dom_context: dict[str, Any],
    current_url: str,
) -> bool:
    plan = _agent_natural_plan(spec)
    try:
        start = plan.index(last_satisfied) + 1
    except ValueError:
        return True
    for next_text in plan[start:]:
        step_text = str(next_text or "").strip()
        if not step_text:
            continue
        return _natural_step_target_observable(
            step_text,
            spec=spec,
            dom_context=dom_context,
            current_url=current_url,
        )
    return True


def _natural_step_target_observable(
    step_text: str,
    *,
    spec: dict[str, Any],
    dom_context: dict[str, Any],
    current_url: str,
) -> bool:
    text = str(step_text or "").strip()
    if not text:
        return True
    url = _first_url(text)
    if url and _looks_like_navigation_instruction(text):
        return _url_matches_expected(current_url, url)
    fill_target, _ = _fill_instruction_parts(text, spec=spec)
    if fill_target:
        return bool(
            _best_dom_target_reference(dom_context, fill_target, actions={"fill"})[0]
        )
    if re.search(r"(?:输入|填写|填入|录入|fill|type|enter|input)", text, flags=re.I):
        input_values = _flatten_input_values(spec.get("inputs") or {})
        mentioned = _mentioned_input_values(text, input_values)
        if mentioned:
            return any(
                _best_fill_reference_for_input_value(
                    dom_context,
                    key=key,
                    value=str(value),
                    history=[],
                )[0]
                for key, value in mentioned
            )
        return any(_dom_item_is_fillable(item) for item in _iter_dom_elements(dom_context))
    click_target = _click_instruction_target(text) or _implicit_interaction_target(text)
    if click_target:
        if _target_text_visible(dom_context, click_target):
            return True
        return bool(
            _best_dom_target_reference(
                dom_context,
                click_target,
                actions={"click", "press"},
            )[0]
        )
    title_value = _title_assertion_value(text)
    if title_value:
        return _page_title_contains(dom_context, title_value)
    return _target_text_visible(dom_context, text)


def _history_or_current_url_satisfies(
    *,
    history: list[dict[str, Any]],
    current_url: str,
    expected_url: str,
) -> bool:
    expected = str(expected_url or "").strip()
    if not expected:
        return False
    urls = [str(current_url or "")]
    for item in _iter_history_items(history):
        if not isinstance(item, dict):
            continue
        urls.append(str(item.get("url_after") or ""))
        step = item.get("step")
        if isinstance(step, dict) and _step_action(step).lower() == "goto":
            urls.append(str(step.get("value") or ""))
    return any(_url_matches_expected(url, expected) for url in urls if url)


def _url_matches_expected(actual: str, expected: str) -> bool:
    actual_text = str(actual or "")
    expected_text = str(expected or "")
    if not actual_text or not expected_text:
        return False
    if expected_text in actual_text:
        return True
    parsed_actual = urlparse(actual_text)
    parsed_expected = urlparse(expected_text)
    if parsed_actual.netloc and parsed_expected.netloc:
        return (
            parsed_actual.netloc == parsed_expected.netloc
            and parsed_actual.path.rstrip("/") == parsed_expected.path.rstrip("/")
        )
    return False


def _short_list(value: Any, *, limit: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    result = [str(item).strip() for item in value if str(item).strip()]
    if len(result) <= limit:
        return result
    return result[:limit] + [f"...(+{len(result) - limit})"]


def _action_categories_for_prompt() -> dict[str, list[str]]:
    return {
        category: list(actions)
        for category, actions in _ACTION_CATEGORIES.items()
    }


def _phase_action_categories(phase: str) -> list[str]:
    text = str(phase or "")
    normalized = _normalized_goal_text(text)
    categories: list[str] = []
    if _first_url(text) or _looks_like_navigation_instruction(text):
        categories.append("navigation")
    if re.search(r"(?:输入|填写|填入|录入|fill|type)", text, flags=re.I):
        categories.append("input")
    if re.search(
        r"(?:点击|单击|点选|按下|查看|打开|click|press)",
        text,
        flags=re.I,
    ):
        categories.append("interaction")
    if _phase_mentions_assertion(normalized):
        categories.append("assertion")
    if not categories:
        categories.append("interaction")
    return _dedupe_texts(categories)


def _phase_mentions_assertion(normalized_phase: str) -> bool:
    return any(
        term in normalized_phase
        for term in (
            "断言",
            "验证",
            "校验",
            "标题",
            "title",
            "url",
            "地址",
            "可见",
            "显示",
            "展示",
            "包含",
            "存在",
            "assert",
            "visible",
            "contains",
        )
    )


def _agent_prompt_metrics(
    *,
    stable_prompt_payload: dict[str, Any],
    realtime_prompt_payload: dict[str, Any],
    dom_context: dict[str, Any],
    history_prompt: list[dict[str, Any]],
    project_context: dict[str, Any],
) -> dict[str, Any]:
    stable_json = _json_payload(stable_prompt_payload)
    realtime_json = _json_payload(realtime_prompt_payload)
    dom_items = 0
    if isinstance(dom_context, dict):
        for key in ("forms", "interactive_elements", "assertion_candidates"):
            values = dom_context.get(key)
            if isinstance(values, list):
                dom_items += len(values)
    project_modules = project_context.get("modules")
    runtime_harness = realtime_prompt_payload.get("runtime_harness")
    phase_categories = (
        runtime_harness.get("phase_categories")
        if isinstance(runtime_harness, dict)
        else []
    )
    prompt_chars = (
        len(_AGENT_CASE_SYSTEM_PROMPT) + len(stable_json) + len(realtime_json)
    )
    return {
        "stable_chars": len(stable_json),
        "realtime_chars": len(realtime_json),
        "prompt_chars": prompt_chars,
        "dom_items": dom_items,
        "history_items": len(history_prompt),
        "harness_phase_categories": (
            len(phase_categories) if isinstance(phase_categories, list) else 0
        ),
        "project_modules": (
            len(project_modules) if isinstance(project_modules, dict) else 0
        ),
    }


def _agent_natural_plan(spec: dict[str, Any]) -> list[str]:
    natural_steps = _normalize_text_list(spec.get("steps"))
    if natural_steps:
        return natural_steps
    return _intent_to_natural_steps(str(spec.get("intent") or ""))


def _intent_to_natural_steps(intent: str) -> list[str]:
    text = str(intent or "").strip()
    if not text:
        return []
    urls: list[str] = []

    def protect_url(match: re.Match[str]) -> str:
        urls.append(match.group(0))
        return f" __URL_{len(urls) - 1}__ "

    protected = _URL_RE.sub(protect_url, text)
    normalized = re.sub(r"[，,]\s*(?:然后|并|再|且)", "，", text)
    normalized = re.sub(r"[，,]\s*(?:然后|并|再|且)", "，", protected)
    normalized = re.sub(r"(?:\s+then\s+|\s+and\s+then\s+)", "，", normalized, flags=re.I)
    parts: list[str] = []
    for part in re.split(r"[，。,.;；\n]", normalized):
        restored = part.strip()
        if not restored:
            continue
        for index, url in enumerate(urls):
            restored = restored.replace(f"__URL_{index}__", url)
        if restored:
            parts.append(restored)
    return parts[:12]


def _natural_step_satisfied(
    step_text: str,
    *,
    spec: dict[str, Any] | None = None,
    history: list[dict[str, Any]],
    dom_context: dict[str, Any] | None = None,
) -> bool:
    text = str(step_text or "")
    if _module_instruction_satisfied(text, history=history):
        return True
    spec = spec or {}
    if _compound_fill_instruction_satisfied(text, spec=spec, history=history):
        return True
    payload = _natural_step_decision_payload(
        step_text,
        spec=spec,
        dom_context=dom_context or {},
        history=history,
    )
    if not payload:
        return _module_instruction_satisfied(step_text, history=history)
    action = str(payload.get("action") or "").lower()
    if action == "goto":
        expected_url = str(payload.get("value") or "")
        return _history_or_current_url_satisfies(
            history=history,
            current_url=str(((dom_context or {}).get("meta") or {}).get("url") or ""),
            expected_url=expected_url,
        )
    if action == "use_module":
        return _history_used_module(history, str(payload.get("module") or ""))
    if action == "fill":
        terms = _intent_action_terms(payload.get("target") or step_text)
        value = str(payload.get("value") or "")
        return _history_action_satisfies(history, actions={"fill"}, terms=terms) or (
            bool(value)
            and _history_action_satisfies(history, actions={"fill"}, terms=[value])
        )
    if action in {"click", "press"}:
        target = payload.get("target") or step_text
        if _auth_submit_history_satisfies_target(history, target=target):
            return True
        return _history_action_satisfies(
            history,
            actions={"click", "press", "press_key"},
            terms=_intent_action_terms(target),
            require_observed_progress=False,
        ) or _latest_click_transition_satisfies_target(
            history,
            target=target,
        )
    if action in {"assert_title", "assert_title_contains"}:
        value = str(payload.get("value") or "")
        if value and _page_title_contains(dom_context or {}, value):
            return True
        return _history_action_satisfies(
            history,
            actions={"assert_title", "assert_title_contains"},
            terms=_intent_action_terms(value or step_text),
        ) or _history_assertion_has(
            history,
            value_contains=value,
        )
    if action == "assert_text":
        return _history_assertion_has(
            history,
            value_contains=str(payload.get("value") or ""),
        )
    return False


def _module_instruction_satisfied(
    step_text: str,
    *,
    history: list[dict[str, Any]],
) -> bool:
    module_name, explicit = _module_name_from_entry_text(step_text, modules={})
    if explicit and module_name:
        return _history_used_module(history, module_name)
    return False


def _compound_fill_instruction_satisfied(
    text: str,
    *,
    spec: dict[str, Any],
    history: list[dict[str, Any]],
) -> bool:
    if not re.search(r"(?:输入|填写|填入|录入|fill|type|enter)", text, flags=re.I):
        return False
    input_values = _flatten_input_values(spec.get("inputs") or {})
    mentioned = _mentioned_input_values(text, input_values)
    return bool(mentioned) and all(
        _fill_history_has_input_value(history, key=key, value=value)
        for key, value in mentioned
    )


def _latest_click_transition_satisfies_target(
    history: list[dict[str, Any]],
    *,
    target: Any,
) -> bool:
    terms = _intent_action_terms(target)
    previous_url = ""
    for item in reversed(_expanded_history_items(history)):
        if not isinstance(item, dict) or item.get("result") not in {None, "passed"}:
            continue
        step = item.get("step")
        if not isinstance(step, dict):
            continue
        url_after = str(item.get("url_after") or "")
        action = _step_action(step).lower()
        if action in {"click", "press", "press_key"}:
            if not _history_step_matches_terms(step, terms):
                return False
            return _history_step_has_observed_progress(step) or _url_state_changed(
                previous_url,
                url_after,
            )
        if not previous_url and url_after:
            previous_url = url_after
    return False


def _auth_submit_history_satisfies_target(
    history: list[dict[str, Any]],
    *,
    target: Any,
) -> bool:
    target_text = _normalized_goal_text(target)
    if not target_text or not _blob_has_any(
        target_text,
        ("登录", "登陆", "login", "sign in"),
    ):
        return False
    if _blob_has_any(target_text, ("oa",)):
        return False
    saw_username = False
    saw_password = False
    for item in _iter_history_items(history):
        if not isinstance(item, dict) or item.get("result") not in {None, "passed"}:
            continue
        step = item.get("step")
        if not isinstance(step, dict):
            continue
        action = _step_action(step).lower()
        if action == "fill":
            blob = _normalized_goal_text(_history_step_semantic_blob(step))
            if _blob_has_any(blob, ("username", "user name", "账号", "账户", "用户名")):
                saw_username = True
            if _blob_has_any(blob, ("password", "pwd", "密码")):
                saw_password = True
            continue
        if action not in {"click", "press", "press_key"}:
            continue
        if not (saw_username and saw_password):
            continue
        before_url = str(step.get("_action_before_url") or "")
        after_url = str(step.get("_action_after_url") or item.get("url_after") or "")
        auth_route_changed = _url_looks_like_login(before_url) and not _url_looks_like_login(after_url)
        if not _history_step_has_observed_progress_in_history(history, item) and not auth_route_changed:
            continue
        if auth_route_changed:
            return True
        blob = _normalized_goal_text(_history_step_semantic_blob(step))
        if _blob_has_any(blob, ("submit", "登录", "登陆", "login", "sign in")):
            return True
    return False


def _url_looks_like_login(value: Any) -> bool:
    parsed = urlparse(str(value or ""))
    text = _normalized_goal_text(" ".join([parsed.path, parsed.fragment, parsed.query]))
    return _blob_has_any(text, ("login", "sso", "auth", "cas", "登录", "登陆"))


def _latest_click_changed_page(history: list[dict[str, Any]]) -> bool:
    latest_before = ""
    latest_after = ""
    previous_url = ""
    for item in _iter_history_items(history):
        if not isinstance(item, dict) or item.get("result") not in {None, "passed"}:
            continue
        step = item.get("step")
        url_after = str(item.get("url_after") or "")
        if isinstance(step, dict) and _step_action(step).lower() in {
            "click",
            "press",
            "press_key",
        }:
            latest_before = previous_url
            latest_after = url_after
        if url_after:
            previous_url = url_after
    return _url_state_changed(latest_before, latest_after)


def _history_step_matches_terms(step: dict[str, Any], terms: list[str]) -> bool:
    if not terms:
        return True
    normalized_terms = [
        _normalized_criterion_match_text(term)
        for term in terms
        if _normalized_criterion_match_text(term)
    ]
    return _semantic_terms_match_values(
        normalized_terms,
        _history_step_semantic_values(step),
    )


def _url_state_changed(before: str, after: str) -> bool:
    if not before or not after:
        return False
    before_parsed = urlparse(str(before))
    after_parsed = urlparse(str(after))
    if before_parsed.netloc and after_parsed.netloc and before_parsed.netloc != after_parsed.netloc:
        return True
    return (
        before_parsed.path.rstrip("/") != after_parsed.path.rstrip("/")
        or before_parsed.fragment != after_parsed.fragment
    )


def _dom_context_text_blob(dom_context: dict[str, Any]) -> str:
    if not isinstance(dom_context, dict):
        return ""
    texts: list[str] = []
    meta = dom_context.get("meta")
    if isinstance(meta, dict):
        texts.extend([meta.get("url") or "", meta.get("title") or "", meta.get("route_hint") or ""])
    page_summary = dom_context.get("page_summary")
    if isinstance(page_summary, dict):
        texts.append(page_summary.get("main_heading") or "")
        texts.extend(page_summary.get("visible_text_summary") or [])
    for section in ("forms", "interactive_elements", "assertion_candidates"):
        values = dom_context.get(section)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            texts.extend(
                str(item.get(field) or "")
                for field in (
                    "text",
                    "name",
                    "label",
                    "placeholder",
                    "near_text",
                    "role",
                    "tag",
                )
            )
    return " ".join(text for text in texts if text)


def _natural_step_decision_payload(
    step_text: str,
    *,
    spec: dict[str, Any],
    dom_context: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    text = str(step_text or "").strip()
    if not text:
        return {}
    url = _first_url(text)
    if url and _looks_like_navigation_instruction(text):
        return {
            "action": "goto",
            "value": url,
            "reason": "按自然步骤访问URL",
        }
    module_name, explicit_module = _module_name_from_entry_text(
        text,
        modules={},
    )
    if explicit_module and module_name:
        return {
            "action": "use_module",
            "module": module_name,
            "reason": "按自然步骤执行项目模块",
        }
    title_value = _title_assertion_value(text)
    if title_value:
        return {
            "action": "assert_title_contains"
            if _title_assertion_is_contains(text)
            else "assert_title",
            "value": title_value,
            "reason": "按自然步骤断言页面标题",
        }
    compound_fill = _compound_fill_decision_payload(
        text,
        spec=spec,
        dom_context=dom_context,
        history=history or [],
    )
    if compound_fill:
        return compound_fill
    fill_target, fill_value = _fill_instruction_parts(text, spec=spec)
    if fill_target and fill_value is not None:
        element_id, selector = _best_dom_target_reference(
            dom_context,
            fill_target,
            actions={"fill"},
        )
        payload = {
            "action": "fill",
            "target": fill_target,
            "value": fill_value,
            "reason": "按自然步骤输入字段",
        }
        if element_id:
            payload["element_id"] = element_id
        elif selector:
            payload["selector"] = selector
        return payload
    click_target = _click_instruction_target(text)
    if click_target:
        element_id, selector = _best_dom_target_reference(
            dom_context,
            click_target,
            actions={"click", "press"},
        )
        if _target_text_visible(dom_context, click_target):
            target = click_target
        else:
            target = click_target
        payload = {
            "action": "click",
            "target": target,
            "reason": "按自然步骤点击目标",
        }
        if element_id:
            payload["element_id"] = element_id
        elif selector:
            payload["selector"] = selector
        return payload
    action_target = _implicit_interaction_target(text)
    if action_target:
        element_id, selector = _best_dom_target_reference(
            dom_context,
            action_target,
            actions={"click", "press"},
        )
        payload = {
            "action": "click",
            "target": action_target,
            "reason": "按自然步骤执行交互目标",
        }
        if element_id:
            payload["element_id"] = element_id
        elif selector:
            payload["selector"] = selector
        return payload
    return {}


def _title_assertion_value(text: str) -> str:
    match = re.search(
        r"(?:标题|title).{0,12}(?:包含|为|是)\s*([^，。,.;；]+)",
        text,
        flags=re.I,
    )
    if match:
        return match.group(1).strip(" ：:「」\"'")
    match = re.search(
        r"(?:确认|验证).{0,10}页面标题.{0,8}([A-Za-z0-9_\-\u4e00-\u9fff]{2,})",
        text,
        flags=re.I,
    )
    if match:
        return match.group(1).strip(" ：:「」\"'")
    return ""


def _title_assertion_is_contains(text: str) -> bool:
    return "包含" in str(text or "") or re.search(
        r"\bcontains?\b",
        str(text or ""),
        flags=re.I,
    ) is not None


def _looks_like_navigation_instruction(text: str) -> bool:
    normalized = _normalized_goal_text(text)
    return bool(
        re.search(r"\b(?:go|goto|open|visit|navigate)\b", normalized)
        or any(term in str(text or "") for term in ("打开", "访问", "进入", "跳转"))
    )


def _compound_fill_decision_payload(
    text: str,
    *,
    spec: dict[str, Any],
    dom_context: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    if not re.search(r"(?:输入|填写|填入|录入)", text, flags=re.I):
        return {}
    input_values = _flatten_input_values(spec.get("inputs") or {})
    if len(input_values) < 2:
        return {}
    mentioned = _mentioned_input_values(text, input_values)
    if len(mentioned) < 2:
        return {}
    for key, value in mentioned:
        if _fill_history_has_input_value(history, key=key, value=value):
            continue
        element_id, selector = _best_dom_target_reference(
            dom_context,
            key,
            actions={"fill"},
        )
        payload: dict[str, Any] = {
            "action": "fill",
            "target": key,
            "value": str(value),
            "reason": "按复合输入步骤补齐未完成字段",
        }
        if element_id:
            payload["element_id"] = element_id
        elif selector:
            payload["selector"] = selector
        return payload
    return {}


def _mentioned_input_values(
    text: str,
    input_values: list[tuple[str, Any]],
) -> list[tuple[str, Any]]:
    normalized_text = _normalized_goal_text(text)
    result: list[tuple[str, Any]] = []
    for key, value in input_values:
        key_text = str(key or "").strip()
        if not key_text:
            continue
        key_terms = _input_key_terms(key_text)
        if not any(term and term in normalized_text for term in key_terms):
            continue
        result.append((key_text, value))
    return result


def _input_key_terms(key: str) -> list[str]:
    normalized = _normalize_identifier_words(key)
    terms = [normalized] if normalized else []
    leaf = str(key or "").split(".")[-1]
    leaf_normalized = _normalize_identifier_words(leaf)
    if leaf_normalized and leaf_normalized not in terms:
        terms.append(leaf_normalized)
    cjk_aliases = {
        "username": ["用户名", "账号", "账户"],
        "user name": ["用户名", "账号", "账户"],
        "password": ["密码"],
        "pwd": ["密码"],
    }
    for alias in cjk_aliases.get(leaf_normalized, []):
        normalized_alias = _normalized_goal_text(alias)
        if normalized_alias not in terms:
            terms.append(normalized_alias)
    for term in list(terms):
        compact = str(term or "").replace(" ", "")
        if compact and compact not in terms:
            terms.append(compact)
    return terms


def _fill_history_has_input_value(
    history: list[dict[str, Any]],
    *,
    key: str,
    value: Any,
) -> bool:
    terms = _input_key_terms(key)
    value_text = str(value or "")
    for item in _iter_history_items(history):
        if not isinstance(item, dict) or item.get("result") not in {None, "passed"}:
            continue
        step = item.get("step")
        if not isinstance(step, dict):
            continue
        if str(step.get("action") or "").lower() != "fill":
            continue
        step_values = [
            str(step.get("value") or ""),
            str(step.get("_resolved_value") or ""),
            str(step.get("_resolved_value_after") or ""),
        ]
        if value_text and value_text not in step_values:
            continue
        blob = _normalized_goal_text(
            " ".join(str(step.get(field) or "") for field in ("selector", "target"))
        )
        if any(term and term in blob for term in terms) or _fill_selector_matches_input_key(
            step.get("selector"),
            key,
        ):
            return True
    return False


def _fill_selector_matches_input_key(selector: Any, key: str) -> bool:
    selector_text = _normalized_goal_text(selector)
    if not selector_text:
        return False
    key_terms = _input_key_terms(key)
    normalized_key = _normalize_identifier_words(key).replace(" ", "")
    compact_selector = selector_text.replace(" ", "")
    if normalized_key and normalized_key in compact_selector:
        return True
    return any(term and term.replace(" ", "") in compact_selector for term in key_terms)


def _fill_instruction_parts(
    text: str,
    *,
    spec: dict[str, Any],
) -> tuple[str, str | None]:
    patterns = (
        r"\b(?:fill|type|enter|input)\s+(?P<target>[A-Za-z][A-Za-z0-9_ \-/]{0,40}?)\s+(?:with|as|to)\s+(?P<value>[^，。,.;；]+)",
        r"\b(?:fill|type|enter|input)\s+(?P<target>[A-Za-z][A-Za-z0-9_ \-/]{1,40}?)\s+(?P<value>[A-Za-z0-9_${}._:-]*\d[A-Za-z0-9_${}._:-]*)",
        r"(?:在|向)?\s*(?P<target>[^，。,.;；]{1,40}?(?:输入框|文本框|搜索框|查询框|字段))(?:中|里|内)?(?:输入|填写|填入|录入)\s*(?P<value>[^，。,.;；]+)",
        r"(?:在|向)?\s*(?P<target>[^，。,.;；]{1,40}?)(?:中|里|内)?(?:输入|填写|填入|录入)\s*(?P<value>[^，。,.;；]+)",
        r"(?:输入|填写|填入|录入)\s*(?P<target>[\u4e00-\u9fffA-Za-z_][\u4e00-\u9fffA-Za-z0-9_ \-/]{0,39}?)\s+(?P<value>[A-Za-z0-9_${}._-][A-Za-z0-9_${}._:\-/]*)",
        r"(?:输入|填写|填入|录入)\s*(?P<value>[A-Za-z0-9_${}._-]+)\s*(?:到|至|进)?\s*(?P<target>[^，。,.;；]{1,40})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        target = _clean_natural_target(match.group("target"))
        value = _clean_natural_value(match.group("value"))
        if not target:
            continue
        resolved_value = _resolve_natural_input_value(value, spec=spec, target=target)
        return target, resolved_value
    return "", None


def _click_instruction_target(text: str) -> str:
    match = re.search(
        r"(?:点击|单击|点选|按下|查看|打开)\s*(?P<target>[^，。,.;；]+)",
        text,
        flags=re.I,
    ) or re.search(
        r"\b(?:click|tap|press|select|choose|open|view|expand)\s+(?P<target>[^，。,.;；]+)",
        text,
        flags=re.I,
    )
    if not match:
        return ""
    target = _clean_natural_target(match.group("target"))
    for suffix in ("展开菜单", "菜单", "按钮", "链接", "入口", " button", " link", " menu"):
        if target.endswith(suffix) and len(target) > len(suffix):
            target = target[: -len(suffix)].strip()
            break
    return target


def _implicit_interaction_target(text: str) -> str:
    raw = str(text or "").strip()
    if not raw or _first_url(raw) or _looks_like_navigation_instruction(raw):
        return ""
    if _text_declares_module_entry(raw):
        return ""
    match = re.search(
        r"(?:使用|选择|选用|启用|展开)\s*(?P<target>[^，。,.;；]+)",
        raw,
        flags=re.I,
    ) or re.search(
        r"\b(?:use|select|choose|enable|expand)\s+(?P<target>[^，。,.;；]+)",
        raw,
        flags=re.I,
    )
    if not match:
        return ""
    return _clean_natural_target(match.group("target"))


def _clean_natural_target(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip(" ：:「」\"'")
    for prefix in ("中的", "里的", "内的", "输入中的"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text


def _clean_natural_value(value: Any) -> str:
    return " ".join(str(value or "").split()).strip(" ：:「」\"'")


def _resolve_natural_input_value(
    value: str,
    *,
    spec: dict[str, Any],
    target: str,
) -> str:
    text = str(value or "").strip()
    if text and text not in {"对应值", "指定值"}:
        return text
    inputs = spec.get("inputs") if isinstance(spec, dict) else {}
    if not isinstance(inputs, dict):
        return text
    target_key = _normalize_identifier_words(target)
    for key, candidate in _flatten_input_values(inputs):
        candidate_key = _normalize_identifier_words(key)
        if candidate_key and target_key and (
            candidate_key in target_key or target_key in candidate_key
        ):
            return str(candidate)
    return text


def _best_dom_target_reference(
    dom_context: dict[str, Any],
    target: str,
    *,
    actions: set[str],
) -> tuple[str, str]:
    terms = _intent_action_terms(target)
    if not terms:
        terms = [_normalized_criterion_match_text(target)]
    best: tuple[int, str, str] | None = None
    for section, item in _iter_dom_elements_with_section(dom_context):
        action_tag = str(item.get("tag") or "").lower()
        action_role = str(item.get("role") or "").lower()
        input_type = str(item.get("input_type") or item.get("type") or "").lower()
        if "fill" in actions and not (
            action_tag in {"input", "textarea", "select"}
            or action_role in {"textbox", "combobox", "searchbox"}
            or input_type in {"text", "search", "number", "tel", "email"}
        ):
            continue
        if actions & {"click", "press"}:
            is_semantic_click_target = (
                section == "interactive_elements"
                or
                action_tag in {"a", "button", "input", "summary", "option"}
                or action_role
                in {
                    "button",
                    "link",
                    "menuitem",
                    "option",
                    "radio",
                    "checkbox",
                    "tab",
                }
                or str(item.get("aria_label") or "").strip()
            )
            if not is_semantic_click_target:
                continue
        selector = selector_for_element_id(dom_context, str(item.get("id") or ""))
        if not selector:
            continue
        blob = _normalized_criterion_match_text(_element_primary_blob(item))
        if not blob:
            continue
        score = 0
        for term in terms:
            if term and term in blob:
                score += 40 + min(len(term), 20)
        if not score:
            continue
        if "fill" in actions:
            score += 10
        if actions & {"click", "press"} and (
            action_role in {"button", "link", "menuitem", "tab"}
            or action_tag in {"a", "button"}
        ):
            score += 10
        item_id = str(item.get("id") or "")
        if best is None or score > best[0]:
            best = (score, item_id, selector)
    if not best:
        return "", ""
    return best[1], best[2]


def _best_fill_reference_for_input_value(
    dom_context: dict[str, Any],
    *,
    key: str,
    value: str,
    history: list[dict[str, Any]],
) -> tuple[str, str]:
    terms = _input_key_terms(key)
    best: tuple[int, str, str] | None = None
    for item in _iter_dom_elements(dom_context):
        if not _dom_item_is_fillable(item):
            continue
        selector = selector_for_element_id(dom_context, str(item.get("id") or ""))
        if not selector:
            continue
        if _fill_history_has_selector_or_id(
            history,
            selector=selector,
            element_id=str(item.get("id") or ""),
            value=value,
        ):
            continue
        blob = _normalized_criterion_match_text(_element_primary_blob(item))
        score = 0
        for term in terms:
            compact_term = str(term or "").replace(" ", "")
            compact_blob = blob.replace(" ", "")
            if term and term in blob:
                score += 50
            elif compact_term and compact_term in compact_blob:
                score += 45
        if not score:
            continue
        input_type = str(item.get("input_type") or item.get("type") or "").lower()
        if _fill_target_is_password_like(key) and input_type == "password":
            score += 30
        if _fill_target_is_username_like(key) and input_type in {"text", "email"}:
            score += 15
        if best is None or score > best[0]:
            best = (score, str(item.get("id") or ""), selector)
    if not best:
        return "", ""
    return best[1], best[2]


def _dom_item_is_fillable(item: dict[str, Any]) -> bool:
    tag = str(item.get("tag") or item.get("type") or "").lower()
    role = str(item.get("role") or "").lower()
    input_type = str(item.get("input_type") or item.get("type") or "").lower()
    return (
        tag in {"input", "textarea", "select"}
        or role in {"textbox", "combobox", "searchbox"}
        or input_type in {"text", "search", "number", "tel", "email", "password"}
    )


def _fill_history_has_selector_or_id(
    history: list[dict[str, Any]],
    *,
    selector: str,
    element_id: str,
    value: str,
) -> bool:
    normalized_selector = _normalize_selector_for_compare(selector)
    for item in _iter_history_items(history):
        if not isinstance(item, dict) or item.get("result") not in {None, "passed"}:
            continue
        step = item.get("step")
        if not isinstance(step, dict) or _step_action(step).lower() != "fill":
            continue
        step_values = {
            str(step.get("value") or ""),
            str(step.get("_resolved_value_after") or ""),
        }
        if value and value not in step_values:
            continue
        step_selector = _normalize_selector_for_compare(step.get("selector"))
        step_target = str(step.get("target") or "")
        if normalized_selector and normalized_selector == step_selector:
            return True
        if element_id and step_target == element_id:
            return True
    return False


def _fill_target_is_credential_like(value: Any) -> bool:
    return _fill_target_is_username_like(value) or _fill_target_is_password_like(value)


def _fill_target_is_username_like(value: Any) -> bool:
    normalized = _normalized_goal_text(value)
    normalized_identifier = _normalize_identifier_words(value)
    return any(
        term in normalized
        for term in ("username", "user name", "账号", "账户", "用户名")
    ) or "user name" in normalized_identifier


def _fill_target_is_password_like(value: Any) -> bool:
    normalized = _normalized_goal_text(value)
    normalized_identifier = _normalize_identifier_words(value)
    return any(term in normalized for term in ("password", "pwd", "密码")) or any(
        term in normalized_identifier for term in ("password", "pwd")
    )


def _target_text_visible(dom_context: dict[str, Any], target: str) -> bool:
    if not isinstance(dom_context, dict):
        return False
    return _dom_contains_text(dom_context, target)


def _first_entry_module_name(
    *,
    case_data: dict[str, Any],
    intent: str,
    steps: list[str],
    modules: dict[str, Any],
) -> tuple[str | None, bool]:
    first_step = _first_raw_step(case_data.get("steps"))
    if first_step is None:
        first_step = _first_raw_step(steps)
    if first_step is not None:
        return _module_name_from_entry_step(first_step, modules=modules)
    return _module_name_from_entry_text(_first_instruction_clause(intent), modules=modules)


def _has_explicit_module_entry(
    *,
    case_data: dict[str, Any],
    intent: str,
    steps: list[str],
    modules: dict[str, Any],
) -> bool:
    _, explicit_reference = _first_entry_module_name(
        case_data=case_data,
        intent=intent,
        steps=steps,
        modules=modules,
    )
    return bool(explicit_reference)


def _first_raw_step(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    if not value and isinstance(value, list):
        return None
    return None


def _module_name_from_entry_step(
    value: Any,
    *,
    modules: dict[str, Any],
) -> tuple[str | None, bool]:
    if isinstance(value, dict):
        module_name = _module_name_from_structured_step(value)
        explicit = _structured_step_is_use_module(value)
        if module_name:
            return module_name, True
        return None, explicit
    if isinstance(value, str):
        return _module_name_from_entry_text(value, modules=modules)
    return None, False


def _module_name_from_structured_step(step: dict[str, Any]) -> str:
    if step.get("use_module") is not None:
        value = step.get("use_module")
        if isinstance(value, dict):
            return str(value.get("name") or value.get("module") or "").strip()
        return str(value or "").strip()
    action = str(step.get("action") or "").strip().lower()
    if action == "use_module":
        return str(
            step.get("module")
            or step.get("target")
            or step.get("name")
            or ""
        ).strip()
    return ""


def _structured_step_is_use_module(step: dict[str, Any]) -> bool:
    return step.get("use_module") is not None or str(
        step.get("action") or ""
    ).strip().lower() == "use_module"


def _module_name_from_entry_text(
    value: Any,
    *,
    modules: dict[str, Any],
) -> tuple[str | None, bool]:
    text = str(value or "").strip()
    if not text:
        return None, False
    declares_module = _text_declares_module_entry(text)
    explicit_name = _explicit_module_token(text)
    if not declares_module and not explicit_name:
        return None, False

    normalized_text = _normalized_goal_text(text)
    matches = [
        name
        for name in sorted((modules or {}).keys())
        if name in text or _normalized_goal_text(name) in normalized_text
    ]
    if len(matches) == 1:
        return matches[0], True
    if len(matches) > 1:
        return None, True

    if explicit_name:
        return explicit_name, True
    return None, True


def _text_declares_module_entry(text: str) -> bool:
    return bool(
        re.search(r"\buse[_\s-]*module\b", text, flags=re.I)
        or re.search(r"\b(?:use|run|execute|call)\s+project\s+module\b", text, flags=re.I)
        or re.search(r"(?:使用|执行|调用).{0,12}模块", text)
        or "项目模块" in text
    )


def _first_instruction_clause(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    parts = re.split(r"[，。,.;；\n]", raw, maxsplit=1)
    return parts[0].strip()


def _explicit_module_token(text: str) -> str:
    patterns = (
        r"\buse[_\s-]*module\s+([A-Za-z0-9_.-]+)",
        r"项目模块\s*([A-Za-z0-9_.-]+)",
        r"模块\s*([A-Za-z0-9_.-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return match.group(1).strip()
    return ""


def _format_step(step: dict[str, Any]) -> str:
    action = step.get("action") or ("use_module" if step.get("use_module") else "")
    parts = [f"action={action}"]
    for key in ("use_module", "selector", "target", "value", "key"):
        if step.get(key) is not None:
            parts.append(f"{key}={step.get(key)}")
    expected = _expected_from_step(step)
    if expected:
        parts.append(f"expected={expected}")
    return " | ".join(parts)


def _step_action(step: dict[str, Any] | None) -> str:
    if not isinstance(step, dict):
        return ""
    return str(step.get("action") or ("use_module" if step.get("use_module") else ""))


def _is_assertion_step(step: dict[str, Any] | None) -> bool:
    return _step_action(step).lower().startswith("assert_")


def _completion_wait_seconds_for_step(
    step: dict[str, Any] | None, *, default_seconds: float
) -> float:
    action = _step_action(step).lower()
    if action in {"goto", "click", "press", "press_key", "use_module"}:
        return max(float(default_seconds or 0), 0)
    return 0.0


def _is_title_metadata_selector(selector: Any) -> bool:
    normalized = re.sub(r"\s+", " ", str(selector or "").strip().lower())
    return normalized in {"title", "head title", "head > title"}


def _page_context_pages(page: Any) -> list[Any]:
    context = getattr(page, "context", None)
    if context is None:
        return []
    try:
        return list(getattr(context, "pages", []) or [])
    except Exception:
        return []


def _click_may_open_new_page(page: Any, selector: Any) -> bool:
    if not selector or not hasattr(page, "locator"):
        return False
    try:
        locator = page.locator(str(selector)).first
        if callable(locator):
            locator = locator()
        return bool(
            locator.evaluate(
                """(el) => {
                    const link = el.closest && el.closest('a[target="_blank"]');
                    if (link) return true;
                    const form = el.closest && el.closest('form[target="_blank"]');
                    return Boolean(form);
                }"""
            )
        )
    except Exception:
        return False


def _new_page_navigation_hint(page: Any, selector: Any) -> str | None:
    if not selector or not hasattr(page, "locator"):
        return None
    try:
        locator = page.locator(str(selector)).first
        if callable(locator):
            locator = locator()
        url = locator.evaluate(
            """(el) => {
                const link = el.closest && el.closest('a[target="_blank"]');
                if (link && link.href) return link.href;
                const form = el.closest && el.closest('form[target="_blank"]');
                if (!form) return null;
                const action = form.action || window.location.href;
                const method = String(form.method || 'get').toLowerCase();
                if (method !== 'get') return null;
                const target = new URL(action, window.location.href);
                const data = new FormData(form);
                for (const [key, value] of data.entries()) {
                    if (typeof value === 'string') target.searchParams.set(key, value);
                }
                return String(target);
            }"""
        )
        text = str(url or "").strip()
        return text or None
    except Exception:
        return None


def _expected_from_step(step: dict[str, Any]) -> str:
    action = str(step.get("action") or "").lower()
    if action in {"assert_text", "assert_text_contains"}:
        return str(step.get("value") or "")
    if action == "assert_visible":
        return "visible"
    if action == "assert_url_contains":
        return f"url contains {step.get('value')}"
    if action == "assert_title":
        return f"title is {step.get('value')}"
    if action == "assert_title_contains":
        return f"title contains {step.get('value')}"
    return ""


def _safe_page_title(page: Any) -> str:
    try:
        title = page.title()
        return str(title).strip()
    except Exception:
        return ""


def _complete_agent_decision(
    provider: Any,
    messages: list[dict[str, str]],
    *,
    spec: dict[str, Any],
    dom_context: dict[str, Any],
    schema_name: str,
    usage_operation: str,
    usage_metadata: dict[str, Any],
) -> AgentCaseDecision:
    if not callable(getattr(provider, "complete", None)):
        return provider.complete_model(
            messages,
            AgentCaseDecision,
            schema_name=schema_name,
            usage_operation=usage_operation,
            usage_metadata=usage_metadata,
        )
    content = provider.complete(
        messages,
        response_json=True,
        response_model=AgentCaseRuntimeDecision,
        schema_name="AgentCaseRuntimeDecision",
        usage_operation=usage_operation,
        usage_metadata=usage_metadata,
    )
    try:
        return _parse_agent_decision_response(
            content,
            spec=spec,
            dom_context=dom_context,
        )
    except Exception as exc:
        get_token_usage_tracker().record_model_io(
            operation=f"{usage_operation}.parse_error",
            request_payload={"schema_name": schema_name},
            response_payload={"content": content},
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def _parse_agent_decision_response(
    content: str,
    *,
    spec: dict[str, Any],
    dom_context: dict[str, Any],
) -> AgentCaseDecision:
    data = parse_json_object(content)
    normalized = _normalize_agent_decision_payload(
        data,
        spec=spec,
        dom_context=dom_context,
    )
    try:
        return AgentCaseDecision.model_validate(normalized)
    except Exception as exc:
        raise ValueError(f"模型响应不符合契约: {exc}") from exc


def _normalize_agent_decision_payload(
    data: dict[str, Any],
    *,
    spec: dict[str, Any],
    dom_context: dict[str, Any],
) -> dict[str, Any]:
    if str(data.get("action") or "").strip().lower() == "need_more_context":
        data = {**data, "status": "need_more_context"}
        data.pop("action", None)
        logger.warning("Agent模型将need_more_context误写为action，已归一化为status")
    if "action" not in data:
        if data.get("use_module") is not None:
            if spec.get("module_entry_allowed"):
                nested = data.get("use_module")
                if isinstance(nested, dict):
                    data = {
                        **nested,
                        "action": "use_module",
                        "module": nested.get("module") or nested.get("name"),
                    }
                else:
                    data = {**data, "action": "use_module", "module": nested}
                logger.warning("Agent模型将use_module误写为外层字段，已归一化")
            else:
                data = {k: v for k, v in data.items() if k != "use_module"}
        for action_name in _AGENT_RUNTIME_ACTIONS:
            if action_name not in data:
                continue
            nested = data.get(action_name)
            if isinstance(nested, dict):
                data = {**nested, "action": action_name}
                logger.warning("Agent模型将动作名误写为外层对象，已归一化")
                break
            if isinstance(nested, str):
                normalized_nested: dict[str, Any] = {"action": action_name}
                if action_name in {
                    "goto",
                    "assert_url_contains",
                    "assert_title",
                    "assert_title_contains",
                }:
                    normalized_nested["value"] = nested
                elif action_name in {"click", "assert_visible", "assert_text"}:
                    normalized_nested["target"] = nested
                else:
                    continue
                data = normalized_nested
                logger.warning("Agent模型将动作名误写为外层字符串，已归一化")
                break
    if "action" not in data and isinstance(data.get("decision"), str):
        decision_action = str(data.get("decision") or "").strip()
        if decision_action in _AGENT_RUNTIME_ACTIONS or (
            decision_action == "use_module" and spec.get("module_entry_allowed")
        ):
            data = {**data, "action": decision_action}
            logger.warning("Agent模型将action误写为decision，已按合法动作字段归一化")
    if not data.get("selector") and isinstance(data.get("selectors"), list):
        selector = next(
            (
                str(item).strip()
                for item in data.get("selectors") or []
                if str(item or "").strip()
            ),
            "",
        )
        if selector:
            data = {**data, "selector": selector}
            logger.warning("Agent模型将selector误写为selectors列表，已取首个有效selector")
    allowed_fields = (
        _AGENT_DECISION_FIELDS
        if spec.get("module_entry_allowed")
        else _AGENT_RUNTIME_DECISION_FIELDS
    )
    normalized = {key: value for key, value in data.items() if key in allowed_fields}
    extra_fields = sorted(set(data) - allowed_fields)
    if extra_fields:
        logger.warning(
            "Agent模型响应包含未声明字段，已丢弃: " f"{', '.join(extra_fields)}"
        )
    for text_field in ("reason", "expected"):
        if (
            isinstance(normalized.get(text_field), str)
            and len(normalized[text_field]) > 120
        ):
            normalized[text_field] = normalized[text_field][:117] + "..."
            logger.warning(f"Agent模型{text_field}超过120字符，已截断")
    if (
        normalized.get("action") == "assert_text"
        and normalized.get("value") is None
        and data.get("text") is not None
    ):
        normalized["value"] = data.get("text")
        logger.warning("Agent模型assert_text将value误写为text，已归一化")
    if (
        normalized.get("action") == "assert_text"
        and normalized.get("value") is None
        and normalized.get("target") is not None
    ):
        normalized["value"] = normalized.get("target")
        logger.warning("Agent模型assert_text缺少value，已使用target作为断言文本")
    if (
        normalized.get("action") == "assert_text"
        and normalized.get("value") is not None
        and not (
            normalized.get("element_id")
            or normalized.get("selector")
            or normalized.get("target")
        )
    ):
        normalized["target"] = normalized.get("value")
        logger.warning("Agent模型assert_text缺少定位目标，已使用value作为可见文本目标")
    if normalized.get("action") == "wait" and normalized.get("wait_ms") is None:
        wait_value = normalized.get("value")
        if isinstance(wait_value, (int, float)) and wait_value >= 0:
            normalized["wait_ms"] = int(wait_value)
            normalized.pop("value", None)
            logger.warning("Agent模型wait将wait_ms误写为数值value，已归一化")
        elif isinstance(wait_value, str) and wait_value.strip().isdigit():
            normalized["wait_ms"] = int(wait_value.strip())
            normalized.pop("value", None)
            logger.warning("Agent模型wait将wait_ms误写为字符串value，已归一化")
    if normalized.get("action") == "wait" and normalized.get("wait_ms") is None:
        normalized["wait_ms"] = 1000
        logger.warning("Agent模型wait动作缺少wait_ms，已使用默认1000ms")
    if normalized.get("action") == "press" and not normalized.get("key"):
        key_value = normalized.get("value")
        if isinstance(key_value, str) and key_value.strip():
            normalized["key"] = key_value.strip()
            normalized.pop("value", None)
            logger.warning("Agent模型press将key误写为value，已归一化")
    if normalized.get("action") == "press" and not (
        normalized.get("element_id")
        or normalized.get("selector")
        or normalized.get("target")
    ):
        raise ValueError("press action requires element_id, selector or target")
    if normalized.get("action") == "fill" and normalized.get("value") is None:
        inferred_value = _infer_fill_value_from_inputs(
            normalized,
            spec=spec,
            dom_context=dom_context,
        )
        if inferred_value is not None:
            normalized["value"] = inferred_value
            logger.warning("Agent模型fill动作缺少value，已根据DOM字段和inputs唯一补全")
    return normalized


def _selector_for_equivalent_dom_text(
    *,
    dom_context: dict[str, Any],
    current_selector: str | None,
) -> str | None:
    selector_text = _text_query_from_selector(current_selector)
    if not selector_text:
        return None
    selector_key = _compact_text_key(selector_text)
    if not selector_key:
        return None
    tag_hint = _selector_tag_hint(current_selector)
    for item in _iter_dom_elements(dom_context):
        if tag_hint and not _dom_item_matches_tag_hint(item, tag_hint):
            continue
        item_texts = (
            item.get("text"),
            item.get("name"),
            item.get("label"),
            item.get("placeholder"),
        )
        if not any(_compact_text_key(value) == selector_key for value in item_texts):
            continue
        selector = selector_for_element_id(dom_context, str(item.get("id") or ""))
        if selector:
            return selector
    return None


def _text_query_from_selector(selector: str | None) -> str:
    text = str(selector or "").strip()
    if not text:
        return ""
    match = re.search(r":has-text\(\s*(['\"])(.*?)\1\s*\)", text, flags=re.I)
    if match:
        return match.group(2).strip()
    match = re.search(r"\btext\s*=\s*(['\"]?)([^'\"]+?)\1\s*$", text, flags=re.I)
    if match:
        return match.group(2).strip()
    if not re.search(
        r"(?:text\s*\(|normalize-space\s*\(|contains\s*\(|translate\s*\()",
        text,
        flags=re.I,
    ):
        return ""
    for pattern in (
        r"contains\s*\(\s*(?:text\s*\(\)|normalize-space\s*\([^)]*\)|\.)\s*,\s*(['\"])(.*?)\1\s*\)",
        r"(?:text\s*\(\)|normalize-space\s*\([^)]*\)|translate\s*\([^)]*\))\s*=\s*(['\"])(.*?)\1",
    ):
        match = re.search(pattern, text, flags=re.I)
        if match and _selector_text_literal_looks_meaningful(match.group(2)):
            return match.group(2).strip()
    literals = [
        match.group(2).strip()
        for match in re.finditer(r"(['\"])(.*?)\1", text)
        if _selector_text_literal_looks_meaningful(match.group(2))
    ]
    if literals:
        return literals[-1]
    return ""


def _selector_text_literal_looks_meaningful(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text):
        return False
    return True


def _dom_target_text_for_step(
    dom_context: dict[str, Any],
    step: dict[str, Any],
    *,
    element_id: str | None = None,
) -> str:
    item = _dom_element_for_step(dom_context, step, element_id=element_id)
    if not item:
        return ""
    return _element_display_text(item)


def _dom_element_for_step(
    dom_context: dict[str, Any],
    step: dict[str, Any],
    *,
    element_id: str | None = None,
) -> dict[str, Any] | None:
    ids: list[str] = []
    for value in (element_id, step.get("element_id"), step.get("target")):
        text = str(value or "").strip()
        if looks_like_internal_element_id(text):
            ids.append(text)
    selector = str(step.get("selector") or "").strip()
    for item in _iter_dom_elements(dom_context):
        item_id = str(item.get("id") or "").strip()
        if item_id and item_id in ids:
            return item
        if selector and _selector_matches_element(
            selector,
            selectors_for_element_id(dom_context, item_id),
        ):
            return item
    return None


def _element_display_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "text",
        "name",
        "label",
        "placeholder",
        "aria_label",
        "title",
    ):
        value = str(item.get(key) or "").strip()
        if value:
            parts.append(value)
    attributes = item.get("attributes")
    if isinstance(attributes, dict):
        for key in ("aria-label", "title"):
            value = str(attributes.get(key) or "").strip()
            if value:
                parts.append(value)
    return " ".join(_dedupe_texts(parts))


def _selector_tag_hint(selector: str | None) -> str:
    match = re.match(r"\s*([a-zA-Z][\w-]*)", str(selector or ""))
    return match.group(1).lower() if match else ""


def _dom_item_matches_tag_hint(item: dict[str, Any], tag_hint: str) -> bool:
    tag = str(item.get("tag") or "").lower()
    role = str(item.get("role") or "").lower()
    if tag_hint in {tag, role}:
        return True
    selectors = item.get("selector_candidates")
    if isinstance(selectors, list):
        return any(str(selector).lower().startswith(tag_hint) for selector in selectors)
    return False


def _compact_text_key(value: Any) -> str:
    return _normalized_goal_text(value).replace(" ", "")


def _flatten_agent_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        result: list[str] = []
        for key, item in value.items():
            result.append(str(key))
            result.extend(_flatten_agent_text(item))
        return result
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_flatten_agent_text(item))
        return result
    return [str(value)]


def _normalized_goal_text(value: Any) -> str:
    text = str(value or "").lower()
    return " ".join(re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text).split())


def _assert_text_should_use_contains(*, criteria: Any, expected: Any) -> bool:
    expected_text = str(expected or "").strip()
    if not expected_text or not isinstance(criteria, dict):
        return False
    expected_l = expected_text.lower()
    for criterion in _flatten_agent_text(criteria):
        criterion_text = str(criterion or "")
        criterion_l = criterion_text.lower()
        if expected_l not in criterion_l:
            continue
        if any(term in criterion_l for term in ("包含", "contains", "contain")):
            return True
    return False


def _blob_has_any(blob: str, terms: tuple[str, ...]) -> bool:
    normalized = _normalized_goal_text(blob)
    return any(_normalized_goal_text(term) in normalized for term in terms)


def _element_primary_blob(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "id",
        "tag",
        "role",
        "name",
        "text",
        "label",
        "placeholder",
        "type",
    ):
        parts.append(str(item.get(key) or ""))
    attributes = item.get("attributes")
    if isinstance(attributes, dict):
        parts.extend(str(value or "") for value in attributes.values())
    selectors = item.get("selector_candidates")
    if isinstance(selectors, list):
        parts.extend(str(selector or "") for selector in selectors)
    return " ".join(part for part in parts if part)


def _risky_click_mismatch(*, current_blob: str, intent_blob: str) -> bool:
    if not current_blob:
        return False
    destructive_terms = (
        "remove",
        "delete",
        "logout",
        "log out",
        "sign out",
        "exit",
        "删除",
        "移除",
        "退出",
        "登出",
        "注销",
    )
    menu_terms = ("open menu", "menu trigger", "hamburger", "菜单", "展开菜单")
    if any(term in current_blob for term in destructive_terms):
        return not any(term in intent_blob for term in destructive_terms)
    if any(term in current_blob for term in menu_terms):
        return not any(term in intent_blob for term in menu_terms)
    return False


def _repair_click_target_from_spec(
    spec: dict[str, Any], *, current_blob: str = ""
) -> str:
    texts = [
        str(spec.get("intent") or ""),
        " ".join(str(item) for item in (spec.get("steps") or [])),
    ]
    criteria = spec.get("criteria")
    if isinstance(criteria, dict):
        texts.extend(str(item) for item in (criteria.get("final") or []))
        texts.extend(str(item) for item in (criteria.get("checkpoints") or []))
    blob = _normalized_criterion_match_text(" ".join(texts))
    candidates: list[str] = []
    if "review" in blob or "order" in blob:
        candidates.extend(["order review", "review order"])
    if any(term in current_blob for term in ("logout", "log out", "sign out")):
        candidates.extend(["close menu", "close"])
    if "close" in blob or "menu" in current_blob:
        candidates.extend(["close menu", "close"])
    if "complete" in blob or "submission" in blob:
        candidates.extend(["submit", "complete"])
    for text in texts:
        click_target = _click_instruction_target(text) or _implicit_interaction_target(text)
        if click_target:
            candidates.append(click_target)
    return next((candidate for candidate in candidates if candidate), "")


def _selector_is_visible_enabled(page: Any, selector: str | None) -> bool:
    if not selector or not hasattr(page, "locator"):
        return False
    try:
        locator = page.locator(selector)
        count = int(locator.count())
        if count < 1:
            return False
        first = locator.first
        if callable(first):
            first = first()
        is_visible = getattr(first, "is_visible", None)
        if callable(is_visible):
            try:
                if not bool(is_visible(timeout=500)):
                    return False
            except TypeError:
                if not bool(is_visible()):
                    return False
        is_enabled = getattr(first, "is_enabled", None)
        if callable(is_enabled):
            try:
                return bool(is_enabled(timeout=500))
            except TypeError:
                return bool(is_enabled())
        return True
    except Exception:
        return False


def _infer_fill_value_from_inputs(
    decision_payload: dict[str, Any],
    *,
    spec: dict[str, Any],
    dom_context: dict[str, Any],
) -> Any:
    element = _dom_element_for_decision_payload(decision_payload, dom_context)
    field_text = _field_descriptor_for_fill(element, decision_payload)
    if not field_text:
        return None
    best_score = 0
    best_value: Any = None
    tied = False
    for path, value in _flatten_input_values(spec.get("inputs") or {}):
        score = _score_input_match(field_text, path)
        if score <= 0:
            continue
        if score > best_score:
            best_score = score
            best_value = value
            tied = False
        elif score == best_score:
            tied = True
    if best_score < 35 or tied:
        return None
    return best_value


def _dom_element_for_decision_payload(
    decision_payload: dict[str, Any],
    dom_context: dict[str, Any],
) -> dict[str, Any] | None:
    references = [
        str(decision_payload.get(key) or "").strip()
        for key in ("element_id", "target", "selector")
    ]
    internal_ids = {ref for ref in references if looks_like_internal_element_id(ref)}
    for item in _iter_dom_elements(dom_context):
        if str(item.get("id") or "") in internal_ids:
            return item
    explicit_selector = str(decision_payload.get("selector") or "").strip()
    if not explicit_selector or looks_like_internal_element_id(explicit_selector):
        return None
    for item in _iter_dom_elements(dom_context):
        selectors = selectors_for_element_id(dom_context, str(item.get("id") or ""))
        if _selector_matches_element(explicit_selector, selectors):
            return item
    return None


def _field_descriptor_for_fill(
    element: dict[str, Any] | None,
    decision_payload: dict[str, Any],
) -> str:
    parts: list[str] = []
    if isinstance(element, dict):
        for key in ("id", "name", "label", "placeholder", "text", "near_text"):
            value = element.get(key)
            if value:
                parts.append(str(value))
        attributes = element.get("attributes")
        if isinstance(attributes, dict):
            for key in ("data-test", "aria-label", "name", "id", "type"):
                value = attributes.get(key)
                if value:
                    parts.append(str(value))
    for key in ("element_id", "selector", "target"):
        value = decision_payload.get(key)
        if value:
            parts.append(str(value))
    return " ".join(parts)


def _flatten_input_values(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        result: list[tuple[str, Any]] = []
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_flatten_input_values(item, path))
        return result
    if isinstance(value, list):
        return []
    return [(prefix, value)]


def _score_input_match(field_text: str, input_path: str) -> int:
    field_words = _normalize_identifier_words(field_text)
    path_words = _normalize_identifier_words(input_path)
    leaf_words = _normalize_identifier_words(input_path.split(".")[-1])
    if not field_words or not leaf_words:
        return 0
    field_tokens = set(field_words.split())
    leaf_tokens = set(leaf_words.split())
    common = field_tokens & leaf_tokens
    score = len(common) * 10
    if leaf_words in field_words:
        score += 60
    if leaf_tokens and leaf_tokens <= field_tokens:
        score += 30
    if path_words and path_words in field_words:
        score += 40
    if "zip" in field_tokens and "postal" in leaf_tokens:
        score += 10
    if "postal" in field_tokens and "zip" in leaf_tokens:
        score += 10
    if score < 60 and not common:
        return 0
    return score


def _normalize_identifier_words(value: Any) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
    text = re.sub(r"[^0-9A-Za-z]+", " ", text).lower()
    replacements = {
        "username": "user name",
        "firstname": "first name",
        "lastname": "last name",
        "zipcode": "zip code",
        "postalcode": "postal code",
    }
    for source, replacement in replacements.items():
        text = re.sub(rf"\b{source}\b", replacement, text)
    return " ".join(text.split())


def _selector_matches_element(selector: str | None, selectors: list[str]) -> bool:
    normalized = _normalize_selector_for_compare(selector)
    if not normalized:
        return False
    return any(
        _normalize_selector_for_compare(item) == normalized for item in selectors
    )


def _normalize_selector_for_compare(selector: Any) -> str:
    text = str(selector or "").strip()
    if not text:
        return ""
    return re.sub(r"\s+", "", text).replace("'", '"')


def _selector_for_misrouted_page_title_assertion(
    *,
    page: Any,
    dom_context: dict[str, Any],
    elements: dict[str, Any],
    expected_text: Any,
) -> str | None:
    expected = " ".join(str(expected_text or "").split()).strip()
    if not expected or "${" in expected:
        return None
    if _safe_page_title(page) == expected:
        return None
    selector = _selector_for_assertion_text(
        page=page,
        dom_context=dom_context,
        elements=elements,
        expected_text=expected,
        current_selector=None,
    )
    if selector:
        logger.warning(
            "Agent将页面可见标题误判为浏览器title，已改用assert_text: "
            f"value={expected} | selector={selector}"
        )
    return selector


def _selector_for_assertion_text(
    *,
    page: Any,
    dom_context: dict[str, Any],
    elements: dict[str, Any],
    expected_text: Any,
    current_selector: str | None,
) -> str | None:
    expected = " ".join(str(expected_text or "").split()).strip()
    if not expected or "${" in expected:
        return current_selector
    if _selector_has_exact_text(
        page,
        _resolve_project_selector(elements, current_selector),
        expected,
    ):
        return current_selector
    dom_selector = _selector_for_visible_text(dom_context, expected)
    if _selector_has_exact_text(page, dom_selector, expected):
        return dom_selector
    for selector in _project_assertion_text_selectors(elements):
        if _selector_has_exact_text(page, selector, expected):
            return selector
    for selector in _COMMON_ASSERTION_TEXT_SELECTORS:
        if _selector_has_exact_text(page, selector, expected):
            return selector
    return current_selector if current_selector else dom_selector


def _project_assertion_text_selectors(elements: dict[str, Any]) -> list[str]:
    result: list[str] = []
    if not isinstance(elements, dict):
        return result
    for key, value in elements.items():
        key_text = str(key or "").lower()
        if not any(term in key_text for term in _ASSERTION_TEXT_ELEMENT_KEYWORDS):
            continue
        if isinstance(value, str) and value.strip():
            result.append(value.strip())
    return _dedupe_strings(result)


def _resolve_project_selector(
    elements: dict[str, Any],
    selector: str | None,
) -> str | None:
    if not selector:
        return None
    if isinstance(elements, dict) and selector in elements:
        value = elements.get(selector)
        return str(value).strip() if value else selector
    return selector


def _selector_has_exact_text(
    page: Any, selector: str | None, expected_text: str
) -> bool:
    if not selector or not hasattr(page, "locator"):
        return False
    try:
        locator = page.locator(selector)
        count = int(locator.count())
        if count < 1:
            return False
        first = locator.first
        actual = _locator_text(first)
    except Exception:
        return False
    return _normalize_assertion_text(actual) == _normalize_assertion_text(expected_text)


def _locator_text(locator: Any) -> str:
    for method_name in ("inner_text", "text_content", "input_value"):
        method = getattr(locator, method_name, None)
        if not callable(method):
            continue
        try:
            value = method(timeout=500)
        except TypeError:
            try:
                value = method()
            except Exception:
                continue
        except Exception:
            continue
        if value is not None:
            return str(value)
    return ""


def _normalize_assertion_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _candidate_text_summary(
    candidates: list[dict[str, Any]], *, limit: int = 8
) -> list[str]:
    texts: list[str] = []
    for candidate in candidates:
        text = str(candidate.get("text") or candidate.get("value") or "").strip()
        if not text:
            continue
        text = " ".join(text.split())
        if text and text not in texts:
            texts.append(text[:80])
        if len(texts) >= limit:
            break
    return texts


def _iter_dom_elements(dom_context: dict[str, Any]):
    for _, item in _iter_dom_elements_with_section(dom_context):
        yield item


def _iter_dom_elements_with_section(dom_context: dict[str, Any]):
    if not isinstance(dom_context, dict):
        return
    for section in ("forms", "interactive_elements", "assertion_candidates"):
        values = dom_context.get(section)
        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict):
                    yield section, item


def _selector_for_visible_text(
    dom_context: dict[str, Any],
    expected_text: Any,
) -> str | None:
    token = " ".join(str(expected_text or "").split()).strip()
    if not token or "${" in token:
        return None
    token_l = token.lower()
    best: tuple[int, str] | None = None
    for item in _iter_dom_elements(dom_context):
        selector = selector_for_element_id(dom_context, str(item.get("id") or ""))
        if not selector:
            continue
        fields = (
            item.get("text"),
            item.get("name"),
            item.get("near_text"),
            item.get("label"),
            item.get("placeholder"),
        )
        blob = " ".join(str(value or "") for value in fields).lower()
        if token_l not in blob:
            continue
        score = 10
        if token_l == str(item.get("text") or "").strip().lower():
            score += 20
        if item.get("id", "").startswith("a"):
            score += 5
        if best is None or score > best[0]:
            best = (score, selector)
    return best[1] if best else None


def _history_has(
    history: list[dict[str, Any]],
    *,
    action: str | None = None,
    selector_contains: str | None = None,
    value_contains: str | None = None,
) -> bool:
    for item in _iter_history_items(history):
        if not isinstance(item, dict):
            continue
        step = item.get("step")
        if not isinstance(step, dict):
            continue
        if action and str(step.get("action") or "").lower() != action.lower():
            continue
        selector_blob = str(step.get("selector") or step.get("target") or "").lower()
        value_blob = str(step.get("value") or "").lower()
        if selector_contains and selector_contains.lower() not in selector_blob:
            continue
        if value_contains and value_contains.lower() not in value_blob:
            continue
        return True
    return False


def _remaining_step_hints(
    *,
    spec: dict[str, Any],
    history: list[dict[str, Any]],
    limit: int = 20,
) -> list[str]:
    plan = _agent_natural_plan(spec)
    if not history:
        return plan[: max(0, limit)]
    start = 0
    for index, step_text in enumerate(plan):
        if not _natural_step_satisfied(
            step_text,
            spec=spec,
            history=history,
            dom_context={},
        ):
            start = index
            break
    else:
        start = len(plan)
    return plan[start : start + max(0, limit)]


def _current_agent_goal(
    *,
    spec: dict[str, Any],
    history: list[dict[str, Any]],
    runtime_harness: dict[str, Any] | None = None,
) -> str:
    if runtime_harness:
        phase = str(runtime_harness.get("phase") or "").strip()
        if phase:
            return f"根据实时DOM推进当前阶段: {phase}"
    if history:
        return "根据实时DOM和unmet_requirements推进，不重复history中已成功动作"
    return str(spec.get("intent") or "") or "根据自然语言步骤和验收标准完成用例"


def _unmet_final_criteria(
    *,
    criteria: Any,
    history: list[dict[str, Any]],
    current_url: str,
    dom_context: dict[str, Any],
) -> list[str]:
    if not isinstance(criteria, dict):
        return []
    finals = _normalize_text_list(criteria.get("final"))
    return [
        item
        for item in finals
        if not _final_criterion_satisfied(
            item,
            history=history,
            current_url=current_url,
            dom_context=dom_context,
        )
    ]


def _unmet_completion_criteria(
    *,
    criteria: Any,
    history: list[dict[str, Any]],
    current_url: str,
    dom_context: dict[str, Any],
) -> list[str]:
    return [
        item
        for item in _completion_criteria_items(criteria)
        if not _final_criterion_satisfied(
            item,
            history=history,
            current_url=current_url,
            dom_context=dom_context,
        )
    ]


def _unmet_intent_action_requirements(
    *,
    spec: dict[str, Any],
    history: list[dict[str, Any]],
    current_url: str = "",
    dom_context: dict[str, Any] | None = None,
) -> list[str]:
    requirements = _intent_action_requirements(spec)
    unmet: list[str] = []
    for action, terms, label, source_text in requirements:
        actions = {"fill"} if action == "fill" else {"click", "press", "press_key"}
        if action == "click" and _auth_submit_history_satisfies_target(
            history,
            target=source_text,
        ):
            continue
        if not _history_action_satisfies(
            history,
            actions=actions,
            terms=terms,
            require_observed_progress=False,
        ):
            unmet.append(label)
    return unmet


def _final_criterion_satisfied(
    criterion: str,
    *,
    history: list[dict[str, Any]],
    current_url: str,
    dom_context: dict[str, Any],
) -> bool:
    text = str(criterion or "").lower()
    if not text:
        return True
    terms = _criterion_evidence_terms(criterion)
    if _criterion_requires_fill_history(text):
        if _history_action_satisfies(
            history,
            actions={"fill"},
            terms=terms,
        ):
            return True
        return _history_action_satisfies(
            history,
            actions={"assert_value", "assert_have_values"},
            terms=terms,
        )
    if _criterion_requires_click_history(text):
        return _history_action_satisfies(
            history,
            actions={"click", "press", "press_key"},
            terms=terms,
        )
    if _criterion_requires_title(text):
        return any(_page_title_contains(dom_context, term) for term in terms)
    if _criterion_requires_url(text):
        return any(term.lower() in str(current_url or "").lower() for term in terms)
    for term in terms:
        if (
            _dom_contains_text(dom_context, term)
            or term.lower() in str(current_url or "").lower()
        ):
            return True
        if _history_assertion_has(history, value_contains=term):
            return True
    return False


def _criterion_requires_title(text: str) -> bool:
    return "标题" in text or "title" in text


def _criterion_requires_url(text: str) -> bool:
    return "url" in text or "地址" in text


def _criterion_requires_fill_history(text: str) -> bool:
    return any(
        term in text
        for term in ("已输入", "已填写", "已填入", "输入框已", "填写", "填入")
    )


def _criterion_requires_click_history(text: str) -> bool:
    return any(
        term in text
        for term in (
            "已执行",
            "已点击",
            "点击",
            "单击",
            "点选",
            "按下",
            "clicked",
            "pressed",
            "executed",
        )
    )


def _page_title_contains(dom_context: dict[str, Any], needle: str) -> bool:
    meta = dom_context.get("meta") if isinstance(dom_context, dict) else {}
    title = (
        str((meta or {}).get("title") or "").lower() if isinstance(meta, dict) else ""
    )
    return bool(needle) and str(needle).lower() in title


def _page_observation_signature(page: Any) -> str:
    if page is None or not hasattr(page, "locator"):
        return ""
    try:
        body = page.locator("body").first
        text = _locator_text(body)
    except Exception:
        text = ""
    try:
        title = page.title()
    except Exception:
        title = ""
    try:
        url = getattr(page, "url", "")
    except Exception:
        url = ""
    normalized = _normalize_assertion_text(text)[:4000]
    raw = json.dumps(
        {"url": str(url or ""), "title": str(title or ""), "text": normalized},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _page_contains_text(page: Any, text: Any) -> bool:
    token = _normalize_assertion_text(text)
    if not token or page is None:
        return False
    if hasattr(page, "get_by_text"):
        try:
            locator = page.get_by_text(token, exact=True)
            count = int(locator.count())
            for index in range(min(count, 20)):
                try:
                    if locator.nth(index).is_visible():
                        return True
                except Exception:
                    continue
        except Exception:
            pass
    if hasattr(page, "locator"):
        try:
            body_text = _locator_text(page.locator("body").first)
            return token in _normalize_assertion_text(body_text)
        except Exception:
            return False
    return False


def _history_assertion_has(
    history: list[dict[str, Any]],
    *,
    value_contains: str,
) -> bool:
    token = str(value_contains or "").lower()
    if not token:
        return False
    for item in _iter_history_items(history):
        if not isinstance(item, dict) or item.get("result") not in {None, "passed"}:
            continue
        step = item.get("step")
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "").lower()
        if not action.startswith("assert_"):
            continue
        blob = " ".join(
            str(step.get(key) or "") for key in ("value", "target", "selector")
        ).lower()
        if token in blob:
            return True
    return False


def _history_action_satisfies(
    history: list[dict[str, Any]],
    *,
    actions: set[str],
    terms: list[str],
    require_observed_progress: bool = False,
) -> bool:
    normalized_terms = [
        _normalized_criterion_match_text(term)
        for term in terms
        if _normalized_criterion_match_text(term)
    ]
    for item in _iter_history_items(history):
        if not isinstance(item, dict) or item.get("result") not in {None, "passed"}:
            continue
        step = item.get("step")
        if not isinstance(step, dict):
            continue
        if _step_action(step).lower() not in actions:
            continue
        if not normalized_terms:
            return True
        if _semantic_terms_match_values(
            normalized_terms,
            _history_step_semantic_values(step),
        ) and (not require_observed_progress or _history_step_executed(step)):
            return True
    return False


def _iter_history_items(history: list[dict[str, Any]]):
    yield from _expanded_history_items(history)


def _expanded_history_items(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        expanded.append(item)
        step = item.get("step")
        if not isinstance(step, dict):
            continue
        module_steps = step.get("_module_executed_steps")
        if not isinstance(module_steps, list):
            continue
        for module_step in module_steps:
            if not isinstance(module_step, dict):
                continue
            expanded.append(
                {
                    "index": item.get("index"),
                    "source": "module_step",
                    "step": module_step,
                    "decision": item.get("decision"),
                    "result": item.get("result", "passed"),
                    "url_after": item.get("url_after"),
                }
            )
    return expanded


def _history_step_semantic_values(step: dict[str, Any]) -> list[str]:
    values: list[str] = []
    selector = str(step.get("selector") or "").strip()
    executed_selector = str(step.get("_action_executed_selector") or "").strip()
    selector_text = _text_query_from_selector(selector)
    executed_selector_text = _text_query_from_selector(executed_selector)
    for value in (
        step.get("target"),
        step.get("value"),
        step.get("_resolved_value"),
        step.get("key"),
        step.get("use_module"),
        step.get("_action_target_text"),
        step.get("_resolved_value_after"),
        executed_selector_text,
        selector_text,
    ):
        normalized = _normalized_criterion_match_text(value)
        if normalized:
            values.append(normalized)
            stripped = _normalized_criterion_match_text(
                _strip_action_words(normalized)
            )
            if stripped and stripped != normalized:
                values.append(stripped)
            values.extend(
                _normalized_criterion_match_text(variant)
                for variant in semantic_text_variants(normalized)
                if _normalized_criterion_match_text(variant)
            )
    for selector_value in (executed_selector, selector):
        selector_words = _normalize_identifier_words(selector_value)
        if selector_words:
            values.append(selector_words)
            if _looks_like_simple_element_key(selector_value):
                values.extend(
                    token
                    for token in selector_words.split()
                    if len(token) >= 3 and token not in {"btn", "button"}
                )
        if selector_value:
            normalized_selector = _normalized_criterion_match_text(selector_value)
            if normalized_selector:
                values.append(normalized_selector)
    return _dedupe_texts(values)


def _looks_like_simple_element_key(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and bool(re.fullmatch(r"#?[A-Za-z][A-Za-z0-9_-]*", text))


def _history_step_semantic_blob(step: dict[str, Any]) -> str:
    return " ".join(_history_step_semantic_values(step))


def _semantic_terms_match_values(terms: list[str], values: list[str]) -> bool:
    normalized_terms = _dedupe_texts(
        [
            _normalized_criterion_match_text(variant)
            for term in terms
            if term
            for variant in ([term] + (semantic_text_variants(term) or []))
            if _normalized_criterion_match_text(variant)
        ]
    )
    if not normalized_terms:
        return True
    if _generic_login_terms_only(normalized_terms) and any(
        _value_is_oa_login(value) for value in values
    ):
        return False
    if any(
        _term_matches_value(term, value, allow_single_word_subset=False)
        for term in normalized_terms
        for value in values
    ):
        return True
    cjk_terms = [
        term for term in normalized_terms if re.search(r"[\u4e00-\u9fff]", term)
    ]
    if len(cjk_terms) >= 2 and all(
        any(
            _term_matches_value(term, value, allow_single_word_subset=False)
            for value in values
        )
        for term in cjk_terms
    ):
        return True
    ascii_terms = [
        term
        for term in normalized_terms
        if re.search(r"[a-z0-9]", term) and len(term) >= 3
    ]
    if len(ascii_terms) == 1:
        return any(
            _term_matches_value(
                ascii_terms[0],
                value,
                allow_single_word_subset=False,
            )
            for value in values
            if " " not in str(value or "")
        )
    return bool(ascii_terms) and all(
        any(
            _term_matches_value(term, value, allow_single_word_subset=False)
            for value in values
        )
        for term in ascii_terms
    )


def _generic_login_terms_only(terms: list[str]) -> bool:
    login_terms = {"登录", "登陆", "login", "sign in"}
    normalized = {
        _normalized_criterion_match_text(term)
        for term in terms
        if _normalized_criterion_match_text(term)
    }
    return bool(normalized) and normalized <= login_terms


def _value_is_oa_login(value: Any) -> bool:
    text = _normalized_criterion_match_text(value)
    return bool(text) and "oa" in text and any(
        term in text for term in ("登录", "登陆", "login", "sign in")
    )


def _term_matches_value(
    term: str,
    value: str,
    *,
    allow_single_word_subset: bool = False,
) -> bool:
    if not term or not value:
        return False
    if value == term:
        return True
    if re.search(r"[\u4e00-\u9fff]", term):
        if len(term) <= 2 and not re.search(r"[a-z0-9]", term):
            return False
        return term in value
    if " " in term:
        if bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", value)):
            return True
        compact_term = re.sub(r"\s+", "", term)
        compact_value = re.sub(r"\s+", "", value)
        return bool(compact_term) and compact_term in compact_value
    if bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", value)):
        value_words = value.split()
        if len(value_words) == 1:
            return True
        if not allow_single_word_subset:
            return False
        selector_text_markers = {"text", "has", "has text", "label", "placeholder"}
        if any(marker in value for marker in selector_text_markers):
            return False
        return term in value_words and len(term) >= 4
    return False


def _history_step_has_observed_progress_in_history(
    history: list[dict[str, Any]],
    item: dict[str, Any],
) -> bool:
    if not isinstance(item, dict):
        return False
    step = item.get("step")
    if not isinstance(step, dict):
        return False
    if _history_step_has_observed_progress(step):
        return True
    action = _step_action(step).lower()
    if action not in {"click", "press", "press_key"}:
        return True
    previous_url = ""
    for history_item in _iter_history_items(history):
        if not isinstance(history_item, dict) or history_item.get("result") not in {
            None,
            "passed",
        }:
            continue
        history_step = history_item.get("step")
        if not isinstance(history_step, dict):
            continue
        url_after = str(history_item.get("url_after") or "")
        if history_item is item:
            return _url_state_changed(previous_url, url_after)
        if url_after:
            previous_url = url_after
    return False


def _history_step_has_observed_progress(step: dict[str, Any]) -> bool:
    if not isinstance(step, dict):
        return False
    if _step_action(step).lower() not in {"click", "press", "press_key"}:
        return True
    if bool(step.get("_action_dom_changed")):
        return True
    before_url = str(step.get("_action_before_url") or "")
    after_url = str(step.get("_action_after_url") or "")
    return _url_state_changed(before_url, after_url)


def _history_step_executed(step: dict[str, Any]) -> bool:
    if not isinstance(step, dict):
        return False
    action = _step_action(step).lower()
    if action not in {"click", "press", "press_key", "fill"}:
        return True
    if step.get("_action_executed_selector"):
        return True
    if step.get("selector") or step.get("target"):
        return True
    return False


def _intent_action_requirements(
    spec: dict[str, Any],
) -> list[tuple[str, list[str], str, str]]:
    source_texts = _agent_natural_plan(spec)
    if not source_texts:
        return []
    requirements: list[tuple[str, list[str], str]] = []
    for text in source_texts:
        text = str(text or "").strip()
        if not text:
            continue
        _, is_module_step = _module_name_from_entry_text(text, modules={})
        if is_module_step:
            continue
        if _first_url(text) and _looks_like_navigation_instruction(text):
            continue
        fill_target, fill_value = _fill_instruction_parts(text, spec=spec)
        if fill_target and fill_value is not None:
            terms = _dedupe_texts(
                _intent_action_terms(fill_target) + _intent_action_terms(fill_value)
            )
            if terms:
                label = f"intent fill: {'/'.join(terms[:3])}"
                row = ("fill", terms, label, text)
                if row not in requirements:
                    requirements.append(row)
        click_target = _click_instruction_target(text)
        if not click_target:
            click_target = _implicit_interaction_target(text)
        if click_target:
            terms = _intent_action_terms(click_target)
            if terms:
                label = f"intent click: {'/'.join(terms[:3])}"
                row = ("click", terms, label, text)
                if row not in requirements:
                    requirements.append(row)
    return requirements[:12]


def _is_input_reference_target(value: Any) -> bool:
    text = " ".join(str(value or "").split()).strip()
    return text.startswith(("中的", "里的", "内的", "中 ", "里 ", "内 "))


def _intent_action_terms(value: Any) -> list[str]:
    text = _normalize_criterion_evidence_term(value)
    if not text:
        return []
    text = _strip_action_words(text)
    terms: list[str] = []
    if text:
        terms.append(text)
    terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text))
    terms.extend(re.findall(r"[\u4e00-\u9fff]{2,}", text))
    normalized_terms: list[str] = []
    generic_terms = {
        "click",
        "tap",
        "press",
        "select",
        "choose",
        "open",
        "view",
        "use",
        "fill",
        "type",
        "enter",
        "input",
        "button",
        "link",
        "field",
        "menu",
    }
    for term in terms:
        normalized = _normalized_criterion_match_text(
            _normalize_criterion_evidence_term(term)
        )
        if len(normalized) < 2:
            continue
        if normalized in generic_terms:
            continue
        if normalized not in normalized_terms:
            normalized_terms.append(normalized)
    return normalized_terms[:6]


def _strip_action_words(value: Any) -> str:
    text = str(value or "").strip()
    for prefix in (
        "查看",
        "点击",
        "单击",
        "点选",
        "选择",
        "使用",
        "选用",
        "启用",
        "展开",
        "use",
        "click",
        "tap",
        "press",
        "select",
        "choose",
        "enable",
        "expand",
        "打开",
        "输入",
        "键入",
        "填写",
        "填入",
        "录入",
    ):
        if text.startswith(prefix) and len(text) > len(prefix):
            return text[len(prefix) :].strip()
    return text


def _normalized_criterion_match_text(value: Any) -> str:
    text = _normalized_goal_text(value)
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", text)
    return text


def _criterion_evidence_terms(criterion: str) -> list[str]:
    text = str(criterion or "").strip()
    terms: list[str] = []
    terms.extend(re.findall(r"[\"'“”‘’]([^\"'“”‘’]{2,80})[\"'“”‘’]", text))
    terms.extend(
        match.strip(" ，。,.：:")
        for match in re.findall(
            r"([\u4e00-\u9fffA-Za-z0-9 _.-]{2,40}?)"
            r"(?:输入框|文本框|字段|按钮|链接)?已(?:输入|填写|填入|执行|点击)",
            text,
            flags=re.I,
        )
    )
    terms.extend(
        match.strip(" ，。,.：:")
        for match in re.findall(
            r"(?:展示|显示|包含|存在|进入|打开|停留在|仍在|位于|为)\s*"
            r"([\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fffA-Za-z0-9 _:'!?.:/-]{1,80})",
            text,
            flags=re.I,
        )
    )
    terms.extend(
        match.strip(" ，。,.：:")
        for match in re.findall(
            r"(?:展示|显示|包含|存在|为|equals?|contains?)\s*([A-Za-z0-9][A-Za-z0-9 _:'!?.-]{1,80})",
            text,
            flags=re.I,
        )
    )
    terms.extend(
        match.strip(" ，。,.：:")
        for match in re.findall(
            r"([A-Za-z][A-Za-z0-9:'!?.-]+(?:\s+[A-Za-z0-9:'!?.-]+){1,8})", text
        )
    )
    terms.extend(re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{1,60}\b", text))
    deduped: list[str] = []
    for term in terms:
        normalized = _normalize_criterion_evidence_term(term)
        if len(normalized) < 2:
            continue
        if normalized.lower() in {"visible", "exists", "page", "button"}:
            continue
        if normalized not in deduped:
            deduped.append(normalized)
    return deduped[:6]


def _normalize_criterion_evidence_term(value: Any) -> str:
    normalized = " ".join(str(value or "").split()).strip(" ，。,.：:")
    if not normalized:
        return ""
    cjk_prefixes = (
        "当前页面仍在",
        "当前页面位于",
        "当前页面进入",
        "当前页面打开",
        "当前页面显示",
        "当前页面包含",
        "页面仍在",
        "页面位于",
        "页面进入",
        "页面打开",
        "页面显示",
        "页面包含",
        "页面标题包含",
        "标题包含",
    )
    changed = True
    while changed:
        changed = False
        for prefix in cjk_prefixes:
            if normalized.startswith(prefix) and len(normalized) > len(prefix):
                normalized = normalized[len(prefix) :].strip()
                changed = True
                break
    for suffix in ("业务页面", "业务页", "页面", "页"):
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            normalized = normalized[: -len(suffix)].strip()
            break
    for suffix in (
        "输入框",
        "文本框",
        "搜索框",
        "查询框",
        "字段",
        "按钮",
        "链接",
        "菜单",
        "button",
        "link",
        "menu",
        "field",
        "input",
        "textbox",
    ):
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            normalized = normalized[: -len(suffix)].strip()
            break
    return normalized


def _dom_contains_text(dom_context: dict[str, Any], needle: str) -> bool:
    token = str(needle or "").lower()
    texts: list[str] = []
    page_summary = (
        dom_context.get("page_summary") if isinstance(dom_context, dict) else {}
    )
    meta = dom_context.get("meta") if isinstance(dom_context, dict) else {}
    if isinstance(meta, dict):
        texts.extend([meta.get("title") or "", meta.get("url") or ""])
    if isinstance(page_summary, dict):
        texts.extend(page_summary.get("visible_text_summary") or [])
        texts.append(page_summary.get("main_heading") or "")
    for item in (dom_context.get("interactive_elements") or []) + (
        dom_context.get("assertion_candidates") or []
    ):
        if isinstance(item, dict):
            texts.extend(
                [
                    item.get("text") or "",
                    item.get("name") or "",
                    item.get("near_text") or "",
                ]
            )
    return any(token in str(value or "").lower() for value in texts)


def normalize_agent_case(case_data: dict[str, Any]) -> dict[str, Any]:
    """Convert the user-facing agent_case schema into the internal runtime model."""
    if str(case_data.get("type") or "").lower() != "agent_case":
        return {}

    criteria = _criteria_from_case(case_data)
    result: dict[str, Any] = {
        "intent": case_data.get("intent"),
        "steps": copy.deepcopy(case_data.get("steps") or []),
        "inputs": copy.deepcopy(case_data.get("inputs") or {}),
        "criteria": criteria,
    }
    return result


def _agent_spec_to_generation_spec(
    *,
    case_name: str,
    spec: dict[str, Any],
    allowed_modules: list[str] | None = None,
) -> dict[str, Any]:
    criteria = spec.get("criteria") if isinstance(spec, dict) else {}
    checkpoints = []
    final = []
    if isinstance(criteria, dict):
        checkpoints = _normalize_text_list(criteria.get("checkpoints"))
        final = _normalize_text_list(criteria.get("final"))
    case_spec: dict[str, Any] = {"name": case_name}
    if spec.get("steps"):
        case_spec["steps"] = copy.deepcopy(spec.get("steps") or [])
    if spec.get("intent"):
        case_spec["intent"] = spec.get("intent")
    if checkpoints:
        case_spec["checkpoints"] = checkpoints
    if final:
        case_spec["final"] = final
    result: dict[str, Any] = {
        "description": spec.get("description") or spec.get("intent") or "",
        "mode": "smart",
        "cases": [case_spec],
        "runtime_compile": {
            "mode": "agent_case",
            "allow_new_modules": False,
            "allowed_modules": list(allowed_modules or []),
        },
    }
    if spec.get("inputs"):
        result["inputs"] = copy.deepcopy(spec.get("inputs") or {})
    if spec.get("intent"):
        result["intent"] = spec.get("intent")
    if spec.get("steps"):
        result["steps"] = copy.deepcopy(spec.get("steps") or [])
    return result


def _criteria_from_case(case_data: dict[str, Any]) -> Any:
    if case_data.get("checkpoints") is not None or case_data.get("final") is not None:
        criteria: dict[str, Any] = {}
        if case_data.get("checkpoints") is not None:
            criteria["checkpoints"] = copy.deepcopy(case_data.get("checkpoints"))
        if case_data.get("final") is not None:
            criteria["final"] = copy.deepcopy(case_data.get("final"))
        return criteria
    return None


def _normalize_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _first_url(value: Any) -> str:
    if isinstance(value, str):
        match = _URL_RE.search(value)
        return _clean_extracted_url(match.group(0)) if match else ""
    if isinstance(value, list):
        for item in value:
            url = _first_url(item)
            if url:
                return url
    if isinstance(value, dict):
        for item in value.values():
            url = _first_url(item)
            if url:
                return url
    return ""


def _clean_extracted_url(url: str) -> str:
    cleaned = str(url or "").strip()
    cleaned = cleaned.rstrip(".,;:!?，。；：！？、")
    if cleaned != "https://" and cleaned != "http://":
        cleaned = cleaned.rstrip("/")
    return cleaned


def _first_module_url(modules: dict[str, Any]) -> str:
    if not isinstance(modules, dict):
        return ""
    for steps in modules.values():
        url = _first_module_url_from_steps(steps)
        if url:
            return url
    return ""


def _first_module_url_from_steps(steps: Any) -> str:
    if isinstance(steps, dict):
        return _first_module_url_from_steps(steps.get("steps") or [])
    if not isinstance(steps, list):
        return ""
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "").lower()
        value = step.get("value") or step.get("url")
        if action in {"goto", "navigate", "open", "打开", "访问"}:
            url = _first_url(value)
            if url:
                return url
    return ""


def _is_external_url(*, current_url: str, next_url: str, fallback_url: str) -> bool:
    next_host = urlparse(next_url).netloc
    if not next_host:
        return False
    base_host = urlparse(
        current_url if current_url != "about:blank" else fallback_url
    ).netloc
    fallback_host = urlparse(fallback_url).netloc
    if next_host and fallback_host and next_host == fallback_host:
        return False
    return bool(base_host and next_host and next_host != base_host)


def _is_truncated_current_url_prefix(*, current_url: str, next_url: str) -> bool:
    current = str(current_url or "").strip()
    candidate = str(next_url or "").strip()
    if not current or not candidate or current == "about:blank":
        return False
    parsed_candidate = urlparse(candidate)
    parsed_current = urlparse(current)
    if not parsed_candidate.scheme or not parsed_candidate.netloc:
        return False
    if not parsed_current.scheme or not parsed_current.netloc:
        return False
    if candidate == current:
        return False
    return current.startswith(candidate) and len(candidate) < len(current)


def _plan_cache_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "description": spec.get("description"),
        "intent": spec.get("intent"),
        "steps": spec.get("steps"),
        "inputs": spec.get("inputs"),
        "criteria": spec.get("criteria"),
        "entry_scope": spec.get("entry_scope"),
    }


def _cacheable_plan_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_clean_plan_step(step) for step in steps if isinstance(step, dict)]


def _cacheable_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    if not isinstance(result, dict):
        return {}
    result["elements"] = {}
    result["modules"] = {}
    result["vars"] = {}
    return result


def _compiled_payload_safe_for_plan_cache(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    for key in ("elements", "modules", "vars"):
        value = payload.get(key)
        if isinstance(value, dict) and value:
            return False
    return True


def _normalize_runtime_compile_payload(
    *,
    context: ProjectContext,
    spec: dict[str, Any],
    output_name: str,
    payload: dict[str, Any],
    use_ai: bool,
    progress: Any = None,
    artifacts: Any = None,
) -> tuple[dict[str, Any], list[str]]:
    _validate_runtime_compiled_payload(
        payload,
        allowed_modules=set(context.modules or {}),
    )
    harness = GenerationHarness(context=context, spec=spec, output_name=output_name)
    normalized = harness.normalize(payload)
    _validate_runtime_compiled_payload(
        normalized,
        allowed_modules=set(context.modules or {}),
    )
    warnings = harness.validate(normalized)
    return normalized, warnings


def _validate_runtime_compiled_payload(
    payload: dict[str, Any],
    *,
    allowed_modules: set[str],
) -> None:
    if not isinstance(payload, dict):
        raise AgentCaseCompileContractError("Agent编译结果必须是对象")
    raw_modules = payload.get("modules") or {}
    if not isinstance(raw_modules, dict):
        raise AgentCaseCompileContractError(
            "Agent编译结果违反运行时资产边界: "
            "run_case agent_case 的 modules 必须是空对象 {}"
        )
    generated_modules = sorted(raw_modules.keys())
    if generated_modules:
        raise AgentCaseCompileContractError(
            "Agent编译结果违反运行时资产边界: "
            "run_case agent_case 不允许新建module，"
            f"发现 modules={generated_modules}"
        )
    for case_name, case_data in (payload.get("data") or {}).items():
        steps = case_data.get("steps") if isinstance(case_data, dict) else None
        if not isinstance(steps, list):
            continue
        _validate_runtime_module_refs(
            steps,
            allowed_modules=allowed_modules,
            owner=f"data.{case_name}",
        )


def _validate_runtime_module_refs(
    steps: list[Any],
    *,
    allowed_modules: set[str],
    owner: str,
) -> None:
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        module_name = step.get("use_module")
        if not module_name:
            continue
        module_name = str(module_name)
        if module_name not in allowed_modules:
            raise AgentCaseCompileContractError(
                "Agent编译结果违反运行时资产边界: "
                "run_case agent_case 只能引用当前项目YAML中已存在的module，"
                f"{owner} step {index} 引用了不存在的module: {module_name}"
            )


def _first_payload_case_name(payload: dict[str, Any], fallback: str) -> str:
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if isinstance(cases, list):
        for item in cases:
            if isinstance(item, dict) and item.get("name"):
                return str(item["name"])
    return fallback


def _clean_history_step(step: dict[str, Any]) -> dict[str, Any]:
    cleaned = {
        key: copy.deepcopy(value)
        for key, value in step.items()
        if not str(key).startswith("_module_")
    }
    module_steps = step.get("_module_executed_steps")
    if isinstance(module_steps, list):
        cleaned["_module_executed_steps"] = [
            _clean_history_step(module_step)
            for module_step in module_steps
            if isinstance(module_step, dict)
        ]
    return cleaned


def _clean_plan_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in step.items()
        if (
            (
                not str(key).startswith("_")
                or key in {"_resolved_selector", "_resolved_value", "_resolved_value_after"}
            )
            and not str(key).startswith("_module_")
        )
    }


def _hash_payload(data: dict[str, Any]) -> str:
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _json_payload(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)
