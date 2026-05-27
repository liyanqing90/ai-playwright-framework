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

from src.ai_generation.project_context import (
    ProjectContext,
    load_project_context,
)
from src.ai_runtime.ai_cache_store import AiCacheStore, ai_cache_path_from_config
from src.ai_runtime.cache_scope import context_asset_fingerprint, resolve_entry_scope
from src.ai_runtime.config import load_ai_config
from src.ai_runtime.contracts import AgentCaseDecision
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
)
from src.ai_runtime.playwright_selectors import collect_candidates
from src.ai_runtime.provider import ChatCompletionProvider, load_llm_settings
from src.step_actions.step_executor import StepExecutor
from utils.logger import logger


_URL_RE = re.compile(r"https?://[^\s\"'，,。)）]+", re.IGNORECASE)
_PAYMENT_KEYWORDS = ("支付", "付款", "真实支付", "payment", "pay now", "credit card")
_DEFAULT_ALLOWED_ACTIONS = {
    "goto",
    "use_module",
    "click",
    "fill",
    "press",
    "wait",
    "assert_visible",
    "assert_text",
    "assert_url_contains",
    "assert_title",
}
_DEFAULT_GUARDRAILS = {
    "no_runtime_registry_write": True,
    "stop_on_external_domain": True,
    "stop_on_unexpected_payment": True,
    "require_checkpoints_or_final": True,
}


@dataclass
class AgentCaseRunResult:
    case_name: str
    steps_executed: int
    model_calls: int
    cache_replayed_steps: int = 0
    decisions: list[dict[str, Any]] = field(default_factory=list)
    final_reason: str = ""


class AgentCaseAdvisoryCache:
    """Validated replay hints for agent_case. Cache misses never fail a case."""

    def __init__(self, path: str | Path):
        self.store = AiCacheStore(path)
        self.namespace = "agent_trace"

    def load_trace(self, key: str) -> list[dict[str, Any]]:
        try:
            record = self.store.get_payload(namespace=self.namespace, key=key) or {}
            trace = record.get("trace") or []
            return trace if isinstance(trace, list) else []
        except Exception as exc:
            logger.warning(f"agent_case缓存读取失败，忽略缓存: {exc}")
            return []

    def save_trace(
        self,
        *,
        key: str,
        project: str,
        env: str,
        case_name: str,
        intent: str,
        steps: list[str],
        inputs: dict[str, Any],
        trace: list[dict[str, Any]],
        final_reason: str,
        entry_scope: str = "",
        prompt_version: str = "",
        schema_version: str = "",
        model: str = "",
        asset_hash: str = "",
    ) -> None:
        try:
            spec_hash = _hash_payload(
                {"intent": intent, "steps": steps, "inputs": inputs}
            )
            self.store.put_payload(
                namespace=self.namespace,
                key=key,
                project=project,
                env=env,
                entry_scope=entry_scope,
                case_name=case_name,
                input_type="steps" if steps else "intent",
                model=model,
                prompt_version=prompt_version,
                schema_version=schema_version,
                spec_hash=spec_hash,
                asset_hash=asset_hash,
                payload={
                    "final_reason": final_reason,
                    "trace": _cacheable_trace(trace),
                },
                metadata={
                    "intent": intent,
                    "steps": steps,
                    "inputs": inputs,
                    "updated_at": int(time.time()),
                },
            )
        except Exception as exc:
            logger.warning(f"agent_case缓存写入失败，不阻塞执行: {exc}")

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
        runtime_cfg = self.ai_config.get("runtime", {})
        agent_policy = self.ai_config.get("agent_policy", {})
        policy_limits = agent_policy.get("limits") or {}
        self.default_guardrails = {
            **_DEFAULT_GUARDRAILS,
            **(agent_policy.get("guardrails") or {}),
        }
        self.default_allowed_actions = set(
            agent_policy.get("allowed_actions") or _DEFAULT_ALLOWED_ACTIONS
        )
        generation_cfg = self.ai_config.get("generation", {})
        self.max_steps_default = int(
            policy_limits.get("max_steps")
            or 20
        )
        self.max_model_calls_default = int(
            policy_limits.get("max_model_calls")
            or 12
        )
        self.max_duration_default = int(
            policy_limits.get("max_duration_seconds")
            or 180
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
        self.cache_enabled = bool(runtime_cfg.get("agent_case_cache_enabled", True))
        self.cache_max_replay_steps = int(
            runtime_cfg.get("agent_case_cache_max_replay_steps", 30)
        )
        self.cache = AgentCaseAdvisoryCache(ai_cache_path_from_config(self.ai_config))
        self.max_context_items = int(
            runtime_cfg.get(
                "agent_context_items",
                min(int(generation_cfg.get("max_context_items", 160)), 40),
            )
        )
        self.history_limit = int(runtime_cfg.get("agent_history_limit", 10))
        prompts_cfg = self.ai_config.get("prompts", {})
        llm_cfg = self.ai_config.get("llm", {})
        self.prompt_version = str(prompts_cfg.get("agent_case_version", "agent-case-v1"))
        self.schema_version = str(llm_cfg.get("schema_version", "ui-ai-schema-v1"))
        self.step_executor = StepExecutor(
            page,
            ui_helper,
            self.elements,
            default_mode="smart",
        )
        self.current_dom_context: dict[str, Any] = {}
        self.last_decision_used_model = False

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

        cache_replayed = self._try_replay_cache(
            key=cache_key,
            case_name=case_name,
            spec=spec,
            history=history,
        )
        if cache_replayed == 0:
            self._maybe_open_start_url(
                case_name=case_name,
                case_data=case_data,
                intent=spec["intent"],
                steps=spec["steps"],
                history=history,
            )

        model_calls = 0
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

            decision = self._decide_next_action(
                case_name=case_name,
                spec=spec,
                history=history,
                step_index=len(history) + 1,
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
                unmet_final = _unmet_final_criteria(
                    criteria=spec.get("criteria"),
                    history=history,
                    current_url=getattr(self.page, "url", ""),
                    dom_context=self.current_dom_context,
                )
                if unmet_final:
                    logger.warning(
                        "Agent premature finish rejected: "
                        f"case={case_name} | unmet_final={unmet_final}"
                    )
                    history.append(
                        self._history_item(
                            step={
                                "action": "wait",
                                "value": "0",
                                "target": "finish rejected; final criteria still unmet",
                            },
                            source="finish_guard",
                            decision=decision,
                            result="rejected",
                        )
                    )
                    continue
                logger.info(
                    "Agent验收判定通过: "
                    f"case={case_name} | {_criteria_summary(spec['criteria'])} "
                    f"| evidence={normalize_model_text(decision.reason)}"
                )
                result = AgentCaseRunResult(
                    case_name=case_name,
                    steps_executed=len(history),
                    model_calls=model_calls,
                    cache_replayed_steps=cache_replayed,
                    decisions=decisions,
                    final_reason=normalize_model_text(decision.reason),
                )
                self._save_success_cache(
                    key=cache_key,
                    case_name=case_name,
                    spec=spec,
                    history=history,
                    final_reason=result.final_reason,
                )
                logger.info(
                    "Agent用例执行完成: "
                    f"case={case_name} | steps_executed={result.steps_executed} "
                    f"| model_calls={model_calls} | cache_replayed_steps={cache_replayed} "
                    f"| reason={result.final_reason}"
                )
                return result

            if decision.action == "fail":
                raise AssertionError(
                    "Agent用例判定失败: "
                    f"case={case_name} | reason={normalize_model_text(decision.reason)}"
                )

            self._guard_decision(decision, spec=spec)
            step = self._decision_to_step(decision)
            logger.info(
                "Agent执行动作: "
                f"case={case_name} | step={len(history) + 1}/{spec['max_steps']} "
                f"| {_format_step(step)}"
            )
            self._execute_step(step, spec=spec)
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

        raise AssertionError(
            f"Agent用例超过最大步骤数仍未完成: case={case_name} | max_steps={spec['max_steps']}"
        )

    def _agent_spec(self, *, case_name: str, case_data: dict[str, Any]) -> dict[str, Any]:
        agent_case = normalize_agent_case(case_data)
        if not agent_case:
            raise ValueError(f"Agent用例缺少 intent/checkpoints/final 配置: {case_name}")
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
            "allowed_actions": set(self.default_allowed_actions),
            "guardrails": self.default_guardrails,
            "max_steps": self.max_steps_default,
            "max_model_calls": self.max_model_calls_default,
            "max_duration_seconds": self.max_duration_default,
        }

    def _try_replay_cache(
        self,
        *,
        key: str,
        case_name: str,
        spec: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> int:
        if not self.cache_enabled:
            return 0
        trace = self.cache.load_trace(key)
        if not trace:
            return 0
        replayed = 0
        logger.info(
            "Agent用例缓存命中: "
            f"case={case_name} | cached_steps={len(trace)} | mode=advisory"
        )
        for item in trace[: self.cache_max_replay_steps]:
            step = item.get("step") if isinstance(item, dict) else None
            if not isinstance(step, dict):
                continue
            try:
                logger.info(
                    "Agent缓存回放动作: "
                    f"case={case_name} | step={replayed + 1}/{len(trace)} "
                    f"| {_format_step(step)}"
                )
                self._execute_step(copy.deepcopy(step), spec=spec)
                history.append(self._history_item(step=step, source="cache_replay"))
                replayed += 1
            except Exception as exc:
                logger.warning(
                    "Agent用例缓存回放失败，切换实时观察: "
                    f"case={case_name} | replayed={replayed} | error={exc}"
                )
                break
        return replayed

    def _maybe_open_start_url(
        self,
        *,
        case_name: str,
        case_data: dict[str, Any],
        intent: str,
        steps: list[str],
        history: list[dict[str, Any]],
    ) -> None:
        if not self._is_blank_page():
            return
        start_url = (
            _first_url(case_data.get("description"))
            or _first_url(intent)
            or _first_url(steps)
            or _first_module_url(self.context.modules)
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

    def _execute_step(self, step: dict[str, Any], *, spec: dict[str, Any]) -> None:
        guardrails = spec.get("guardrails") if isinstance(spec, dict) else {}
        disable_registry = bool(
            (guardrails or {}).get("no_runtime_registry_write", True)
        )
        previous = os.environ.get("UI_AI_DISABLE_SELECTOR_REGISTRY")
        if disable_registry:
            os.environ["UI_AI_DISABLE_SELECTOR_REGISTRY"] = "1"
            resolver = getattr(self.step_executor, "smart_resolver", None)
            if resolver is not None:
                resolver.registry = None
                resolver.registry_enabled = False
        try:
            self.step_executor.execute_step(step)
        finally:
            if not disable_registry:
                return
            if previous is None:
                os.environ.pop("UI_AI_DISABLE_SELECTOR_REGISTRY", None)
            else:
                os.environ["UI_AI_DISABLE_SELECTOR_REGISTRY"] = previous

    def _decide_next_action(
        self,
        *,
        case_name: str,
        spec: dict[str, Any],
        history: list[dict[str, Any]],
        step_index: int,
    ) -> AgentCaseDecision:
        candidates = self._safe_collect_candidates()
        compact_candidates = compact_dom_candidates(
            candidates,
            limit=self.agent_candidate_limit,
            hints=[spec["intent"], spec["steps"], spec["criteria"], spec["inputs"]],
        )
        dom_context = build_dom_context(
            candidates,
            url=getattr(self.page, "url", ""),
            title=_safe_page_title(self.page),
            context_level=2,
            limit=self.agent_candidate_limit,
            hints=[spec["intent"], spec["steps"], spec["criteria"], spec["inputs"]],
        )
        self.current_dom_context = dom_context
        logger.info(
            "Agent页面观察: "
            f"case={case_name} | step={step_index}/{spec['max_steps']} "
            f"| url={getattr(self.page, 'url', '')} "
            f"| title={_safe_page_title(self.page)} "
            f"| candidates={len(candidates)} | compact_candidates={len(compact_candidates)} "
            f"| visible_text={_candidate_text_summary(candidates)}"
        )
        logger.debug(
            "Agent用例观察: "
            f"case={case_name} | step={step_index}/{spec['max_steps']} "
                f"| url={getattr(self.page, 'url', '')} | candidates={len(candidates)}"
        )
        self.last_decision_used_model = True
        current_goal = _current_agent_goal(spec=spec, history=history)
        agent_state = _agent_state_summary(
            history=history,
            criteria=spec["criteria"],
        )
        model_dom_context = compact_model_dom_context(
            dom_context,
            candidate_limit=min(self.agent_candidate_limit, 12),
        )
        model_candidates = _lightweight_dom_candidates(dom_context)[:12]
        project_context = compact_project_context(
            self.context,
            max_items=min(self.max_context_items, 16),
            max_modules=3,
            max_module_steps=4,
            hints=[
                spec["intent"],
                current_goal,
                agent_state.get("criteria"),
                spec["inputs"],
            ],
            include_modules=True,
        )
        provider = self._agent_provider()
        return provider.complete_model(
            [
                {
                    "role": "system",
                    "content": (
                        "你是UI自动化运行时Agent。"
                        "你根据当前页面DOM候选、项目资产、缓存回放历史和成功标准，每次只决定下一步一个动作。"
                        "你不是静态用例生成器，不需要一次性输出完整steps。"
                        "必须围绕intent执行，不允许改写业务目标。"
                        "如果natural_steps非空，必须按列表顺序推进；每轮只处理当前尚未满足的自然语言步骤。"
                        "优先返回dom_context里的element_id，不要自己生成selector。"
                        "selector字段只能填写真实CSS/Playwright selector，禁止把e1/e2这类element_id写到selector。"
                        "找不到稳定DOM时返回target，执行层会用smart/vision定位。"
                        "不要返回mode字段，执行器会自动用smart执行标准动作。"
                        "当验收标准全部满足后返回finish；无法继续或页面状态与目标不一致时返回fail。"
                        "信息不足时返回status=need_more_context；危险或外部不可控状态返回status=blocked。"
                        "reason和expected不超过80个中文字符。不要输出完整推理过程。只返回JSON对象。"
                    ),
                },
                {
                    "role": "user",
                    "content": _json_payload(
                        {
                            "case": case_name,
                            "url": getattr(self.page, "url", ""),
                            "prompt_version": self.prompt_version,
                            "schema_version": self.schema_version,
                            "input_type": spec["input_type"],
                            "intent": spec["intent"],
                            "current_goal": current_goal,
                            "natural_steps": _remaining_step_hints(
                                spec=spec,
                                history=history,
                            ),
                            "inputs": spec["inputs"],
                            "criteria": agent_state.get("criteria"),
                            "step_index": step_index,
                            "max_steps": spec["max_steps"],
                            "history": compact_history(
                                history,
                                limit=self.history_limit,
                            ),
                            "agent_state": agent_state,
                            "project_context": project_context,
                            "dom_context": model_dom_context,
                            "dom_candidates": model_candidates,
                            "allowed_actions": sorted(
                                spec["allowed_actions"] | {"done", "finish", "fail"}
                            ),
                            "action_contract": {
                                "goto": "value",
                                "use_module": "module,params?",
                                "click": "selector|target",
                                "fill": "selector|target,value",
                                "press": "key,selector|target?",
                                "wait": "wait_ms",
                                "assert_visible": "selector|target",
                                "assert_text": "selector|target,value",
                                "assert_url_contains": "value",
                                "assert_title": "value",
                                "done": "reason",
                                "finish": "reason",
                                "fail": "reason",
                            },
                        }
                    ),
                },
            ],
            AgentCaseDecision,
            schema_name="AgentCaseDecision",
            usage_operation="runtime.agent_case",
            usage_metadata={
                "project": self.context.project,
                "schema_name": "AgentCaseDecision",
                "prompt_version": self.prompt_version,
                "page_url": getattr(self.page, "url", ""),
                "step_index": step_index,
            },
        )

    def _agent_provider(self) -> ChatCompletionProvider:
        if self.agent_reasoning_effort is None and self.agent_timeout_seconds is None:
            return ChatCompletionProvider()
        try:
            settings = load_llm_settings()
            reasoning_effort = (
                None
                if str(self.agent_reasoning_effort or "").lower() in {"", "none", "false"}
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
                    reasoning_effort=reasoning_effort,
                    timeout_seconds=timeout_seconds,
                )
            )
        except Exception:
            return ChatCompletionProvider()

    def _guard_decision(
        self, decision: AgentCaseDecision, *, spec: dict[str, Any]
    ) -> None:
        if decision.action not in spec["allowed_actions"]:
            raise ValueError(f"Agent动作不在allowed_actions内: {decision.action}")
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
            and _is_external_url(
                current_url=getattr(self.page, "url", ""),
                next_url=decision.value,
                fallback_url=self.context.base_url,
            )
        ):
            raise ValueError(f"Agent动作跳转外部域名，已拦截: {decision.value}")

    def _decision_to_step(self, decision: AgentCaseDecision) -> dict[str, Any]:
        if decision.action == "use_module":
            step: dict[str, Any] = {"use_module": decision.module}
            if decision.params:
                step["params"] = decision.params
            return step
        if decision.action == "wait":
            return {
                "action": "wait",
                "value": str((decision.wait_ms or 1000) / 1000),
            }
        action = "press_key" if decision.action == "press" else decision.action
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
        selected_element_id = decision.element_id or target_as_element_id
        element_selector = selector_for_element_id(
            self.current_dom_context,
            selected_element_id or decision.selector,
        )
        if element_selector:
            step["selector"] = element_selector
        elif selected_element_id:
            raise ValueError(f"Agent返回未知element_id: {selected_element_id}")
        elif looks_like_internal_element_id(decision.target) and not decision.selector:
            raise ValueError(
                f"Agent返回未解析的内部element_id target: {decision.target}"
            )
        if decision.selector and not element_selector:
            step["selector"] = decision.selector
        if action in {"assert_text", "assert_visible"} and not step.get("selector"):
            assertion_text = decision.value or decision.target
            assertion_selector = _selector_for_visible_text(
                self.current_dom_context,
                assertion_text,
            )
            if assertion_selector:
                step["selector"] = assertion_selector
        if decision.target and decision.target != target_as_element_id and not step.get("selector"):
            step["target"] = decision.target
        if decision.value is not None:
            step["value"] = decision.value
        if decision.key:
            step["key"] = decision.key
            step.setdefault("value", decision.key)
        return step

    def _safe_collect_candidates(self) -> list[dict[str, Any]]:
        if self._is_blank_page():
            return []
        try:
            return collect_candidates(self.page, limit=self.agent_candidate_scan_limit)
        except Exception as exc:
            logger.warning(f"Agent DOM观察失败，继续基于上下文决策: {exc}")
            return []

    def _save_success_cache(
        self,
        *,
        key: str,
        case_name: str,
        spec: dict[str, Any],
        history: list[dict[str, Any]],
        final_reason: str,
    ) -> None:
        if not self.cache_enabled:
            return
        self.cache.save_trace(
            key=key,
            project=self.context.project,
            env=self.env,
            case_name=case_name,
            intent=spec["intent"],
            steps=spec["steps"],
            inputs=spec["inputs"],
            trace=history,
            final_reason=final_reason,
            entry_scope=self._entry_scope(spec),
            prompt_version=self.prompt_version,
            schema_version=self.schema_version,
            model=self._model_cache_key(),
            asset_hash=_hash_payload(context_asset_fingerprint(self.context)),
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
            "step": _clean_step(step),
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


def _criteria_summary(criteria: Any) -> str:
    if not isinstance(criteria, dict):
        return "checkpoints=[] | final=[]"
    checkpoints = _short_list(criteria.get("checkpoints"))
    final = _short_list(criteria.get("final"))
    return f"checkpoints={checkpoints} | final={final}"


def _short_list(value: Any, *, limit: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    result = [str(item).strip() for item in value if str(item).strip()]
    if len(result) <= limit:
        return result
    return result[:limit] + [f"...(+{len(result) - limit})"]


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
    return ""


def _safe_page_title(page: Any) -> str:
    try:
        title = page.title()
        return str(title).strip()
    except Exception:
        return ""


def _candidate_text_summary(candidates: list[dict[str, Any]], *, limit: int = 8) -> list[str]:
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


def _lightweight_dom_candidates(dom_context: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not isinstance(dom_context, dict):
        return items
    for section in ("forms", "interactive_elements", "assertion_candidates"):
        values = dom_context.get(section)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            compact = {
                key: item.get(key)
                for key in (
                    "id",
                    "role",
                    "name",
                    "text",
                    "near_text",
                    "type",
                    "selector_candidates",
                )
                if item.get(key) not in (None, "", [])
            }
            if "selector_candidates" in compact and isinstance(compact["selector_candidates"], list):
                compact["selector_candidates"] = compact["selector_candidates"][:2]
            if compact:
                items.append(compact)
    return items[:24]


def _iter_dom_elements(dom_context: dict[str, Any]):
    if not isinstance(dom_context, dict):
        return
    for section in ("forms", "interactive_elements", "assertion_candidates"):
        values = dom_context.get(section)
        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict):
                    yield item


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
    for item in history:
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
    steps = _normalize_text_list(spec.get("steps"))
    if not steps:
        return []
    # The framework cannot know whether an arbitrary natural-language business
    # step is satisfied without becoming domain-specific. Give the model the
    # compact plan plus history and let it choose the next unmet action.
    return steps[: max(0, limit)]


def _current_agent_goal(*, spec: dict[str, Any], history: list[dict[str, Any]]) -> str:
    if _normalize_text_list(spec.get("steps")):
        return "继续完成 natural_steps 中尚未满足的下一步"
    return str(spec.get("intent") or "")


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
    for term in _criterion_evidence_terms(criterion):
        if _dom_contains_text(dom_context, term) or term.lower() in str(current_url or "").lower():
            return True
        if _history_has(history, value_contains=term):
            return True
    return False


def _criterion_evidence_terms(criterion: str) -> list[str]:
    text = str(criterion or "").strip()
    terms: list[str] = []
    terms.extend(re.findall(r"[\"'“”‘’]([^\"'“”‘’]{2,80})[\"'“”‘’]", text))
    terms.extend(
        match.strip(" ，。,.：:")
        for match in re.findall(r"(?:展示|显示|包含|存在|为|equals?|contains?)\s*([A-Za-z0-9][A-Za-z0-9 _:'!?.-]{1,80})", text, flags=re.I)
    )
    terms.extend(
        match.strip(" ，。,.：:")
        for match in re.findall(r"([A-Za-z][A-Za-z0-9:'!?.-]+(?:\s+[A-Za-z0-9:'!?.-]+){1,8})", text)
    )
    terms.extend(re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{1,60}\b", text))
    deduped: list[str] = []
    for term in terms:
        normalized = " ".join(str(term or "").split())
        if len(normalized) < 2:
            continue
        if normalized.lower() in {"visible", "exists", "page", "button"}:
            continue
        if normalized not in deduped:
            deduped.append(normalized)
    return deduped[:6]


def _dom_contains_text(dom_context: dict[str, Any], needle: str) -> bool:
    token = str(needle or "").lower()
    texts: list[str] = []
    page_summary = dom_context.get("page_summary") if isinstance(dom_context, dict) else {}
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


def _agent_state_summary(
    *,
    history: list[dict[str, Any]],
    criteria: Any,
) -> dict[str, Any]:
    recent_actions: list[str] = []
    for item in history[-5:]:
        if not isinstance(item, dict):
            continue
        step = item.get("step")
        if not isinstance(step, dict):
            continue
        target = step.get("target") or step.get("selector") or step.get("use_module") or ""
        action = step.get("action") or ("use_module" if step.get("use_module") else "")
        recent_actions.append(
            f"{action} {target}".strip() + f" -> {item.get('result', 'passed')}"
        )
    pending = []
    if isinstance(criteria, dict):
        pending = _short_list((criteria.get("checkpoints") or []) + (criteria.get("final") or []), limit=8)
    return {
        "steps_done": len(history),
        "recent_actions": recent_actions,
        "criteria": {
            "pending": pending,
            "passed_count": 0,
            "failed_count": 0,
        },
    }


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
        return match.group(0).rstrip("/") if match else ""
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
    base_host = urlparse(current_url if current_url != "about:blank" else fallback_url).netloc
    return bool(base_host and next_host and next_host != base_host)


def _cacheable_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        step = item.get("step")
        if isinstance(step, dict):
            result.append(
                {
                    "source": item.get("source"),
                    "step": _clean_step(step),
                    "decision": item.get("decision"),
                    "result": item.get("result", "passed"),
                    "url_after": item.get("url_after"),
                }
            )
    return result


def _clean_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in step.items()
        if not str(key).startswith("_resolved")
    }


def _hash_payload(data: dict[str, Any]) -> str:
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _json_payload(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)
