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
from src.ai_runtime.playwright_selectors import collect_candidates
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
_AGENT_DECISION_FIELDS = set(AgentCaseDecision.model_fields)
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
_SESSION_EXIT_TERMS = ("logout", "log out", "sign out", "退出", "登出", "注销")
_DESTRUCTIVE_CLICK_TERMS = (
    "remove",
    "delete",
    "reset",
    "clear",
    "discard",
    "移除",
    "删除",
    "重置",
    "清空",
    "丢弃",
)
_MENU_CLICK_TERMS = (
    "open menu",
    "navigation menu",
    "menu button",
    "hamburger",
    "菜单",
    "导航菜单",
)
_MENU_CLOSE_TERMS = (
    "close",
    "dismiss",
    "collapse",
    "关闭",
    "收起",
)
_GENERIC_GOAL_STOPWORDS = (
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "onto",
    "then",
    "this",
    "that",
    "page",
    "button",
    "link",
    "input",
    "field",
    "visible",
    "exists",
    "contains",
    "complete",
    "current",
    "standard",
    "user",
)


class AgentDecisionRejected(ValueError):
    """Recoverable Agent decision rejection; the next loop asks the model again."""


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

    def mark_stale(self, key: str, *, reason: str) -> None:
        try:
            self.store.mark_stale(namespace=self.namespace, key=key, reason=reason)
        except Exception as exc:
            logger.warning(f"agent_case缓存标记stale失败，不阻塞执行: {exc}")


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
        self.default_allowed_actions = set(
            agent_policy.get("allowed_actions") or _DEFAULT_ALLOWED_ACTIONS
        )
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
        self.agent_model = str(runtime_cfg.get("agent_model") or "").strip()
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
        if cache_replayed and _has_completion_criteria(spec["criteria"]):
            unmet_after_replay = self._wait_for_completion_criteria(
                spec=spec,
                history=history,
                timeout_seconds=self.agent_completion_wait_seconds,
            )
            if not unmet_after_replay:
                final_reason = "cache replay satisfies final criteria"
                logger.info(
                    "Agent缓存回放满足最终验收: "
                    f"case={case_name} | replayed={cache_replayed} "
                    f"| {_criteria_summary(spec['criteria'])}"
                )
                result = AgentCaseRunResult(
                    case_name=case_name,
                    steps_executed=len(history),
                    model_calls=0,
                    cache_replayed_steps=cache_replayed,
                    decisions=decisions,
                    final_reason=final_reason,
                )
                logger.info(
                    "Agent用例执行完成: "
                    f"case={case_name} | steps_executed={result.steps_executed} "
                    f"| model_calls=0 | cache_replayed_steps={cache_replayed} "
                    f"| reason={result.final_reason}"
                )
                return result
            logger.info(
                "Agent缓存回放未满足最终验收，切换实时观察: "
                f"case={case_name} | replayed={cache_replayed} "
                f"| unmet_final={unmet_after_replay}"
            )
            self.cache.mark_stale(
                cache_key,
                reason=f"cache replay unmet criteria: {unmet_after_replay}",
            )
            history.clear()
            cache_replayed = 0
            self._maybe_open_start_url(
                case_name=case_name,
                case_data=case_data,
                intent=spec["intent"],
                steps=spec["steps"],
                history=history,
                force=True,
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
                unmet_final = _unmet_completion_criteria(
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
            try:
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
            if _has_completion_criteria(spec["criteria"]):
                unmet_after_step = self._wait_for_completion_criteria(
                    spec=spec,
                    history=history,
                    timeout_seconds=_completion_wait_seconds_for_step(
                        step,
                        default_seconds=self.agent_completion_wait_seconds,
                    ),
                )
                if not unmet_after_step:
                    final_reason = "final criteria satisfied after executed step"
                    result = AgentCaseRunResult(
                        case_name=case_name,
                        steps_executed=len(history),
                        model_calls=model_calls,
                        cache_replayed_steps=cache_replayed,
                        decisions=decisions,
                        final_reason=final_reason,
                    )
                    self._save_success_cache(
                        key=cache_key,
                        case_name=case_name,
                        spec=spec,
                        history=history,
                        final_reason=final_reason,
                    )
                    logger.info(
                        "Agent执行后最终验收通过: "
                        f"case={case_name} | steps_executed={result.steps_executed} "
                        f"| model_calls={model_calls} "
                        f"| cache_replayed_steps={cache_replayed}"
                    )
                    return result

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
            logger.info(
                "Agent用例缓存未命中: " f"case={case_name} | key_prefix={key[:12]}"
            )
            return 0
        replayed = 0
        skip_cached_assertions = _has_completion_criteria(spec.get("criteria"))
        logger.info(
            "Agent用例缓存命中: "
            f"case={case_name} | cached_steps={len(trace)} | mode=advisory"
        )
        for item in trace[: self.cache_max_replay_steps]:
            step = item.get("step") if isinstance(item, dict) else None
            if not isinstance(step, dict):
                continue
            if skip_cached_assertions and _is_assertion_step(step):
                logger.info(
                    "Agent缓存回放跳过断言动作，改由最终验收统一判断: "
                    f"case={case_name} | step={replayed + 1}/{len(trace)} "
                    f"| {_format_step(step)}"
                )
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

    def _refresh_dom_context(self, *, spec: dict[str, Any]) -> None:
        candidates = self._safe_collect_candidates()
        self.current_dom_context = build_dom_context(
            candidates,
            url=getattr(self.page, "url", ""),
            title=_safe_page_title(self.page),
            context_level=2,
            limit=self.agent_candidate_limit,
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

    def _maybe_open_start_url(
        self,
        *,
        case_name: str,
        case_data: dict[str, Any],
        intent: str,
        steps: list[str],
        history: list[dict[str, Any]],
        force: bool = False,
    ) -> None:
        if not force and not self._is_blank_page():
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
        if disable_registry:
            os.environ["UI_AI_DISABLE_SELECTOR_REGISTRY"] = "1"
            resolver = getattr(self.step_executor, "smart_resolver", None)
            if resolver is not None:
                resolver.registry = None
                resolver.registry_enabled = False
        try:
            self.step_executor.execute_step(step)
            if getattr(self.step_executor, "step_has_error", False):
                raise AssertionError(f"Agent动作执行失败: {_format_step(step)}")
            self._sync_runtime_page_from_step_executor()
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
        finally:
            if not disable_registry:
                return
            if previous is None:
                os.environ.pop("UI_AI_DISABLE_SELECTOR_REGISTRY", None)
            else:
                os.environ["UI_AI_DISABLE_SELECTOR_REGISTRY"] = previous

    def _sync_runtime_page_from_step_executor(self) -> None:
        ui_page = getattr(getattr(self.step_executor, "ui_helper", None), "page", None)
        step_page = getattr(self.step_executor, "page", None)
        page = ui_page or step_page
        if page is not None and page is not self.page:
            self._adopt_runtime_page(page, reason="step executor page changed")

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
            candidate_limit=min(self.agent_candidate_limit, 10),
            selector_limit=1,
            form_limit=4,
            assertion_limit=6,
            include_business_objects=False,
            include_compression=False,
        )
        criteria_prompt = _criteria_prompt_summary(
            spec["criteria"],
            checkpoint_limit=4 if spec["input_type"] == "intent" else 3,
        )
        project_context = self._agent_project_context_for_prompt(
            spec=spec,
            current_goal=current_goal,
            criteria_prompt=criteria_prompt,
            history=history,
        )
        local_decision = self._local_module_decision(
            spec=spec,
            history=history,
            dom_context=dom_context,
        )
        if local_decision is not None:
            self.last_decision_used_model = False
            logger.info(
                "Agent本地模块决策: "
                f"module={local_decision.module} | reason={local_decision.reason}"
            )
            return local_decision
        provider = self._agent_provider()
        messages = [
            {
                "role": "system",
                "content": (
                    "你是UI自动化运行时Agent。每次只返回一个JSON动作。"
                    "围绕intent/current_goal推进；natural_steps非空时按顺序处理。"
                    "优先返回dom_context中的element_id；selector只能是真实选择器，不能写e1/e2。"
                    "fill必须带value；页面可见标题用assert_text/assert_visible，不用assert_title。"
                    "use_module如果模块占位符不是项目全局变量，必须在params传入对应值。"
                    "禁止输出thought/analysis/steps/mode等契约外字段。"
                    "除非目标明确要求，禁止点击退出、删除、重置、清空等破坏状态的动作。"
                    "验收满足后finish；无法继续fail；信息不足need_more_context。"
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
                        "criteria": criteria_prompt,
                        "step_index": step_index,
                        "max_steps": spec["max_steps"],
                        "history": compact_history(
                            history,
                            limit=self.history_limit,
                        ),
                        "agent_state": agent_state,
                        "project_context": project_context,
                        "dom_context": model_dom_context,
                        "allowed_actions": sorted(
                            spec["allowed_actions"] | {"done", "finish", "fail"}
                        ),
                        "action_contract": (
                            "click/fill/assert_* use element_id|selector|target; "
                            "fill also needs value; goto/assert_url_contains use value; "
                            "use_module uses module and required params; finish/fail use reason"
                        ),
                    }
                ),
            },
        ]
        try:
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
                },
            )
        except Exception as exc:
            local_decision = (
                self._local_module_decision(
                    spec=spec,
                    history=history,
                    dom_context=dom_context,
                )
                if _is_model_timeout_error(exc)
                else None
            )
            if local_decision is not None:
                self.last_decision_used_model = False
                logger.warning(
                    "Agent模型超时，改用本地模块决策: "
                    f"module={local_decision.module} | error={exc}"
                )
                return local_decision
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
        ):
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
            raise ValueError(f"Agent动作跳转外部域名，已拦截: {decision.value}")

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
        if action == "click":
            repaired_selector = _selector_for_guarded_click(
                page=self.page,
                dom_context=self.current_dom_context,
                spec=spec or {},
                decision=decision,
                current_selector=step.get("selector"),
            )
            if repaired_selector and repaired_selector != step.get("selector"):
                logger.warning(
                    "Agent点击目标已按通用语义约束修正: "
                    f"from={step.get('selector') or decision.target or decision.element_id} "
                    f"| to={repaired_selector}"
                )
                step["selector"] = repaired_selector
                step.pop("target", None)
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
        return step

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

    def _agent_project_context_for_prompt(
        self,
        *,
        spec: dict[str, Any],
        current_goal: str,
        criteria_prompt: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not _should_include_project_context(history):
            return {
                "project": self.context.project,
                "base_url": self.context.base_url,
            }
        return compact_project_context(
            self.context,
            max_items=min(self.max_context_items, 6),
            max_modules=2,
            max_module_steps=3,
            hints=[
                spec["intent"],
                current_goal,
                criteria_prompt,
                spec["inputs"],
            ],
            include_modules=True,
        )

    def _local_module_decision(
        self,
        *,
        spec: dict[str, Any],
        history: list[dict[str, Any]],
        dom_context: dict[str, Any],
    ) -> AgentCaseDecision | None:
        if not _should_consider_local_module(history, dom_context):
            return None
        best: tuple[int, str, dict[str, Any]] | None = None
        tied = False
        for module_name in sorted((self.context.modules or {}).keys()):
            params, missing = self._infer_module_params(
                module_name,
                spec=spec,
                provided_params={},
            )
            if missing:
                continue
            score = _module_goal_score(
                module_name=module_name,
                module_steps=_module_steps_from_value(
                    self.context.modules[module_name]
                ),
                spec=spec,
                dom_context=dom_context,
            )
            if score < 35:
                continue
            if best is None or score > best[0]:
                best = (score, module_name, params)
                tied = False
            elif score == best[0]:
                tied = True
        if best is None or tied:
            return None
        _, module_name, params = best
        payload: dict[str, Any] = {
            "action": "use_module",
            "module": module_name,
            "reason": "本地匹配项目模块并补齐参数",
        }
        if params:
            payload["params"] = params
        return AgentCaseDecision.model_validate(payload)

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
    finals = _normalize_text_list(criteria.get("final"))
    if finals:
        return finals
    return _normalize_text_list(criteria.get("checkpoints"))


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


def _should_include_project_context(history: list[dict[str, Any]]) -> bool:
    business_actions = 0
    for item in history:
        if not isinstance(item, dict):
            continue
        if item.get("source") in {"bootstrap", "cache_replay"}:
            continue
        if item.get("result") != "passed":
            continue
        business_actions += 1
    return business_actions == 0


def _should_consider_local_module(
    history: list[dict[str, Any]],
    dom_context: dict[str, Any],
) -> bool:
    if not _should_include_project_context(history):
        return False
    if not isinstance(dom_context, dict):
        return False
    return bool(dom_context.get("forms") or dom_context.get("interactive_elements"))


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


def _module_goal_score(
    *,
    module_name: str,
    module_steps: list[dict[str, Any]],
    spec: dict[str, Any],
    dom_context: dict[str, Any],
) -> int:
    goal_terms = _meaningful_goal_terms(
        " ".join(
            _flatten_agent_text(
                [
                    spec.get("description"),
                    spec.get("intent"),
                    spec.get("steps"),
                    spec.get("inputs"),
                    _dom_form_text(dom_context),
                ]
            )
        )
    )
    if not goal_terms:
        return 0
    module_blob = _normalized_goal_text(
        " ".join(_flatten_agent_text([module_name, module_steps]))
    )
    if not module_blob:
        return 0
    score = _semantic_overlap_score(goal_terms, module_blob)
    if any(str(step.get("action") or "").lower() == "fill" for step in module_steps):
        score += 10
    return score


def _dom_form_text(dom_context: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in (dom_context.get("forms") or []) + (
        dom_context.get("interactive_elements") or []
    ):
        if not isinstance(item, dict):
            continue
        for key in ("name", "label", "placeholder", "text", "type", "input_type"):
            if item.get(key):
                parts.append(str(item[key]))
    return " ".join(parts)


def _is_model_timeout_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "timeout" in text or "timed out" in text or "超时" in text


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
        response_model=AgentCaseDecision,
        schema_name=schema_name,
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
    if "action" not in data and isinstance(data.get("decision"), str):
        decision_action = str(data.get("decision") or "").strip()
        if decision_action in set(spec.get("allowed_actions") or set()) | {
            "done",
            "finish",
            "fail",
        }:
            data = {**data, "action": decision_action}
            logger.warning("Agent模型将action误写为decision，已按合法动作字段归一化")
    normalized = {
        key: value for key, value in data.items() if key in _AGENT_DECISION_FIELDS
    }
    extra_fields = sorted(set(data) - _AGENT_DECISION_FIELDS)
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
    if normalized.get("action") == "wait" and normalized.get("wait_ms") is None:
        normalized["wait_ms"] = 1000
        logger.warning("Agent模型wait动作缺少wait_ms，已使用默认1000ms")
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


def _selector_for_guarded_click(
    *,
    page: Any,
    dom_context: dict[str, Any],
    spec: dict[str, Any],
    decision: AgentCaseDecision,
    current_selector: str | None,
) -> str | None:
    if decision.action != "click" or not isinstance(spec, dict):
        return None
    decision_payload = decision.model_dump(exclude_none=True)
    element = _dom_element_for_decision_payload(decision_payload, dom_context)
    selected_click_blob = _normalized_goal_text(
        _click_target_blob(
            decision=decision,
            element=element,
            current_selector=current_selector,
            include_decision_text=False,
        )
    )
    click_blob = _normalized_goal_text(
        _click_target_blob(
            decision=decision,
            element=element,
            current_selector=current_selector,
            include_decision_text=True,
        )
    )
    if not selected_click_blob and not click_blob:
        return None
    goal_blob = _normalized_goal_text(_agent_spec_text(spec))
    if not goal_blob:
        return None
    menu_open = _menu_is_open(page, dom_context)
    decision_intent_blob = _normalized_goal_text(
        " ".join(
            str(value or "")
            for value in (
                decision.target,
                decision.reason,
                decision.expected,
                decision.value,
            )
        )
    )

    if menu_open and _is_menu_open_click(selected_click_blob):
        close_selector = _semantic_selector_for_terms(
            page=page,
            dom_context=dom_context,
            terms=_MENU_CLOSE_TERMS,
            skip_selector=current_selector,
        )
        if close_selector:
            return close_selector
        raise AgentDecisionRejected("菜单已打开，重复点击菜单入口会遮挡当前目标")

    if _blob_has_any(selected_click_blob, _SESSION_EXIT_TERMS) and not _blob_has_any(
        goal_blob, _SESSION_EXIT_TERMS
    ):
        close_selector = (
            _semantic_selector_for_terms(
                page=page,
                dom_context=dom_context,
                terms=_MENU_CLOSE_TERMS,
                skip_selector=current_selector,
            )
            if menu_open
            else None
        )
        if close_selector:
            return close_selector
        progress_selector = _semantic_progress_selector(
            dom_context=dom_context,
            goal_blob=goal_blob,
            decision_blob=decision_intent_blob,
            skip_selector=current_selector,
        )
        if progress_selector:
            return progress_selector
        raise AgentDecisionRejected("点击退出登录会重置当前目标")

    if _blob_has_any(
        selected_click_blob, _DESTRUCTIVE_CLICK_TERMS
    ) and not _blob_has_any(goal_blob, _DESTRUCTIVE_CLICK_TERMS):
        progress_selector = _semantic_progress_selector(
            dom_context=dom_context,
            goal_blob=goal_blob,
            decision_blob=decision_intent_blob,
            skip_selector=current_selector,
        )
        if progress_selector:
            return progress_selector
        raise AgentDecisionRejected("点击删除/重置/清空类动作会破坏当前目标")

    if _is_menu_open_click(selected_click_blob) and not _goal_allows_menu(goal_blob):
        progress_selector = _semantic_progress_selector(
            dom_context=dom_context,
            goal_blob=goal_blob,
            decision_blob=decision_intent_blob,
            skip_selector=current_selector,
        )
        if progress_selector:
            return progress_selector
        raise AgentDecisionRejected("菜单入口与当前目标不匹配")
    return None


def _click_target_blob(
    *,
    decision: AgentCaseDecision,
    element: dict[str, Any] | None,
    current_selector: str | None,
    include_decision_text: bool = True,
) -> str:
    parts: list[str] = [
        str(current_selector or ""),
        str(decision.element_id or ""),
        str(decision.selector or ""),
        str(decision.target or ""),
    ]
    if include_decision_text:
        parts.extend(
            [
                str(decision.reason or ""),
                str(decision.expected or ""),
            ]
        )
    if isinstance(element, dict):
        parts.append(_element_primary_blob(element))
    return " ".join(part for part in parts if part)


def _agent_spec_text(spec: dict[str, Any]) -> str:
    return " ".join(
        _flatten_agent_text(
            [
                spec.get("description"),
                spec.get("intent"),
                spec.get("steps"),
                spec.get("criteria"),
                spec.get("inputs"),
            ]
        )
    )


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


def _is_menu_open_click(click_blob: str) -> bool:
    return _blob_has_any(click_blob, _MENU_CLICK_TERMS)


def _goal_allows_menu(goal_blob: str) -> bool:
    return _blob_has_any(
        goal_blob,
        _MENU_CLICK_TERMS
        + ("left menu", "sidebar", "side panel", "侧边菜单", "左侧菜单"),
    )


def _semantic_progress_selector(
    *,
    dom_context: dict[str, Any],
    goal_blob: str,
    decision_blob: str,
    skip_selector: str | None,
) -> str | None:
    terms = _meaningful_goal_terms(f"{goal_blob} {decision_blob}")
    if not terms:
        return None
    best: tuple[int, str] | None = None
    for item in _iter_dom_elements(dom_context):
        selector = selector_for_element_id(dom_context, str(item.get("id") or ""))
        if not selector or _selector_matches_element(selector, [skip_selector or ""]):
            continue
        if item.get("visible") is False or item.get("enabled") is False:
            continue
        item_blob = _normalized_goal_text(_element_semantic_blob(item))
        if not item_blob:
            continue
        if _blob_has_any(item_blob, _SESSION_EXIT_TERMS + _DESTRUCTIVE_CLICK_TERMS):
            continue
        if _is_menu_open_click(item_blob) and not _goal_allows_menu(goal_blob):
            continue
        score = _semantic_overlap_score(terms, item_blob)
        if score < 20:
            continue
        role = str(item.get("role") or "").lower()
        tag = str(item.get("tag") or "").lower()
        if role in {"button", "link", "menuitem", "tab"} or tag in {"a", "button"}:
            score += 5
        if best is None or score > best[0]:
            best = (score, selector)
    return best[1] if best else None


def _menu_is_open(page: Any, dom_context: dict[str, Any]) -> bool:
    return (
        _semantic_selector_for_terms(
            page=page,
            dom_context=dom_context,
            terms=_MENU_CLOSE_TERMS,
            skip_selector=None,
        )
        is not None
    )


def _semantic_selector_for_terms(
    *,
    page: Any,
    dom_context: dict[str, Any],
    terms: tuple[str, ...],
    skip_selector: str | None,
) -> str | None:
    normalized_terms = [_normalized_goal_text(term) for term in terms]
    for item in _iter_dom_elements(dom_context):
        selector = selector_for_element_id(dom_context, str(item.get("id") or ""))
        if not selector or _selector_matches_element(selector, [skip_selector or ""]):
            continue
        if item.get("visible") is False or item.get("enabled") is False:
            continue
        item_blob = _normalized_goal_text(_element_primary_blob(item))
        if not any(term and term in item_blob for term in normalized_terms):
            continue
        if hasattr(page, "locator") and not _selector_is_visible_enabled(
            page, selector
        ):
            continue
        return selector
    return None


def _element_semantic_blob(item: dict[str, Any]) -> str:
    parts: list[str] = [_element_primary_blob(item)]
    parts.append(str(item.get("near_text") or ""))
    return " ".join(part for part in parts if part)


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


def _meaningful_goal_terms(text: str) -> list[str]:
    normalized = _normalized_goal_text(text)
    terms: list[str] = []
    for token in normalized.split():
        if len(token) < 3 and not re.search(r"[\u4e00-\u9fff]", token):
            continue
        if token in _GENERIC_GOAL_STOPWORDS:
            continue
        terms.append(token)
    return _dedupe_strings(terms)[:24]


def _semantic_overlap_score(terms: list[str], item_blob: str) -> int:
    score = 0
    for term in terms:
        if not term:
            continue
        if term == item_blob:
            score += 80
        elif term in item_blob:
            score += max(20, min(len(term), 20))
        elif item_blob in term and len(item_blob) >= 3:
            score += 12
    return score


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


def _unmet_completion_criteria(
    *,
    criteria: Any,
    history: list[dict[str, Any]],
    current_url: str,
    dom_context: dict[str, Any],
) -> list[str]:
    if _has_final_criteria(criteria):
        return _unmet_final_criteria(
            criteria=criteria,
            history=history,
            current_url=current_url,
            dom_context=dom_context,
        )
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


def _page_title_contains(dom_context: dict[str, Any], needle: str) -> bool:
    meta = dom_context.get("meta") if isinstance(dom_context, dict) else {}
    title = (
        str((meta or {}).get("title") or "").lower() if isinstance(meta, dict) else ""
    )
    return bool(needle) and str(needle).lower() in title


def _history_assertion_has(
    history: list[dict[str, Any]],
    *,
    value_contains: str,
) -> bool:
    token = str(value_contains or "").lower()
    if not token:
        return False
    for item in history:
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


def _criterion_evidence_terms(criterion: str) -> list[str]:
    text = str(criterion or "").strip()
    terms: list[str] = []
    terms.extend(re.findall(r"[\"'“”‘’]([^\"'“”‘’]{2,80})[\"'“”‘’]", text))
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
        target = (
            step.get("target") or step.get("selector") or step.get("use_module") or ""
        )
        action = step.get("action") or ("use_module" if step.get("use_module") else "")
        recent_actions.append(
            f"{action} {target}".strip() + f" -> {item.get('result', 'passed')}"
        )
    checkpoints_count = 0
    final_count = 0
    if isinstance(criteria, dict):
        checkpoints_count = len(_normalize_text_list(criteria.get("checkpoints")))
        final_count = len(_normalize_text_list(criteria.get("final")))
    return {
        "steps_done": len(history),
        "recent_actions": recent_actions,
        "criteria_counts": {
            "checkpoints": checkpoints_count,
            "final": final_count,
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
