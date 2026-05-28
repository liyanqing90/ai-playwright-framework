from __future__ import annotations

import os
import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.ai_runtime.config import load_ai_config
from src.ai_runtime.contracts import AiStepDecision, SelectorDecision
from src.ai_runtime.native_observe import NativeObserveSettings
from src.ai_runtime.payload_compactor import (
    build_dom_context,
    build_locator_context,
    looks_like_internal_element_id,
    normalize_model_text,
    selector_for_element_id,
    selectors_for_element_id,
)
from src.ai_runtime.playwright_selectors import (
    collect_candidates,
    heuristic_selectors,
    normalize_selector,
    selector_matches_target,
    semantic_selectors,
    stable_selector_for_locator,
    validate_selector,
    verify_selector,
)
from src.ai_runtime.provider import ChatCompletionProvider
from src.ai_runtime.selector_registry import SelectorRegistry
from src.ai_runtime.vision_client import (
    VisionConfigurationError,
    VisionServiceUnavailable,
    load_vision_settings,
)
from src.ai_runtime.vision_resolver import VisionResolver
from utils.logger import logger


@dataclass(frozen=True)
class ResolvedSelector:
    selector: str | None
    source: str
    healed: bool = False
    healing_attempted: bool = False
    original_selector: str | None = None
    original_error: str | None = None
    ai_called: bool = False
    confidence: float | None = None
    prompt_version: str | None = None
    schema_version: str | None = None
    model: str | None = None
    candidate_count: int | None = None
    candidate_hash: str | None = None
    coordinate: tuple[float, float] | None = None
    vision_method: str | None = None
    vision_reason: str | None = None


@dataclass(frozen=True)
class AiStepOperation:
    action: str
    selector: str | None = None
    value: str | None = None
    key: str | None = None
    wait_ms: int | None = None
    reason: str | None = None
    source: str = "ai_step"
    prompt_version: str | None = None
    schema_version: str | None = None
    model: str | None = None
    candidate_count: int | None = None
    candidate_hash: str | None = None


class SmartResolver:
    def __init__(self, page, *, project: str | None = None, env: str | None = None):
        self.page = page
        self.project = project or os.environ.get("TEST_PROJECT", "default")
        self.env = env or os.environ.get("TEST_ENV", "local")
        self.config = load_ai_config()
        self.native_observe = NativeObserveSettings.from_config(self.config)
        registry_cfg = self.config.get("selector_registry", {})
        registry_disabled = str(
            os.environ.get("UI_AI_DISABLE_SELECTOR_REGISTRY") or ""
        ).lower() in {"1", "true", "yes", "on"}
        self.registry_enabled = (
            bool(registry_cfg.get("enabled", True)) and not registry_disabled
        )
        self.unstable_threshold = int(registry_cfg.get("unstable_threshold", 3))
        self.deprecated_after_failures = int(
            registry_cfg.get(
                "deprecated_after_failures",
                max(self.unstable_threshold + 1, 3),
            )
        )
        self.registry_min_score = float(registry_cfg.get("min_score_to_use", 0.0))
        self.registry = None
        if self.registry_enabled:
            sqlite_path = Path(registry_cfg.get("sqlite_path", ".ui_auto/selectors.db"))
            self.registry = SelectorRegistry(sqlite_path)
        runtime_cfg = self.config.get("runtime", {})
        self.allow_ai_in_smart = bool(runtime_cfg.get("allow_ai_in_smart", True))
        self.ai_enabled = bool(runtime_cfg.get("ai_enabled", True))
        self.max_ai_calls = int(runtime_cfg.get("max_ai_calls_per_test", 3))
        self.candidate_limit = int(runtime_cfg.get("candidate_limit", 120))
        self.ai_step_candidate_limit = int(
            runtime_cfg.get(
                "ai_step_candidate_limit",
                min(self.candidate_limit, 40),
            )
        )
        self.llm_selector_candidate_scan_limit = int(
            runtime_cfg.get("llm_selector_candidate_scan_limit", self.candidate_limit)
        )
        self.llm_selector_candidate_limit = int(
            runtime_cfg.get(
                "llm_selector_candidate_limit",
                min(self.llm_selector_candidate_scan_limit, 40),
            )
        )
        self.ai_call_count = 0
        self.vision_settings = load_vision_settings(self.config)
        self.vision_call_count = 0
        prompts_cfg = self.config.get("prompts", {})
        llm_cfg = self.config.get("llm", {})
        self.selector_prompt_version = str(
            prompts_cfg.get("selector_version", "selector-v1")
        )
        self.ai_step_prompt_version = str(
            prompts_cfg.get("ai_step_version")
            or prompts_cfg.get("observe_version")
            or "ai-step-v1"
        )
        self.vision_prompt_version = str(prompts_cfg.get("vision_version", "vision-v1"))
        self.schema_version = str(llm_cfg.get("schema_version", "ui-ai-schema-v1"))

    def resolve(
        self,
        *,
        action: str,
        target: str | None,
        selector: str | None,
        mode: str,
        timeout: int,
    ) -> ResolvedSelector:
        mode = str(mode or "strict").lower()
        normalized_selector = normalize_selector(selector) if selector else None
        target_text = target or selector
        page_key = self._page_key()
        healing_attempted = False
        original_error: str | None = None

        if not normalized_selector and looks_like_internal_element_id(target_text):
            raise ValueError(
                f"未解析的内部element_id不能直接用于语义定位: {target_text}"
            )

        if normalized_selector:
            try:
                verify_selector(
                    self.page, normalized_selector, action=action, timeout=timeout
                )
                if target_text and not selector_matches_target(
                    self.page, normalized_selector, target_text, action
                ):
                    raise ValueError(
                        f"explicit selector semantic mismatch: target={target_text}, selector={normalized_selector}"
                    )
                return ResolvedSelector(
                    selector=normalized_selector,
                    source="explicit",
                    original_selector=normalized_selector,
                )
            except Exception as exc:
                if mode == "strict" or not target_text:
                    raise
                healing_attempted = True
                original_error = str(exc)
                self._log_heal_start(
                    action=action,
                    selector=normalized_selector,
                    target=target_text,
                    error=original_error,
                )

        if mode == "strict":
            raise ValueError(f"strict模式需要可用selector: {action}")
        if not target_text:
            raise ValueError(f"{mode}模式需要target或selector语义描述: {action}")

        if self.registry:
            record = self.registry.find(
                project=self.project,
                env=self.env,
                page_key=page_key,
                action=action,
                target=target_text,
                min_score=self.registry_min_score,
            )
            if record:
                try:
                    verify_selector(
                        self.page, record.selector, action=action, timeout=timeout
                    )
                    if not selector_matches_target(
                        self.page, record.selector, target_text, action
                    ):
                        self.registry.deprecate(
                            record.id,
                            last_error="registry selector semantic mismatch",
                        )
                        logger.warning(
                            f"历史定位语义不匹配，已废弃: target={target_text} selector={record.selector}"
                        )
                        raise ValueError("registry selector semantic mismatch")
                    self.registry.mark_success(record.id)
                    self._log_heal_success(
                        source="registry",
                        selector=record.selector,
                        confidence=record.confidence,
                        healing_attempted=healing_attempted,
                    )
                    return self._with_healing_context(
                        ResolvedSelector(
                            selector=record.selector,
                            source="registry",
                            confidence=record.confidence,
                        ),
                        healing_attempted=healing_attempted,
                        original_selector=normalized_selector,
                        original_error=original_error,
                    )
                except Exception as exc:
                    if str(exc) != "registry selector semantic mismatch":
                        self.registry.mark_failed(
                            record.id,
                            unstable_threshold=self.unstable_threshold,
                            deprecated_after_failures=self.deprecated_after_failures,
                            last_error="registry selector verification failed",
                        )
                    self._log_heal_rejected(
                        source="registry",
                        selector=record.selector,
                        reason=str(exc),
                        healing_attempted=healing_attempted,
                    )

        selector_candidates = (
            semantic_selectors(
                self.page,
                target_text,
                action,
                limit=self.candidate_limit,
                ignore_selectors=self.native_observe.ignore_selectors,
                include_open_shadow_dom=self.native_observe.include_open_shadow_dom,
            )
            if self.native_observe.enabled
            else []
        )
        selector_candidates.extend(heuristic_selectors(target_text, action))
        for candidate in _dedupe(selector_candidates):
            try:
                verify_selector(
                    self.page, candidate, action=action, timeout=min(timeout, 1000)
                )
                stable = stable_selector_for_locator(self.page.locator(candidate))
                if not selector_matches_target(self.page, stable, target_text, action):
                    logger.warning(
                        f"DOM语义匹配不一致，跳过: target={target_text} selector={stable}"
                    )
                    continue
                self._save_selector(
                    action=action,
                    target=target_text,
                    selector=stable,
                    source="heuristic",
                    confidence=0.8,
                )
                self._log_heal_success(
                    source="heuristic",
                    selector=stable,
                    confidence=0.8,
                    healing_attempted=healing_attempted,
                )
                return self._with_healing_context(
                    ResolvedSelector(
                        selector=stable,
                        source="heuristic",
                        confidence=0.8,
                    ),
                    healing_attempted=healing_attempted,
                    original_selector=normalized_selector,
                    original_error=original_error,
                )
            except Exception as exc:
                self._log_heal_rejected(
                    source="heuristic",
                    selector=candidate,
                    reason=str(exc),
                    healing_attempted=healing_attempted,
                )
                continue

        if mode == "smart" and not self.allow_ai_in_smart:
            message = f"smart模式未解析到元素，且配置禁止AI兜底: {target_text}"
            self._log_heal_failed(
                action=action,
                target=target_text,
                selector=normalized_selector,
                errors=[message],
                healing_attempted=healing_attempted,
            )
            raise ValueError(message)
        if not self.ai_enabled:
            message = f"AI定位未启用，无法解析目标: {target_text}"
            self._log_heal_failed(
                action=action,
                target=target_text,
                selector=normalized_selector,
                errors=[message],
                healing_attempted=healing_attempted,
            )
            raise ValueError(message)

        errors: list[str] = []
        resolved: ResolvedSelector | None = None
        if self.vision_settings.enabled:
            try:
                resolved = self._resolve_with_vision(
                    action=action, target=target_text, timeout=timeout
                )
            except (VisionConfigurationError, VisionServiceUnavailable) as vision_exc:
                logger.warning(f"UI Vision服务不可用，继续LLM兜底: {vision_exc}")
                errors.append(f"vision={vision_exc}")
            except Exception as vision_exc:
                errors.append(f"vision={vision_exc}")

        if resolved is None:
            try:
                resolved = self._resolve_with_ai(
                    action=action, target=target_text, timeout=timeout
                )
            except Exception as exc:
                errors.append(f"llm={exc}")
                self._log_heal_failed(
                    action=action,
                    target=target_text,
                    selector=normalized_selector,
                    errors=errors,
                    healing_attempted=healing_attempted,
                )
                raise

        if resolved.selector:
            if not selector_matches_target(
                self.page, resolved.selector, target_text, action
            ):
                raise ValueError(
                    f"智能定位结果与目标语义不匹配: target={target_text}, selector={resolved.selector}"
                )
            self._save_selector(
                action=action,
                target=target_text,
                selector=resolved.selector,
                source=resolved.source,
                confidence=resolved.confidence or 0.6,
                prompt_version=resolved.prompt_version,
                schema_version=resolved.schema_version,
                model=resolved.model,
                candidate_hash=resolved.candidate_hash,
                candidate_count=resolved.candidate_count,
                replace_active=True,
            )
            self._log_heal_success(
                source=resolved.source,
                selector=resolved.selector,
                confidence=resolved.confidence,
                healing_attempted=healing_attempted,
            )
        elif not resolved.coordinate:
            detail = "; ".join(errors) if errors else target_text
            self._log_heal_failed(
                action=action,
                target=target_text,
                selector=normalized_selector,
                errors=[detail],
                healing_attempted=healing_attempted,
            )
            raise ValueError(f"智能定位未返回可执行目标: {detail}")
        return self._with_healing_context(
            resolved,
            healing_attempted=healing_attempted,
            original_selector=normalized_selector,
            original_error=original_error,
        )

    def resolve_ai_step(self, *, instruction: str, timeout: int) -> AiStepOperation:
        if not self.ai_enabled:
            raise ValueError("AI步骤未启用")
        fast_operation = self._resolve_ai_step_fast(
            instruction=instruction,
            timeout=timeout,
        )
        if fast_operation is not None:
            return fast_operation
        self._claim_ai_call("ai_step")
        candidates = self._collect_candidates(limit=self.ai_step_candidate_limit)
        provider = ChatCompletionProvider()
        decision = provider.complete_model(
            [
                {
                    "role": "system",
                    "content": (
                        "你是UI自动化原生AI步骤编译器。根据用户指令从候选元素中选择一个框架支持的原子动作，"
                        "只返回JSON对象，不要解释。action只允许click/fill/press/wait/skip/reject。"
                        "必须返回候选元素中的element_id，不要自己生成selector。"
                        "selector字段仅作为兼容字段；如填写，只能填写候选element_id。"
                        "ai_step必须是单一UI动作或单一断言意图，只能编译为一个标准step。"
                        "如果指令包含两个或更多动作/断言，或描述需要多步完成的端到端流程，必须返回reject并说明reason。"
                        "reject用于提示用户拆成多个steps或改用agent_case，不要勉强选择其中一个动作。"
                        "click/fill/press必须返回候选元素中的element_id；fill返回value；press返回key；"
                        "wait返回wait_ms。不要直接执行浏览器操作。"
                    ),
                },
                {
                    "role": "user",
                    "content": self._json_payload(
                        {
                            "url": self.page.url,
                            "prompt_version": self.ai_step_prompt_version,
                            "schema_version": self.schema_version,
                            "instruction": instruction,
                            "dom_context": build_dom_context(
                                candidates,
                                url=self.page.url,
                                title=_safe_page_title(self.page),
                                context_level=1,
                                limit=self.ai_step_candidate_limit,
                                hints=[instruction],
                            ),
                        }
                    ),
                },
            ],
            AiStepDecision,
            schema_name="AiStepDecision",
            usage_operation="runtime.ai_step",
            usage_metadata={
                "schema_name": "AiStepDecision",
                "prompt_version": self.ai_step_prompt_version,
                "page_url": self.page.url,
            },
        )
        if decision.status != "ok":
            return AiStepOperation(
                action="reject",
                reason=normalize_model_text(
                    decision.reason or f"AI步骤返回状态: {decision.status}",
                ),
                prompt_version=self.ai_step_prompt_version,
                schema_version=self.schema_version,
                model=provider.settings.model,
                candidate_count=len(candidates),
                candidate_hash=self._candidate_hash(candidates),
            )
        if decision.action == "reject":
            return AiStepOperation(
                action="reject",
                reason=normalize_model_text(
                    decision.reason or "ai_step不是单一原子动作"
                ),
                prompt_version=self.ai_step_prompt_version,
                schema_version=self.schema_version,
                model=provider.settings.model,
                candidate_count=len(candidates),
                candidate_hash=self._candidate_hash(candidates),
            )
        if decision.action in {"skip", "wait"}:
            return AiStepOperation(
                action=decision.action,
                wait_ms=decision.wait_ms or 1000,
                reason=decision.reason,
                prompt_version=self.ai_step_prompt_version,
                schema_version=self.schema_version,
                model=provider.settings.model,
                candidate_count=len(candidates),
                candidate_hash=self._candidate_hash(candidates),
            )
        dom_context = build_dom_context(
            candidates,
            url=self.page.url,
            title=_safe_page_title(self.page),
            context_level=1,
            limit=self.ai_step_candidate_limit,
            hints=[instruction],
        )
        verify_action = "fill" if decision.action == "fill" else "click"
        selected_element_id = self._selected_element_id_from_decision(
            dom_context,
            element_id=decision.element_id,
            selector_ref=decision.selector,
            source="AI步骤",
        )
        selector = self._verified_selector_for_element_id(
            dom_context,
            selected_element_id,
            action=verify_action,
            timeout=timeout,
        )
        return AiStepOperation(
            action=decision.action,
            selector=selector,
            value=decision.value,
            key=decision.key,
            reason=normalize_model_text(decision.reason),
            prompt_version=self.ai_step_prompt_version,
            schema_version=self.schema_version,
            model=provider.settings.model,
            candidate_count=len(candidates),
            candidate_hash=self._candidate_hash(candidates),
        )

    # Backward-compatible method name. The execution layer now compiles this
    # result back into the normal command pipeline instead of applying it here.
    def observe_operation(self, *, instruction: str, timeout: int) -> AiStepOperation:
        return self.resolve_ai_step(instruction=instruction, timeout=timeout)

    def _resolve_ai_step_fast(
        self,
        *,
        instruction: str,
        timeout: int,
    ) -> AiStepOperation | None:
        normalized = " ".join(str(instruction or "").split())
        lowered = normalized.lower()
        if not normalized:
            return None
        if _looks_like_multi_step_instruction(lowered):
            return AiStepOperation(
                action="reject",
                reason="ai_step 只能包含一个原子动作",
                source="ai_step_fast",
                prompt_version=self.ai_step_prompt_version,
                schema_version=self.schema_version,
                model="heuristic-local",
            )

        wait_ms = _parse_wait_ms(normalized)
        if wait_ms is not None:
            return AiStepOperation(
                action="wait",
                wait_ms=wait_ms,
                reason="等待指定时长",
                source="ai_step_fast",
                prompt_version=self.ai_step_prompt_version,
                schema_version=self.schema_version,
                model="heuristic-local",
            )

        fill_target, fill_value = _parse_fill_instruction(normalized)
        if fill_target and fill_value is not None:
            operation = self._build_fast_ai_step_operation(
                action="fill",
                target=fill_target,
                timeout=timeout,
                value=fill_value,
            )
            if operation is not None:
                return operation

        click_target = _parse_click_target(normalized)
        if click_target:
            operation = self._build_fast_ai_step_operation(
                action="click",
                target=click_target,
                timeout=timeout,
            )
            if operation is not None:
                return operation
        return None

    def _build_fast_ai_step_operation(
        self,
        *,
        action: str,
        target: str,
        timeout: int,
        value: str | None = None,
    ) -> AiStepOperation | None:
        for candidate_target in _instruction_target_variants(target):
            semantic_candidates = (
                semantic_selectors(
                    self.page,
                    candidate_target,
                    action,
                    limit=min(self.ai_step_candidate_limit, 8),
                    ignore_selectors=self.native_observe.ignore_selectors,
                    include_open_shadow_dom=self.native_observe.include_open_shadow_dom,
                )
                if self.native_observe.enabled
                else []
            )
            selectors = _dedupe(
                semantic_candidates + heuristic_selectors(candidate_target, action)
            )
            if not selectors:
                continue
            for selector in selectors[:3]:
                normalized_selector = normalize_selector(selector)
                if not normalized_selector:
                    continue
                try:
                    verify_selector(
                        self.page,
                        normalized_selector,
                        action="fill" if action == "fill" else "click",
                        timeout=timeout,
                    )
                    if not selector_matches_target(
                        self.page,
                        normalized_selector,
                        candidate_target,
                        action,
                    ):
                        continue
                    return AiStepOperation(
                        action=action,
                        selector=normalized_selector,
                        value=value,
                        reason=normalize_model_text(f"{action} {target}"),
                        source="ai_step_fast",
                        prompt_version=self.ai_step_prompt_version,
                        schema_version=self.schema_version,
                        model="heuristic-local",
                        candidate_count=len(selectors),
                        candidate_hash=self._candidate_hash(
                            [{"selector": item} for item in selectors]
                        ),
                    )
                except Exception:
                    continue
        return None

    def _resolve_with_ai(
        self, *, action: str, target: str, timeout: int
    ) -> ResolvedSelector:
        self._claim_ai_call("selector")
        candidates = self._collect_candidates(
            limit=self.llm_selector_candidate_scan_limit,
        )
        locator_context = build_locator_context(
            action=action,
            target=target,
            candidates=candidates,
            url=self.page.url,
            title=_safe_page_title(self.page),
            limit=self.llm_selector_candidate_limit,
        )
        provider = ChatCompletionProvider()
        decision = provider.complete_model(
            [
                {
                    "role": "system",
                    "content": (
                        "你是UI自动化selector解析器。根据候选元素为目标选择最稳定的selector。"
                        "只负责当前step的元素选择，不做业务决策。"
                        "必须返回候选元素中的element_id或selected_element_id，不要自己生成selector。"
                        "selector字段仅作为兼容字段；如填写，只能填写候选element_id。"
                        "reason不超过80个中文字符。只返回JSON对象。"
                    ),
                },
                {
                    "role": "user",
                    "content": self._json_payload(
                        {
                            "prompt_version": self.selector_prompt_version,
                            "schema_version": self.schema_version,
                            "smart_locator_context": locator_context,
                        }
                    ),
                },
            ],
            SelectorDecision,
            schema_name="SelectorDecision",
            usage_operation="runtime.resolve_selector",
            usage_metadata={
                "schema_name": "SelectorDecision",
                "prompt_version": self.selector_prompt_version,
                "page_url": self.page.url,
                "action": action,
            },
        )
        if decision.status != "ok":
            raise ValueError(
                f"AI定位未返回可执行元素: status={decision.status} reason={decision.reason}"
            )
        selected_element_id = self._selected_element_id_from_decision(
            locator_context,
            element_id=decision.element_id or decision.selected_element_id,
            selector_ref=decision.selector,
            source="AI定位",
        )
        selector = self._verified_selector_for_element_id(
            locator_context,
            selected_element_id,
            action=action,
            timeout=timeout,
        )
        if not selector_matches_target(self.page, selector, target, action):
            raise ValueError(
                f"AI selector semantic mismatch: target={target}, selector={selector}"
            )
        return ResolvedSelector(
            selector=selector,
            source="ai_selector",
            healed=True,
            ai_called=True,
            confidence=decision.confidence,
            prompt_version=self.selector_prompt_version,
            schema_version=self.schema_version,
            model=provider.settings.model,
            candidate_count=len(candidates),
            candidate_hash=self._candidate_hash(candidates),
        )

    def _selected_element_id_from_decision(
        self,
        payload: dict[str, Any],
        *,
        element_id: str | None,
        selector_ref: str | None,
        source: str,
    ) -> str:
        if element_id:
            if selectors_for_element_id(payload, element_id):
                return element_id
            raise ValueError(f"{source}返回未知element_id: {element_id}")
        if selector_ref and selectors_for_element_id(payload, selector_ref):
            return selector_ref
        raise ValueError(f"{source}必须返回候选element_id，不能直接返回selector")

    def _verified_selector_for_element_id(
        self,
        payload: dict[str, Any],
        element_id: str,
        *,
        action: str,
        timeout: int,
    ) -> str:
        errors: list[str] = []
        for raw_selector in selectors_for_element_id(payload, element_id):
            selector = normalize_selector(raw_selector)
            validation = validate_selector(
                self.page,
                selector,
                action=action,
                timeout=timeout,
                require_unique=True,
            )
            if validation.ok:
                return selector
            errors.append(f"{selector}: {validation.error}")
        detail = "; ".join(errors) if errors else "no selector candidates"
        raise ValueError(f"候选element_id不可执行: {element_id} | {detail}")

    def _resolve_with_vision(
        self, *, action: str, target: str, timeout: int
    ) -> ResolvedSelector:
        self._claim_vision_call()
        candidates = self._collect_candidates(limit=self.candidate_limit)
        candidate_hash = self._candidate_hash(candidates)
        resolution = VisionResolver(
            self.page,
            settings=self.vision_settings,
        ).resolve(
            action=action,
            target=target,
            timeout=timeout,
            candidates=candidates,
        )
        return ResolvedSelector(
            selector=resolution.selector,
            source=resolution.source,
            healed=True,
            ai_called=True,
            confidence=resolution.confidence,
            prompt_version=self.vision_prompt_version,
            schema_version=self.schema_version,
            model=resolution.method,
            candidate_count=len(candidates),
            candidate_hash=candidate_hash,
            coordinate=resolution.coordinate,
            vision_method=resolution.method,
            vision_reason=resolution.reason,
        )

    def _save_selector(
        self,
        *,
        action: str,
        target: str,
        selector: str,
        source: str,
        confidence: float,
        prompt_version: str | None = None,
        schema_version: str | None = None,
        model: str | None = None,
        candidate_hash: str | None = None,
        candidate_count: int | None = None,
        replace_active: bool = False,
    ) -> None:
        if not self.registry:
            return
        if not selector_matches_target(self.page, selector, target, action):
            logger.warning(
                f"跳过保存语义不匹配的定位结果: target={target} selector={selector}"
            )
            return
        self.registry.save(
            project=self.project,
            env=self.env,
            page_key=self._page_key(),
            action=action,
            target=target,
            selector=selector,
            source=source,
            confidence=confidence,
            prompt_version=prompt_version,
            schema_version=schema_version,
            model=model,
            candidate_hash=candidate_hash,
            candidate_count=candidate_count,
            replace_active=replace_active,
        )

    @staticmethod
    def _with_healing_context(
        resolved: ResolvedSelector,
        *,
        healing_attempted: bool,
        original_selector: str | None,
        original_error: str | None,
    ) -> ResolvedSelector:
        return replace(
            resolved,
            healed=resolved.healed or healing_attempted,
            healing_attempted=healing_attempted,
            original_selector=original_selector,
            original_error=original_error,
        )

    @staticmethod
    def _log_heal_start(
        *, action: str, selector: str, target: str | None, error: str
    ) -> None:
        logger.warning(
            "selector自愈开始: "
            f"action={action} | selector={selector} | target={target}"
        )
        logger.warning(f"原selector失败: {error}")

    @staticmethod
    def _log_heal_success(
        *,
        source: str,
        selector: str,
        confidence: float | None,
        healing_attempted: bool,
    ) -> None:
        if not healing_attempted:
            return
        parts = [f"source={source}", f"selector={selector}"]
        if confidence is not None:
            parts.append(f"confidence={confidence}")
        logger.info("自愈验证通过: " + " | ".join(parts))

    @staticmethod
    def _log_heal_rejected(
        *,
        source: str,
        selector: str,
        reason: str,
        healing_attempted: bool,
    ) -> None:
        if not healing_attempted:
            return
        logger.debug(
            "自愈候选失败: " f"source={source} | selector={selector} | reason={reason}"
        )

    @staticmethod
    def _log_heal_failed(
        *,
        action: str,
        target: str | None,
        selector: str | None,
        errors: list[str],
        healing_attempted: bool,
    ) -> None:
        if not healing_attempted:
            return
        detail = "; ".join(errors)
        logger.error(
            "selector自愈失败: "
            f"action={action} | selector={selector} | target={target} | errors={detail}"
        )

    def _claim_ai_call(self, purpose: str) -> None:
        if self.ai_call_count >= self.max_ai_calls:
            raise RuntimeError(
                f"AI调用超过单用例预算: purpose={purpose}, max={self.max_ai_calls}"
            )
        self.ai_call_count += 1

    def _claim_vision_call(self) -> None:
        if self.vision_call_count >= self.vision_settings.max_calls_per_test:
            raise RuntimeError(
                "UI Vision调用超过单用例预算: "
                f"max={self.vision_settings.max_calls_per_test}"
            )
        self.vision_call_count += 1

    def _collect_candidates(self, *, limit: int) -> list[dict[str, Any]]:
        if not self.native_observe.enabled:
            return []
        return collect_candidates(
            self.page,
            limit=min(limit, self.native_observe.max_candidates),
            ignore_selectors=self.native_observe.ignore_selectors,
            include_open_shadow_dom=self.native_observe.include_open_shadow_dom,
        )

    @staticmethod
    def _candidate_hash(candidates: list[dict[str, Any]]) -> str:
        raw = json.dumps(candidates, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _page_key(self) -> str:
        url = getattr(self.page, "url", "") or "about:blank"
        return url.split("?")[0]

    @staticmethod
    def _json_payload(data: dict[str, Any]) -> str:
        import json

        return json.dumps(data, ensure_ascii=False)


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _safe_page_title(page: Any) -> str:
    try:
        return str(page.title()).strip()
    except Exception:
        return ""


def _looks_like_multi_step_instruction(instruction: str) -> bool:
    action_markers = [
        "click",
        "open",
        "tap",
        "fill",
        "input",
        "enter",
        "type",
        "press",
        "wait",
        "goto",
        "navigate",
        "点击",
        "打开",
        "输入",
        "填写",
        "按下",
        "等待",
        "访问",
    ]
    hits = sum(1 for marker in action_markers if marker in instruction)
    if hits <= 1:
        return False
    return any(
        token in instruction
        for token in (" and ", " then ", "；", ";", "，然后", "并且", "后再")
    )


def _parse_wait_ms(instruction: str) -> int | None:
    match = re.search(
        r"\bwait\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|seconds)?\b",
        instruction,
        re.IGNORECASE,
    )
    if match:
        value = float(match.group(1))
        unit = (match.group(2) or "s").lower()
        return int(value if unit.startswith("ms") else value * 1000)
    match = re.search(r"等待\s*(\d+(?:\.\d+)?)\s*(毫秒|秒)?", instruction)
    if match:
        value = float(match.group(1))
        unit = match.group(2) or "秒"
        return int(value if "毫秒" in unit else value * 1000)
    return None


def _parse_click_target(instruction: str) -> str:
    text = instruction.strip().rstrip(".。")
    patterns = [
        r"^(?:click|open|tap|select)\s+(?:the\s+)?(.+)$",
        r"^(?:点击|打开|选择)\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            return _clean_instruction_target(match.group(1))
    return ""


def _parse_fill_instruction(instruction: str) -> tuple[str, str | None]:
    text = instruction.strip().rstrip(".。")
    chinese_target, chinese_value = _parse_chinese_fill_instruction(text)
    if chinese_target and chinese_value is not None:
        return chinese_target, chinese_value

    patterns = [
        (
            r'^(?:fill|enter|input|type)\s+"([^"]+)"\s+(?:into|in)\s+(.+)$',
            "value_target",
        ),
        (
            r"^(?:fill|enter|input|type)\s+'([^']+)'\s+(?:into|in)\s+(.+)$",
            "value_target",
        ),
        (r"^(?:fill|enter|input|type)\s+(.+?)\s+with\s+(.+)$", "target_value"),
    ]
    for pattern, group_order in patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if not match:
            continue
        left, right = (item.strip() for item in match.groups())
        if group_order == "target_value":
            target, value = left, right
        else:
            value, target = left, right
        return _clean_instruction_target(target), _clean_instruction_value(value)
    return "", None


def _clean_instruction_target(value: str) -> str:
    text = " ".join(str(value or "").split()).strip(" .。,:：")
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.IGNORECASE)
    return text


def _parse_chinese_fill_instruction(text: str) -> tuple[str, str | None]:
    if not re.search(r"[\u4e00-\u9fff]", text) or "输入" not in text:
        return "", None
    split_at = text.rfind("输入")
    if split_at <= 0:
        return "", None
    target = text[:split_at].strip()
    value = text[split_at + len("输入") :].strip()
    target = re.sub(r"^在", "", target).strip()
    target = re.sub(r"中$", "", target).strip()
    if not target or not value:
        return "", None
    return _clean_instruction_target(target), _clean_instruction_value(value)


def _instruction_target_variants(target: str) -> list[str]:
    text = " ".join(str(target or "").split()).strip()
    if not text:
        return []
    variants = [text]
    simplified = re.sub(
        r"\s+(?:in|on|from|at)\s+(?:the\s+)?(?:top|bottom|left|right|top-right|top left|header|navigation|nav).*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    if simplified and simplified not in variants:
        variants.append(simplified)
    semantic = simplified or text
    semantic = re.sub(
        r"(?:top|bottom|left|right|header|navigation|nav)",
        "",
        semantic,
        flags=re.IGNORECASE,
    )
    semantic = " ".join(semantic.split()).strip()
    if semantic and semantic not in variants:
        variants.append(semantic)
    return variants


def _clean_instruction_value(value: str) -> str:
    return " ".join(str(value or "").split()).strip(" .。,:：'\"")
