from __future__ import annotations

import os
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.ai_runtime.config import load_ai_config
from src.ai_runtime.contracts import AiStepDecision, SelectorDecision
from src.ai_runtime.playwright_selectors import (
    collect_candidates,
    heuristic_selectors,
    normalize_selector,
    selector_matches_target,
    semantic_selectors,
    stable_selector_for_locator,
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
        registry_cfg = self.config.get("selector_registry", {})
        self.registry_enabled = bool(registry_cfg.get("enabled", True))
        self.unstable_threshold = int(registry_cfg.get("unstable_threshold", 3))
        self.registry = None
        if self.registry_enabled:
            sqlite_path = Path(registry_cfg.get("sqlite_path", ".ui_auto/selectors.db"))
            self.registry = SelectorRegistry(sqlite_path)
        runtime_cfg = self.config.get("runtime", {})
        self.allow_ai_in_smart = bool(runtime_cfg.get("allow_ai_in_smart", True))
        self.ai_enabled = bool(runtime_cfg.get("ai_enabled", True))
        self.max_ai_calls = int(runtime_cfg.get("max_ai_calls_per_test", 3))
        self.candidate_limit = int(runtime_cfg.get("candidate_limit", 120))
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
                            last_error="registry selector verification failed",
                        )
                    self._log_heal_rejected(
                        source="registry",
                        selector=record.selector,
                        reason=str(exc),
                        healing_attempted=healing_attempted,
                    )

        selector_candidates = semantic_selectors(
            self.page, target_text, action, limit=self.candidate_limit
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
                        f"规则定位语义不匹配，跳过: target={target_text} selector={stable}"
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
        try:
            resolved = self._resolve_with_ai(
                action=action, target=target_text, timeout=timeout
            )
        except Exception as exc:
            errors.append(f"llm={exc}")
            if not self.vision_settings.enabled:
                self._log_heal_failed(
                    action=action,
                    target=target_text,
                    selector=normalized_selector,
                    errors=errors,
                    healing_attempted=healing_attempted,
                )
                raise
            try:
                resolved = self._resolve_with_vision(
                    action=action, target=target_text, timeout=timeout
                )
            except (VisionConfigurationError, VisionServiceUnavailable) as vision_exc:
                logger.warning(
                    f"UI Vision服务不可用，跳过视觉兜底并保持原有定位失败: {vision_exc}"
                )
                errors.append(f"vision={vision_exc}")
                self._log_heal_failed(
                    action=action,
                    target=target_text,
                    selector=normalized_selector,
                    errors=errors,
                    healing_attempted=healing_attempted,
                )
                raise exc

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
        self._claim_ai_call("ai_step")
        candidates = collect_candidates(self.page, limit=self.candidate_limit)
        provider = ChatCompletionProvider()
        decision = provider.complete_model(
            [
                {
                    "role": "system",
                    "content": (
                        "你是UI自动化原生AI步骤编译器。根据用户指令从候选元素中选择一个框架支持的原子动作，"
                        "只返回JSON对象，不要解释。action只允许click/fill/press/wait/skip。"
                        "click/fill/press必须返回候选元素中的selector；fill返回value；press返回key；"
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
                            "candidates": candidates,
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
        if decision.action in {"skip", "wait"}:
            return AiStepOperation(
                action=decision.action,
                wait_ms=decision.wait_ms or 1000,
                prompt_version=self.ai_step_prompt_version,
                schema_version=self.schema_version,
                model=provider.settings.model,
                candidate_count=len(candidates),
                candidate_hash=self._candidate_hash(candidates),
            )
        selector = normalize_selector(str(decision.selector or ""))
        if not selector:
            raise ValueError("AI步骤未返回selector")
        verify_action = "fill" if decision.action == "fill" else "click"
        verify_selector(self.page, selector, action=verify_action, timeout=timeout)
        return AiStepOperation(
            action=decision.action,
            selector=selector,
            value=decision.value,
            key=decision.key,
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

    def _resolve_with_ai(
        self, *, action: str, target: str, timeout: int
    ) -> ResolvedSelector:
        self._claim_ai_call("selector")
        candidates = collect_candidates(self.page, limit=self.candidate_limit)
        provider = ChatCompletionProvider()
        decision = provider.complete_model(
            [
                {
                    "role": "system",
                    "content": (
                        "你是UI自动化selector解析器。根据候选元素为目标选择最稳定的selector。"
                        "只返回JSON对象，字段: selector, selector_type(css/xpath/text), confidence。"
                    ),
                },
                {
                    "role": "user",
                    "content": self._json_payload(
                        {
                            "url": self.page.url,
                            "prompt_version": self.selector_prompt_version,
                            "schema_version": self.schema_version,
                            "action": action,
                            "target": target,
                            "candidates": candidates,
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
        selector = normalize_selector(
            decision.selector,
            decision.selector_type,
        )
        if not selector:
            raise ValueError(f"AI未返回selector: {target}")
        verify_selector(self.page, selector, action=action, timeout=timeout)
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

    def _resolve_with_vision(
        self, *, action: str, target: str, timeout: int
    ) -> ResolvedSelector:
        self._claim_vision_call()
        candidates = collect_candidates(self.page, limit=self.candidate_limit)
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
