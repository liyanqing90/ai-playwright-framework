from __future__ import annotations

import os
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.ai_runtime.config import load_ai_config
from src.ai_runtime.contracts import ObservedOperationDecision, SelectorDecision
from src.ai_runtime.playwright_selectors import (
    collect_candidates,
    heuristic_selectors,
    normalize_selector,
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
class ObservedOperation:
    action: str
    selector: str | None = None
    value: str | None = None
    key: str | None = None
    wait_ms: int | None = None
    source: str = "ai_observe"
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
        self.observe_prompt_version = str(
            prompts_cfg.get("observe_version", "observe-v1")
        )
        self.vision_prompt_version = str(
            prompts_cfg.get("vision_version", "vision-v1")
        )
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

        if normalized_selector:
            try:
                verify_selector(
                    self.page, normalized_selector, action=action, timeout=timeout
                )
                return ResolvedSelector(selector=normalized_selector, source="explicit")
            except Exception as exc:
                if mode == "strict" or not target_text:
                    raise
                logger.warning(
                    f"显式selector失效，进入智能定位: {normalized_selector} - {exc}"
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
                    self.registry.mark_success(record.id)
                    return ResolvedSelector(selector=record.selector, source="registry")
                except Exception:
                    self.registry.mark_failed(
                        record.id,
                        unstable_threshold=self.unstable_threshold,
                        last_error="registry selector verification failed",
                    )

        for candidate in heuristic_selectors(target_text, action):
            try:
                verify_selector(
                    self.page, candidate, action=action, timeout=min(timeout, 1000)
                )
                stable = stable_selector_for_locator(self.page.locator(candidate))
                self._save_selector(
                    action=action,
                    target=target_text,
                    selector=stable,
                    source="heuristic",
                    confidence=0.8,
                )
                return ResolvedSelector(selector=stable, source="heuristic")
            except Exception:
                continue

        if mode == "smart" and not self.allow_ai_in_smart:
            raise ValueError(f"smart模式未解析到元素，且配置禁止AI兜底: {target_text}")
        if not self.ai_enabled:
            raise ValueError(f"AI定位未启用，无法解析目标: {target_text}")

        errors: list[str] = []
        try:
            resolved = self._resolve_with_ai(
                action=action, target=target_text, timeout=timeout
            )
        except Exception as exc:
            errors.append(f"llm={exc}")
            if not self.vision_settings.enabled:
                raise
            try:
                resolved = self._resolve_with_vision(
                    action=action, target=target_text, timeout=timeout
                )
            except (VisionConfigurationError, VisionServiceUnavailable) as vision_exc:
                logger.warning(
                    f"UI Vision服务不可用，跳过视觉兜底并保持原有定位失败: {vision_exc}"
                )
                raise exc

        if resolved.selector:
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
        elif not resolved.coordinate:
            detail = "; ".join(errors) if errors else target_text
            raise ValueError(f"智能定位未返回可执行目标: {detail}")
        return resolved

    def observe_operation(self, *, instruction: str, timeout: int) -> ObservedOperation:
        if not self.ai_enabled:
            raise ValueError("AI observe未启用")
        self._claim_ai_call("observe")
        candidates = collect_candidates(self.page, limit=self.candidate_limit)
        provider = ChatCompletionProvider()
        decision = provider.complete_model(
            [
                {
                    "role": "system",
                    "content": (
                        "你是受控的UI自动化观察器。只能从候选元素中选择一个操作，"
                        "返回JSON对象，不要解释。字段: action(click/fill/press/wait/skip), "
                        "selector, value, key, wait_ms。"
                    ),
                },
                {
                    "role": "user",
                    "content": self._json_payload(
                        {
                            "url": self.page.url,
                            "prompt_version": self.observe_prompt_version,
                            "schema_version": self.schema_version,
                            "instruction": instruction,
                            "candidates": candidates,
                        }
                    ),
                },
            ],
            ObservedOperationDecision,
            schema_name="ObservedOperationDecision",
            usage_operation="runtime.observe_operation",
            usage_metadata={
                "schema_name": "ObservedOperationDecision",
                "prompt_version": self.observe_prompt_version,
                "page_url": self.page.url,
            },
        )
        action = decision.action
        if action in {"skip", "wait"}:
            return ObservedOperation(
                action=action,
                wait_ms=decision.wait_ms or 1000,
                prompt_version=self.observe_prompt_version,
                schema_version=self.schema_version,
                model=provider.settings.model,
                candidate_count=len(candidates),
                candidate_hash=self._candidate_hash(candidates),
            )
        selector = normalize_selector(str(decision.selector or ""))
        if not selector:
            raise ValueError("AI observe未返回selector")
        verify_action = "fill" if action == "fill" else "click"
        verify_selector(self.page, selector, action=verify_action, timeout=timeout)
        return ObservedOperation(
            action=action,
            selector=selector,
            value=decision.value,
            key=decision.key,
            prompt_version=self.observe_prompt_version,
            schema_version=self.schema_version,
            model=provider.settings.model,
            candidate_count=len(candidates),
            candidate_hash=self._candidate_hash(candidates),
        )

    def apply_operation(self, operation: ObservedOperation, *, timeout: int) -> None:
        if operation.action == "skip":
            return
        if operation.action == "wait":
            self.page.wait_for_timeout(operation.wait_ms or 1000)
            return
        if not operation.selector:
            raise ValueError(f"AI操作 {operation.action} 缺少selector")
        locator = self.page.locator(operation.selector).first
        if operation.action == "click":
            locator.click(timeout=timeout)
        elif operation.action == "fill":
            locator.fill(operation.value or "", timeout=timeout)
        elif operation.action == "press":
            locator.press(operation.key or "Enter", timeout=timeout)
        else:
            raise ValueError(f"不支持的AI操作: {operation.action}")

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
        return ResolvedSelector(
            selector=selector,
            source="ai_observe",
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
