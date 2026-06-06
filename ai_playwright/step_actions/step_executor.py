"""
步骤执行器的核心实现
"""

import os
import re
import threading
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, Any, List, Callable

import allure

from ai_playwright.constants import DEFAULT_TIMEOUT
from ai_playwright.ai_runtime.config import load_ai_config, runtime_mode
from ai_playwright.ai_runtime.element_store import (
    ElementDefinitionStore,
    register_element_update_thread,
)
from ai_playwright.ai_runtime.playwright_selectors import looks_like_raw_selector
from ai_playwright.ai_runtime.smart_resolver import SmartResolver
from ai_playwright.step_actions.action_registry import (
    NO_SELECTOR_ACTIONS,
    VALID_ACTIONS,
)
from ai_playwright.step_actions.action_result import ActionResult

# 导入命令模式执行器
from ai_playwright.step_actions.command_executor import execute_action_with_command
from ai_playwright.step_actions.flow_control import (
    execute_condition,
    execute_loop,
    evaluate_expression,
)
from ai_playwright.step_actions.module_handler import execute_module
from ai_playwright.step_actions.wait_policy import (
    should_wait_after_action,
    stable_idle_ms,
    stable_timeout_ms,
)
from ai_playwright.utils.logger import logger
from ai_playwright.utils.variable_manager import VariableManager


# 导入所有命令类


_PENDING_SELECTOR_CACHE_LOCK = threading.RLock()
_PENDING_SELECTOR_CACHE: list[tuple[str, Callable[[], None]]] = []
_DEFERRED_SELECTOR_CACHE_MODES = {"deferred", "after_test", "after-case", "after_case"}
_PERSISTED_SELECTOR_UPDATES_LOCK = threading.RLock()
_PERSISTED_SELECTOR_UPDATES: list[dict[str, Any]] = []


def _selector_cache_commit_mode() -> str:
    override = os.environ.get("UI_SELECTOR_CACHE_COMMIT_MODE")
    if override is not None:
        return override.strip().lower()
    if os.environ.get("UI_SELECTOR_CACHE_HOLD_UNTIL_GENERATION_VERIFIED"):
        return "deferred"
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return "deferred"
    return "immediate"


def _defer_selector_cache_commit(description: str, commit: Callable[[], None]) -> None:
    with _PENDING_SELECTOR_CACHE_LOCK:
        _PENDING_SELECTOR_CACHE.append((description, commit))
    logger.info(f"selector cache deferred until test passes: {description}")


def commit_pending_selector_cache() -> None:
    with _PENDING_SELECTOR_CACHE_LOCK:
        pending = list(_PENDING_SELECTOR_CACHE)
        _PENDING_SELECTOR_CACHE.clear()
    for description, commit in pending:
        try:
            commit()
            logger.info(f"selector cache committed after verified test: {description}")
        except Exception as exc:
            logger.exception(
                "selector cache commit failed after verified test: "
                f"{description} | error={exc}"
            )


def pop_persisted_selector_updates() -> list[dict[str, Any]]:
    with _PERSISTED_SELECTOR_UPDATES_LOCK:
        updates = list(_PERSISTED_SELECTOR_UPDATES)
        _PERSISTED_SELECTOR_UPDATES.clear()
    return updates


def discard_pending_selector_cache(reason: str | None = None) -> None:
    with _PENDING_SELECTOR_CACHE_LOCK:
        count = len(_PENDING_SELECTOR_CACHE)
        _PENDING_SELECTOR_CACHE.clear()
    if count:
        logger.info(
            "selector cache discarded before test verification completed: "
            f"count={count} | reason={reason or 'not verified'}"
        )


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class StepExecutor:

    def __init__(
        self,
        page,
        ui_helper,
        elements: Dict[str, Any],
        default_mode: str | None = None,
    ):
        self.has_error = None
        self.page = page
        self.ui_helper = ui_helper
        self.elements = elements or {}
        self.default_mode = default_mode
        self.ai_config = load_ai_config()
        self.smart_resolver = None
        self.element_store = None
        self._healing_threads: list[threading.Thread] = []
        self.start_time = None
        self.step_has_error = False  # 步骤错误状态
        self.last_failure_kind = None
        self._log_buffer = StringIO()  # 步骤日志缓存
        self._buffer_handler_id = None
        self._prepare_evidence_dir()
        self._VALID_ACTIONS = set(VALID_ACTIONS)
        self._NO_SELECTOR_ACTIONS = set(NO_SELECTOR_ACTIONS)

        # 初始化变量管理器
        self.variable_manager = VariableManager()

        # 初始化项目名称
        self.project_name = None

        # 已加载的模块缓存
        self.modules_cache = {}

    @staticmethod
    def _prepare_evidence_dir():
        """创建截图存储目录"""
        Path("./evidence/screenshots").mkdir(parents=True, exist_ok=True)

    def setup(self, elements: Dict[str, Any] = None):
        """设置元素定义，在测试开始前调用"""
        if elements:
            self.elements = elements

    def execute_steps(
        self, steps: List[Dict[str, Any]], project_name: str = None
    ) -> None:
        """
        执行多个测试步骤

        Args:
            steps: 测试步骤列表
            project_name: 项目名称，用于加载模块
        """
        self.project_name = project_name
        for step in steps:
            self.execute_step(step)

    def execute_step(self, step: Dict[str, Any]) -> None:
        try:
            self.start_time = datetime.now()
            self.step_has_error = False
            self.last_failure_kind = None
            self._sync_runtime_page_from_ui_helper()

            # 检查是否为流程控制步骤
            if "use_module" in step:
                execute_module(self, step)
                return
            elif "if" in step:
                execute_condition(self, step)
                return
            elif "for_each" in step:
                execute_loop(self, step)
                return

            action = step.get("action", "").lower()
            if action in {"ai_step", "observe", "ai操作", "智能操作"}:
                self._execute_ai_step(step)
                return

            mode = self._resolve_mode(step)
            pre_selector = step.get("selector")
            element_key = (
                pre_selector
                if isinstance(pre_selector, str) and pre_selector in self.elements
                else None
            )
            raw_selector = (
                self.elements.get(pre_selector, pre_selector)
                if element_key
                else pre_selector
            )
            selector = self.variable_manager.replace_variables_refactored(raw_selector)
            target = self.variable_manager.replace_variables_refactored(
                step.get("target")
            )
            frame = self.variable_manager.replace_variables_refactored(
                step.get("frame")
            )
            if mode != "strict" and not target and pre_selector:
                fallback_target = None
                selector_is_semantic_key = False
                if element_key:
                    fallback_target = _target_from_element_key(element_key)
                elif not (
                    isinstance(pre_selector, str)
                    and looks_like_raw_selector(pre_selector)
                ):
                    fallback_target = _target_from_element_key(pre_selector)
                    selector_is_semantic_key = True
                if fallback_target:
                    target = self.variable_manager.replace_variables_refactored(
                        fallback_target
                    )
                if selector_is_semantic_key:
                    selector = None
            selector = self._resolve_selector(
                action,
                selector,
                target,
                mode,
                step,
                element_key=element_key,
            )
            selector = self._apply_frame(selector, frame)
            selector = self._apply_nth(selector, step)
            value = self.variable_manager.replace_variables_refactored(
                step.get("value")
            )  # 替换变量
            if value is not None:
                step["_resolved_value"] = value
            self._log_step_execution(action=action, selector=selector, value=value)
            self._log_ai_mode(
                mode=mode,
                action=action,
                target=target,
                selector=selector,
                step=step,
            )
            self._validate_step(action, selector, step)
            action_before_page = self._active_page()
            action_before_url = self._safe_page_url(action_before_page)
            action_before_page_errors = self._page_error_count()
            action_result = self._execute_action(action, selector, value, step)
            step["_action_result"] = action_result.to_step_dict()
            self._wait_after_action_stable(action, step)
            self._sync_runtime_page_from_ui_helper()
            self._capture_action_outcome(
                step,
                before_page=action_before_page,
                before_url=action_before_url,
                before_page_errors=action_before_page_errors,
            )
            self._record_action_page_errors(step)
            self._persist_resolved_selector_after_success(
                element_key=element_key,
                original_selector=raw_selector,
                resolved_selector=selector,
                step=step,
            )
        except AssertionError as e:
            self.has_error = True
            self.step_has_error = True
            self.last_failure_kind = self._classify_step_exception(e)
            # 检查是否为硬断言异常
            if hasattr(e, "_hard_assert") and getattr(e, "_hard_assert", False):
                logger.error(f"硬断言失败，终止测试执行: {e}")
                raise  # 硬断言失败时重新抛出异常，终止测试执行
            # 软断言失败时不抛出异常，继续执行
        except Exception as e:
            self.last_failure_kind = self._classify_step_exception(e)
            logger.error(f"步骤执行失败: kind={self.last_failure_kind} | {e}")
            self.has_error = True
            self.step_has_error = True
            raise e
        finally:
            self._finalize_step()

    def _validate_step(
        self, action, selector, step: Dict[str, Any] | None = None
    ) -> None:
        if not action:
            raise ValueError("步骤缺少必要参数: action", f"原始输入: {action}")
        # 操作类型白名单校验
        if action not in self._VALID_ACTIONS:
            raise ValueError(f"不支持的操作类型: {action}")
        # 必要参数校验
        if action not in self._NO_SELECTOR_ACTIONS and not selector:
            raise ValueError(f"操作 {action} 需要提供selector参数")

    def _execute_action(
        self, action: str, selector: str, value: Any = None, step: Dict[str, Any] = None
    ) -> ActionResult:
        """执行具体操作，使用命令模式"""
        try:
            result = execute_action_with_command(
                self.ui_helper, action, selector, value, step
            )
            return ActionResult.ok(data=result)
        except AssertionError as e:
            # 标记异常为断言失败
            self.step_has_error = True
            self.last_failure_kind = self._classify_step_exception(e)
            raise e
        except Exception as e:
            # 统一处理所有异常，不区分超时与非超时
            # 标记步骤失败
            self.step_has_error = True
            self.has_error = True

            # 只添加必要的错误信息，不进行额外处理
            if not hasattr(e, "_logged"):
                logger.error(f"步骤执行失败: {e}")
                setattr(e, "_logged", True)

            # 添加关键信息
            failure_kind = self._classify_step_exception(e)
            setattr(e, "_action", action)
            setattr(e, "_selector", selector)
            setattr(e, "_value", value)
            setattr(e, "_failure_kind", failure_kind)
            self.last_failure_kind = failure_kind

            # 直接抛出异常，不做其他任何处理
            raise

    @staticmethod
    def _classify_step_exception(exc: Exception) -> str:
        text = str(exc).lower()
        exc_name = exc.__class__.__name__.lower()
        if isinstance(exc, AssertionError):
            return "assertion_failed"
        if "pageerror" in text or "page error" in text or "页面错误" in text:
            return "page_error"
        if "expect_request" in text or "expect_response" in text:
            return "network_timeout"
        if "request" in text and "timeout" in text:
            return "network_timeout"
        if "response" in text and "timeout" in text:
            return "network_timeout"
        if "navigation" in text or "wait_for_url" in text or "to_have_url" in text:
            return "navigation_timeout"
        if "timeout" in text or "timeout" in exc_name:
            if "locator" in text or "元素" in text:
                return "locator_timeout"
            return "actionability_timeout"
        return "execution_error"

    def _wait_after_action_stable(self, action: str, step: Dict[str, Any]) -> None:
        if not self._should_wait_after_action(action, step):
            return

        wait_for_stable = getattr(self.ui_helper, "wait_for_stable", None)
        if not callable(wait_for_stable):
            logger.debug(f"跳过页面稳定等待: ui_helper不支持 | action={action}")
            return

        timeout = self._stable_timeout_ms(step)
        idle_ms = self._stable_idle_ms(step)
        logger.debug(
            f"等待页面稳定: action={action} | timeout={timeout}ms | idle={idle_ms}ms"
        )
        wait_for_stable(timeout=timeout, idle_ms=idle_ms)

    def _sync_runtime_page_from_ui_helper(self) -> None:
        page = getattr(self.ui_helper, "page", None)
        if page is None or page is self.page:
            return
        self.page = page
        if self.smart_resolver is not None:
            self.smart_resolver.page = page

    def _active_page(self) -> Any:
        return getattr(self.ui_helper, "page", None) or self.page

    @staticmethod
    def _safe_page_url(page: Any) -> str:
        try:
            return str(getattr(page, "url", "") or "")
        except Exception:
            return ""

    def _page_error_count(self) -> int:
        return len(self._page_errors())

    def _page_errors(self) -> list[str]:
        errors = getattr(self.ui_helper, "page_errors", None)
        if isinstance(errors, list):
            return [str(error) for error in errors]
        return []

    @staticmethod
    def _page_key_from_url(url: str | None) -> str:
        return str(url or "about:blank").split("?")[0]

    def _capture_action_outcome(
        self,
        step: Dict[str, Any],
        *,
        before_page: Any,
        before_url: str,
        before_page_errors: int,
    ) -> None:
        after_page = self._active_page()
        after_url = self._safe_page_url(after_page)
        page_errors = self._page_errors()
        after_page_errors = len(page_errors)
        new_page_errors = page_errors[before_page_errors:after_page_errors]
        step["_action_before_url"] = before_url
        step["_action_after_url"] = after_url
        step["_action_before_page_key"] = self._page_key_from_url(before_url)
        step["_action_after_page_key"] = self._page_key_from_url(after_url)
        step["_action_page_error_count_delta"] = max(
            0, after_page_errors - before_page_errors
        )
        if new_page_errors:
            step["_action_page_errors"] = new_page_errors
        step["_action_page_changed"] = (
            before_page is not None
            and after_page is not None
            and before_page is not after_page
        )

    def _record_action_page_errors(self, step: Dict[str, Any]) -> None:
        if int(step.get("_action_page_error_count_delta") or 0) <= 0:
            return
        errors = step.get("_action_page_errors") or []
        detail = "; ".join(str(error) for error in errors) or "unknown page error"
        logger.warning(f"action triggered page error after execution: {detail}")

    def _should_wait_after_action(self, action: str, step: Dict[str, Any]) -> bool:
        runtime = self.ai_config.get("runtime", {})
        return should_wait_after_action(action, step, runtime)

    def _stable_timeout_ms(self, step: Dict[str, Any]) -> int:
        runtime = self.ai_config.get("runtime", {})
        return stable_timeout_ms(step, runtime)

    def _stable_idle_ms(self, step: Dict[str, Any]) -> int:
        runtime = self.ai_config.get("runtime", {})
        return stable_idle_ms(step, runtime)

    def _resolve_mode(self, step: Dict[str, Any]) -> str:
        config_default = self.ai_config.get("runtime", {}).get("default_mode", "strict")
        mode = step.get("mode") or self.default_mode or runtime_mode(config_default)
        mode = str(mode or "strict").lower()
        if mode not in {"strict", "smart"}:
            raise ValueError(f"不支持的AI执行模式: {mode}")
        return mode

    def _resolve_selector(
        self,
        action: str,
        selector: str | None,
        target: str | None,
        mode: str,
        step: Dict[str, Any],
        *,
        element_key: str | None = None,
    ) -> str | None:
        if action in self._NO_SELECTOR_ACTIONS:
            return selector
        if isinstance(selector, dict):
            return selector
        if mode == "strict":
            return selector
        if (
            target is None
            and isinstance(selector, str)
            and looks_like_raw_selector(selector)
        ):
            return selector
        resolver = self._get_smart_resolver()
        resolved = resolver.resolve(
            action=action,
            target=target,
            selector=selector,
            mode=mode,
            timeout=int(step.get("timeout", DEFAULT_TIMEOUT)),
        )
        step["_resolved_selector_source"] = resolved.source
        step["_resolved_by_ai"] = resolved.ai_called
        step["_resolved_confidence"] = resolved.confidence
        step["_resolved_prompt_version"] = resolved.prompt_version
        step["_resolved_schema_version"] = resolved.schema_version
        step["_resolved_model"] = resolved.model
        step["_resolved_candidate_count"] = resolved.candidate_count
        step["_resolved_candidate_hash"] = resolved.candidate_hash
        step["_resolved_registry_record_id"] = resolved.registry_record_id
        step["_resolved_cache_action"] = resolved.cache_action
        step["_resolved_cache_target"] = resolved.cache_target
        step["_resolved_cache_page_key"] = resolved.cache_page_key
        step["_resolved_cache_replace_active"] = resolved.cache_replace_active
        if element_key:
            step["_resolved_element_key"] = element_key
        if resolved.healed:
            step["_resolved_healed"] = True
        if resolved.healing_attempted:
            step["_resolved_healing_attempted"] = True
        if resolved.original_selector:
            step["_resolved_original_selector"] = resolved.original_selector
        if resolved.original_error:
            step["_resolved_original_error"] = resolved.original_error
        if resolved.selector:
            step["_resolved_selector"] = resolved.selector
        return resolved.selector

    def _persist_healed_selector(
        self,
        *,
        element_key: str | None,
        original_selector: str | None,
        resolved_selector: str | None,
        healing_attempted: bool,
        step: Dict[str, Any],
    ) -> None:
        if not healing_attempted or not element_key:
            return
        if not resolved_selector:
            logger.warning(
                "selector自愈未回写elements: "
                f"key={element_key} | reason=resolved_selector为空"
            )
            return
        if original_selector == resolved_selector:
            return
        if not self._healed_selector_persist_allowed(step):
            return
        if not self._persist_healed_elements_enabled():
            logger.info(
                "selector自愈未回写elements: "
                f"key={element_key} | reason=配置关闭 | selector={resolved_selector}"
            )
            return

        step["_resolved_persist_scheduled"] = True
        thread = threading.Thread(
            target=self._persist_healed_selector_worker,
            kwargs={
                "element_key": element_key,
                "resolved_selector": resolved_selector,
                "old_selector": original_selector,
                "step": step,
            },
            name=f"selector-heal-{element_key}",
            daemon=True,
        )
        thread.start()
        self._healing_threads.append(thread)
        register_element_update_thread(thread)
        logger.info(
            "selector自愈回写elements已提交: "
            f"key={element_key} | old={original_selector} | new={resolved_selector}"
        )

    def _persist_healed_selector_worker(
        self,
        *,
        element_key: str,
        resolved_selector: str,
        old_selector: str | None,
        step: Dict[str, Any],
    ) -> None:
        try:
            result = self._get_element_store().update_selector(
                element_key,
                resolved_selector,
                identifier=step.get("_resolved_cache_target") or step.get("target"),
                allow_semantic_generic_update=bool(
                    step.get("_generation_persist_verified_heals")
                    and str(step.get("action") or "").lower()
                    in {
                        "click",
                        "press",
                        "press_key",
                        "check",
                        "uncheck",
                        "set_checked",
                    }
                ),
            )
            if not result.updated and result.reason != "unchanged":
                logger.error(
                    "selector自愈回写elements失败，不阻塞执行: "
                    f"key={element_key} | selector={resolved_selector} | reason={result.reason}"
                )
                return

            self.elements[result.key] = resolved_selector
            if result.path:
                step["_resolved_persisted_element_file"] = str(result.path)
            if result.key != element_key:
                step["_resolved_persisted_element_key"] = result.key

            self._record_persisted_selector_update(
                source_key=element_key,
                result=result,
                old_selector=old_selector,
                step=step,
            )

            if result.updated:
                logger.info(
                    "selector自愈已回写elements: "
                    f"key={element_key} | file={result.path} | old={result.old_selector} | new={resolved_selector}"
                )
            else:
                logger.info(
                    "selector自愈回写elements无需变更: "
                    f"key={element_key} | file={result.path} | selector={resolved_selector}"
                )
        except Exception as exc:
            logger.exception(
                "selector自愈回写elements异常，不阻塞执行: "
                f"key={element_key} | old={old_selector} | new={resolved_selector} | error={exc}"
            )

    def _persist_healed_selector_after_verified(
        self,
        *,
        element_key: str | None,
        original_selector: str | None,
        resolved_selector: str | None,
        healing_attempted: bool,
        step: Dict[str, Any],
    ) -> bool:
        if not healing_attempted or not element_key:
            return False
        if not resolved_selector:
            logger.warning(
                "selector healed but elements not updated: "
                f"key={element_key} | reason=empty resolved_selector"
            )
            return False
        if original_selector == resolved_selector:
            return True
        generation_persist_allowed = self._generation_verified_heal_persist_allowed(
            step
        )
        if not generation_persist_allowed:
            return False

        self._bind_element_store_for_verified_heal()
        step["_resolved_persist_scheduled"] = True
        self._persist_healed_selector_worker(
            element_key=element_key,
            resolved_selector=resolved_selector,
            old_selector=original_selector,
            step=step,
        )
        return True

    @staticmethod
    def _record_persisted_selector_update(
        *,
        source_key: str,
        result: Any,
        old_selector: str | None,
        step: Dict[str, Any],
    ) -> None:
        if not step.get("_generation_persist_verified_heals"):
            return
        update = {
            "source_key": source_key,
            "persisted_key": result.key,
            "selector": result.new_selector,
            "old_selector": old_selector,
            "updated": result.updated,
            "reason": result.reason,
            "path": str(result.path) if result.path else "",
            "target": step.get("_resolved_cache_target") or step.get("target"),
            "action": step.get("action"),
        }
        with _PERSISTED_SELECTOR_UPDATES_LOCK:
            _PERSISTED_SELECTOR_UPDATES.append(update)

    def _bind_element_store_for_verified_heal(self) -> None:
        if self.element_store is None:
            self.element_store = ElementDefinitionStore(
                test_dir=os.environ.get("TEST_DIR")
            )

    def _persist_healed_elements_enabled(self) -> bool:
        override = os.environ.get("UI_AI_PERSIST_HEALED_SELECTORS")
        if override is not None:
            return override.strip().lower() not in {"0", "false", "no", "off"}
        cfg = self.ai_config.get("self_healing", {})
        return bool(cfg.get("persist_elements", True))

    def _healed_selector_persist_allowed(self, step: Dict[str, Any]) -> bool:
        cfg = self.ai_config.get("self_healing", {})
        action = str(step.get("action") or "").lower()
        if action.startswith("assert") and not bool(
            cfg.get("persist_assertion_selectors", False)
        ):
            logger.info(
                "selector自愈未回写elements: "
                f"key={step.get('_resolved_element_key')} | reason=断言步骤默认不持久化"
            )
            return False
        confidence = step.get("_resolved_confidence")
        if self._generation_verified_heal_persist_allowed(step):
            logger.info(
                "selector healed persistence allowed: "
                f"key={step.get('_resolved_element_key')} | "
                f"reason=generation verified action | confidence={confidence}"
            )
            return True
        min_confidence = float(cfg.get("min_persist_confidence", 0.85))
        if confidence is None or float(confidence) < min_confidence:
            logger.info(
                "selector自愈未回写elements: "
                f"key={step.get('_resolved_element_key')} | reason=置信度不足 "
                f"| confidence={confidence} | min={min_confidence}"
            )
            return False
        return True

    @staticmethod
    def _generation_verified_heal_persist_allowed(step: Dict[str, Any]) -> bool:
        if not (
            step.get("_generation_persist_verified_heals")
            or _truthy_env("UI_GENERATION_PERSIST_VERIFIED_HEALS")
        ):
            return False
        action_result = step.get("_action_result")
        if isinstance(action_result, dict) and action_result.get("success") is False:
            return False
        return bool(step.get("_resolved_healing_attempted"))

    def _persist_resolved_selector_after_success(
        self,
        *,
        element_key: str | None,
        original_selector: str | None,
        resolved_selector: str | None,
        step: Dict[str, Any],
    ) -> None:
        cache_allowed, reason = self._verified_selector_cache_allowed(
            step=step,
            resolved_selector=resolved_selector,
        )
        if not cache_allowed:
            if self._has_resolved_selector_cache_work(step):
                logger.info(f"selector cache skipped after action: {reason}")
            return
        self._schedule_verified_selector_persist(
            element_key=element_key,
            original_selector=original_selector,
            resolved_selector=resolved_selector,
            step=step,
        )

    @staticmethod
    def _has_resolved_selector_cache_work(step: Dict[str, Any]) -> bool:
        return bool(
            step.get("_resolved_registry_record_id")
            or step.get("_resolved_cache_action")
            or step.get("_resolved_healing_attempted")
        )

    def _verified_selector_cache_allowed(
        self,
        *,
        step: Dict[str, Any],
        resolved_selector: str | None,
    ) -> tuple[bool, str]:
        if not resolved_selector:
            return False, "empty selector"

        before_key = step.get("_action_before_page_key")
        after_key = step.get("_action_after_page_key")
        cache_key = step.get("_resolved_cache_page_key")
        if before_key and cache_key and before_key != cache_key:
            return (
                False,
                f"resolved page changed before action: resolved={cache_key} before={before_key}",
            )
        if before_key and after_key and before_key != after_key:
            return (
                False,
                f"action navigated before cache verification: before={before_key} after={after_key}",
            )
        if step.get("_action_page_changed"):
            return False, "action switched page before cache verification"
        if int(step.get("_action_page_error_count_delta") or 0) > 0:
            return False, "action triggered page error before cache verification"
        return True, "verified"

    def _persist_verified_registry_selector(
        self,
        *,
        resolved_selector: str | None,
        step: Dict[str, Any],
    ) -> None:
        if not (
            step.get("_resolved_registry_record_id")
            or step.get("_resolved_cache_action")
            or step.get("_resolved_cache_target")
        ):
            return
        resolver = self.smart_resolver
        if resolver is None or not hasattr(resolver, "record_verified_selector"):
            return
        try:
            resolver.record_verified_selector(
                action=step.get("_resolved_cache_action"),
                target=step.get("_resolved_cache_target"),
                selector=resolved_selector,
                source=step.get("_resolved_selector_source"),
                confidence=step.get("_resolved_confidence"),
                registry_record_id=step.get("_resolved_registry_record_id"),
                page_key=step.get("_resolved_cache_page_key"),
                prompt_version=step.get("_resolved_prompt_version"),
                schema_version=step.get("_resolved_schema_version"),
                model=step.get("_resolved_model"),
                candidate_hash=step.get("_resolved_candidate_hash"),
                candidate_count=step.get("_resolved_candidate_count"),
                replace_active=bool(step.get("_resolved_cache_replace_active")),
            )
        except Exception as exc:
            logger.exception(
                "selector registry persist failed after verified action: "
                f"selector={resolved_selector} | error={exc}"
            )

    def _schedule_verified_selector_persist(
        self,
        *,
        element_key: str | None,
        original_selector: str | None,
        resolved_selector: str | None,
        step: Dict[str, Any],
    ) -> None:
        if element_key and step.get("_resolved_healing_attempted"):
            self._bind_element_store_for_verified_heal()
        snapshot = dict(step)
        snapshot["_generation_persist_verified_heals"] = _truthy_env(
            "UI_GENERATION_PERSIST_VERIFIED_HEALS"
        )
        description = self._selector_cache_description(
            step=snapshot,
            resolved_selector=resolved_selector,
            element_key=element_key,
        )

        def commit() -> None:
            self._persist_verified_registry_selector(
                resolved_selector=resolved_selector,
                step=snapshot,
            )
            persisted = self._persist_healed_selector_after_verified(
                element_key=element_key,
                original_selector=original_selector,
                resolved_selector=resolved_selector,
                healing_attempted=bool(snapshot.get("_resolved_healing_attempted")),
                step=snapshot,
            )
            if not persisted:
                self._log_healed_selector_yaml_suggestion(
                    element_key=element_key,
                    original_selector=original_selector,
                    resolved_selector=resolved_selector,
                    healing_attempted=bool(snapshot.get("_resolved_healing_attempted")),
                )

        if _selector_cache_commit_mode() in _DEFERRED_SELECTOR_CACHE_MODES:
            _defer_selector_cache_commit(description, commit)
            return
        commit()

    @staticmethod
    def _selector_cache_description(
        *,
        step: Dict[str, Any],
        resolved_selector: str | None,
        element_key: str | None,
    ) -> str:
        return (
            f"action={step.get('_resolved_cache_action') or step.get('action')} "
            f"| target={step.get('_resolved_cache_target') or step.get('target') or step.get('selector')} "
            f"| selector={resolved_selector} | element={element_key or ''}"
        )

    def _get_element_store(self) -> ElementDefinitionStore:
        if self.element_store is None:
            self.element_store = ElementDefinitionStore()
        return self.element_store

    @staticmethod
    def _log_healed_selector_yaml_suggestion(
        *,
        element_key: str | None,
        original_selector: str | None,
        resolved_selector: str | None,
        healing_attempted: bool,
    ) -> None:
        if not healing_attempted or not element_key or not resolved_selector:
            return
        if original_selector == resolved_selector:
            return
        logger.warning(
            "selector self-healed and verified; YAML update recommended: "
            f"element={element_key} | old={original_selector} | new={resolved_selector}"
        )

    @staticmethod
    def _apply_nth(selector: Any, step: Dict[str, Any]) -> Any:
        if selector is None or "nth" not in step:
            return selector
        nth = step["nth"]
        if nth is None:
            return selector
        if isinstance(selector, dict):
            return {**selector, "nth": int(nth)}
        if ">> nth=" in str(selector):
            return selector
        return f"{selector} >> nth={int(nth)}"

    @staticmethod
    def _apply_frame(selector: Any, frame: Any) -> Any:
        if not frame or selector is None:
            return selector
        if isinstance(selector, dict):
            return {**selector, "frame": frame}
        return {"frame": frame, "selector": selector}

    def _execute_ai_step(self, step: Dict[str, Any]) -> None:
        instruction = self.variable_manager.replace_variables_refactored(
            step.get("instruction") or step.get("value") or step.get("target")
        )
        if not instruction:
            raise ValueError("AI步骤需要instruction、value或target")
        timeout = int(step.get("timeout", DEFAULT_TIMEOUT))
        operation = self._get_smart_resolver().resolve_ai_step(
            instruction=instruction,
            timeout=timeout,
        )
        compiled_step = self._compile_ai_step(operation, timeout=timeout)
        if compiled_step is None:
            logger.debug(f"AI步骤跳过: 指令: {instruction}")
            return

        action = str(compiled_step["action"]).lower()
        selector = compiled_step.get("selector")
        value = self.variable_manager.replace_variables_refactored(
            compiled_step.get("value")
        )
        logger.debug(
            "AI步骤编译: "
            f"指令: {instruction} | action={action} | selector={selector} | value={value}"
        )
        self._log_step_execution(action=action, selector=selector, value=value)
        step.update(
            {
                "_resolved_selector_source": operation.source,
                "_resolved_by_ai": True,
                "_resolved_prompt_version": operation.prompt_version,
                "_resolved_schema_version": operation.schema_version,
                "_resolved_model": operation.model,
                "_resolved_candidate_count": operation.candidate_count,
                "_resolved_candidate_hash": operation.candidate_hash,
            }
        )
        self._log_ai_mode(
            mode="smart",
            action=action,
            target=instruction,
            selector=selector,
            step=step,
        )
        self._validate_step(action, selector, compiled_step)
        action_result = self._execute_action(action, selector, value, compiled_step)
        step["_action_result"] = action_result.to_step_dict()
        self._wait_after_action_stable(action, compiled_step)
        self._sync_runtime_page_from_ui_helper()

    @staticmethod
    def _compile_ai_step(operation, *, timeout: int) -> Dict[str, Any] | None:
        if operation.action == "reject":
            reason = operation.reason or "该指令不是单一原子动作"
            raise ValueError(
                "ai_step只能编译为一个标准step；"
                f"{reason}。请拆成多个steps，或改用agent_case。"
            )
        if operation.action == "skip":
            return None
        if operation.action == "wait":
            return {
                "action": "wait",
                "value": str((operation.wait_ms or 1000) / 1000),
                "timeout": timeout,
            }
        if operation.action == "press":
            return {
                "action": "press_key",
                "selector": operation.selector,
                "key": operation.key or "Enter",
                "value": operation.key or "Enter",
                "timeout": timeout,
            }
        if operation.action in {"click", "fill"}:
            compiled = {
                "action": operation.action,
                "selector": operation.selector,
                "timeout": timeout,
            }
            if operation.action == "fill":
                compiled["value"] = operation.value or ""
            return compiled
        raise ValueError(f"AI步骤返回了不支持的动作: {operation.action}")

    def _get_smart_resolver(self) -> SmartResolver:
        self._sync_runtime_page_from_ui_helper()
        if self.smart_resolver is None:
            self.smart_resolver = SmartResolver(self.page)
        return self.smart_resolver

    @staticmethod
    def _log_step_execution(action: str, selector: Any, value: Any) -> None:
        logger.debug(f"执行步骤: {action} | 选择器: {selector} | 值: {value}")

    def _log_ai_mode(
        self,
        *,
        mode: str,
        action: str,
        target: Any,
        selector: Any,
        step: Dict[str, Any],
    ) -> None:
        if mode == "strict":
            return
        source = step.get("_resolved_selector_source")
        ai_called = "是" if step.get("_resolved_by_ai") else "否"
        if action in self._NO_SELECTOR_ACTIONS:
            logger.debug(f"AI执行模式: {mode} | 定位来源: 无需定位")
            return
        parts = [
            f"AI执行模式: {mode}",
            f"定位来源: {self._source_label(source)}",
        ]
        if target:
            parts.append(f"目标: {target}")
        parts.extend([f"选择器: {selector}", f"AI兜底: {ai_called}"])
        parts.extend(self._ai_metadata_parts(step))
        logger.debug(" | ".join(parts))

    @staticmethod
    def _ai_metadata_parts(source: Dict[str, Any]) -> List[str]:
        metadata = [
            ("healed", "_resolved_healed"),
            ("healing_attempted", "_resolved_healing_attempted"),
            ("element_key", "_resolved_element_key"),
            ("original_selector", "_resolved_original_selector"),
            ("confidence", "_resolved_confidence"),
            ("prompt_version", "_resolved_prompt_version"),
            ("schema_version", "_resolved_schema_version"),
            ("model", "_resolved_model"),
            ("candidate_count", "_resolved_candidate_count"),
            ("candidate_hash", "_resolved_candidate_hash"),
            ("registry_record_id", "_resolved_registry_record_id"),
            ("cache_page_key", "_resolved_cache_page_key"),
            ("persist_scheduled", "_resolved_persist_scheduled"),
            ("persisted_element_file", "_resolved_persisted_element_file"),
        ]
        parts: List[str] = []
        for label, key in metadata:
            value = source.get(key)
            if value is not None:
                parts.append(f"{label}: {value}")
        return parts

    @staticmethod
    def _source_label(source: Any) -> str:
        labels = {
            "explicit": "显式选择器",
            "registry": "历史定位",
            "heuristic": "DOM语义匹配",
            "ai_step": "AI原生步骤",
            "ai_step_fast": "AI step local compile",
            "ai_selector": "AI selector兜底",
        }
        return labels.get(str(source or ""), str(source or "未知"))

    def _replace_variables(self, value: Any) -> Any:
        """
        替换值中的变量引用

        Args:
            value: 原始值，可能包含变量引用 ${var_name} 或 $<var_name> 或 $[[expression]]

        Returns:
            替换后的值
        """
        if value is None:
            return value

        if isinstance(value, (int, float, bool)):
            return value

        if isinstance(value, str):
            # 处理数学表达式引用，如 $[[1 + 2 * ${var}]]
            if (
                value.startswith("$[[")
                and value.endswith("]]")
                and value.count("$[[") == 1
            ):
                try:
                    from ai_playwright.step_actions.expression_evaluator import (
                        evaluate_math_expression,
                    )

                    # 提取表达式内容
                    expr = value[3:-2].strip()
                    # 计算表达式
                    result = evaluate_math_expression(expr, self.variable_manager)
                    return result
                except Exception as e:
                    logger.error(f"计算表达式错误: {value} - {e}")
                    raise

            # 处理完整的变量引用，如 ${var_name} 或 $<var_name>
            if (
                value.startswith("${")
                and value.endswith("}")
                and value.count("${") == 1
            ) or (
                value.startswith("$<")
                and value.endswith(">")
                and value.count("$<") == 1
            ):

                if value.startswith("${"):
                    var_name = value[2:-1]
                else:  # value.startswith("$<")
                    var_name = value[2:-1]

                return self.variable_manager.get_variable(var_name)

            # 替换内嵌变量引用
            import re

            # 同时匹配 ${var_name} 和 $<var_name> 两种模式
            pattern = r"\${([^{}]+)}|\$<([^<>]+)>"

            def replace_var(match):
                # 获取匹配的组，第一个组是 ${} 形式，第二个组是 $<> 形式
                var_name = (
                    match.group(1) if match.group(1) is not None else match.group(2)
                )
                var_value = self.variable_manager.get_variable(var_name)
                return str(var_value) if var_value is not None else match.group(0)

            # 使用正则表达式替换所有变量引用
            result = re.sub(pattern, replace_var, value)

            # 处理内嵌的数学表达式引用，如 "Total: $[[1 + 2 * ${var}]]"
            pattern_expr = r"\$\[\[([^\[\]]+)\]\]"

            def replace_expr(match):
                try:
                    from ai_playwright.step_actions.expression_evaluator import (
                        evaluate_math_expression,
                    )

                    # 提取表达式内容
                    expr = match.group(1).strip()
                    # 计算表达式
                    result = evaluate_math_expression(expr, self.variable_manager)
                    return str(result)
                except Exception as e:
                    logger.error(f"计算表达式错误: {match.group(0)} - {e}")
                    raise

            # 替换所有内嵌的数学表达式
            result = re.sub(pattern_expr, replace_expr, result)

            return result

        if isinstance(value, list):
            return [self._replace_variables(item) for item in value]

        if isinstance(value, dict):
            return {k: self._replace_variables(v) for k, v in value.items()}

        return value

    def _evaluate_expression(self, expression: str) -> bool:
        """
        计算表达式的值

        Args:
            expression: 表达式字符串，如 "${{ ${count} > 5 }}"

        Returns:
            表达式的布尔结果
        """
        return evaluate_expression(self, expression)

    def _finalize_step(self):
        """统一后处理逻辑"""
        # 移除日志handler
        if self._buffer_handler_id:
            logger.remove(self._buffer_handler_id)
            self._buffer_handler_id = None

        # 记录耗时
        self._log_step_duration()

        # 失败时采集证据
        if self.step_has_error:
            self._capture_failure_evidence()

    def _log_step_duration(self):
        """统一记录步骤耗时"""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            if self.step_has_error:
                logger.error(f"[失败] 步骤耗时: {duration:.2f}s")
            else:
                logger.info(f"[成功] 步骤耗时: {duration:.2f}s")

    def _capture_failure_evidence(self):
        """统一失败证据采集"""
        try:
            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            context_info = (
                f"URL: {self.page.url}\n"
                f"错误时间: {timestamp}\n"
                f"失败分类: {self.last_failure_kind or 'unknown'}"
            )
            allure.attach(
                context_info,
                name="失败上下文",
                attachment_type=allure.attachment_type.TEXT,
            )

        except Exception as e:
            logger.error(f"证据采集失败: {str(e)}")


def _target_from_element_key(value: str) -> str:
    text = str(value or "").strip()
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
                words.append(replacements.get(item.lower(), item))
    return " ".join(words).strip() or text
