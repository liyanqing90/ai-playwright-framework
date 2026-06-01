"""Post-action wait policy for UI steps."""

from __future__ import annotations

from typing import Any

from ai_playwright.constants import DEFAULT_TIMEOUT
from ai_playwright.step_actions.action_types import StepAction


def _lower_actions(*groups: list[str]) -> set[str]:
    return {action.lower() for group in groups for action in group}


STABILIZE_AFTER_ACTIONS = _lower_actions(
    StepAction.NAVIGATE,
    StepAction.CLICK,
    StepAction.FILL,
    StepAction.PRESS_KEY,
    StepAction.REFRESH,
    StepAction.UPLOAD,
    StepAction.CHECK,
    StepAction.UNCHECK,
    StepAction.SET_CHECKED,
    StepAction.TAP,
    StepAction.PRESS_SEQUENTIALLY,
    StepAction.DOUBLE_CLICK,
    StepAction.RIGHT_CLICK,
    StepAction.SELECT,
    StepAction.SELECT_TEXT,
    StepAction.DRAG_AND_DROP,
    StepAction.TYPE,
    StepAction.CLEAR,
    StepAction.ACCEPT_ALERT,
    StepAction.DISMISS_ALERT,
    StepAction.EXPECT_POPUP,
    StepAction.SWITCH_WINDOW,
    StepAction.CLOSE_WINDOW,
    StepAction.WAIT_FOR_NEW_WINDOW,
    StepAction.GO_BACK,
    StepAction.GO_FORWARD,
    StepAction.WAIT_FOR_URL,
    StepAction.TAB_SWITCH,
    StepAction.DOWNLOAD_FILE,
    StepAction.KEYBOARD_SHORTCUT,
    StepAction.KEYBOARD_PRESS,
    StepAction.KEYBOARD_TYPE,
    StepAction.WAIT_FOR_FUNCTION,
    StepAction.ADD_INIT_SCRIPT,
    StepAction.DISPATCH_EVENT,
    StepAction.MANAGE_COOKIES,
    StepAction.SET_VIEWPORT_SIZE,
    StepAction.EMULATE_MEDIA,
    StepAction.MONITOR_REQUEST,
    StepAction.MONITOR_RESPONSE,
)


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def should_wait_after_action(
    action: str,
    step: dict[str, Any] | None,
    runtime: dict[str, Any] | None,
) -> bool:
    if action not in STABILIZE_AFTER_ACTIONS:
        return False
    if step and "wait_for_stable" in step:
        return _as_bool(step.get("wait_for_stable"), default=False)
    runtime = runtime or {}
    return _as_bool(
        (
            runtime.get("wait_for_stable_after_action")
            if "wait_for_stable_after_action" in runtime
            else runtime.get("auto_wait_for_stable")
        ),
        default=False,
    )


def stable_timeout_ms(
    step: dict[str, Any] | None, runtime: dict[str, Any] | None
) -> int:
    runtime = runtime or {}
    value = (
        (step or {}).get("stable_timeout_ms")
        or (step or {}).get("stable_timeout")
        or runtime.get("action_stable_timeout_ms")
        or DEFAULT_TIMEOUT
    )
    return int(value)


def stable_idle_ms(step: dict[str, Any] | None, runtime: dict[str, Any] | None) -> int:
    runtime = runtime or {}
    value = (
        (step or {}).get("stable_idle_ms")
        or runtime.get("action_stable_idle_ms")
        or 500
    )
    return int(value)
