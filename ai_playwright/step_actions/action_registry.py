from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ai_playwright.step_actions.action_types import StepAction


COMMON_ALLOWED_FIELDS = frozenset(
    {
        "action",
        "description",
        "selector",
        "target",
        "value",
        "expected",
        "timeout",
        "mode",
        "nth",
        "frame",
    }
)


@dataclass(frozen=True)
class ActionSpec:
    actions: frozenset[str]
    fields: frozenset[str] = field(default_factory=frozenset)
    selector_required: bool = True
    required: tuple[str, ...] = ()
    required_any: tuple[str, ...] = ()
    unsafe: bool = False
    category: str = "general"

    @property
    def allowed_fields(self) -> frozenset[str]:
        return COMMON_ALLOWED_FIELDS | self.fields


def action_set(*groups: Iterable[str]) -> frozenset[str]:
    return frozenset(action.lower() for group in groups for action in group)


def _spec(
    *groups: Iterable[str],
    fields: Iterable[str] = (),
    selector_required: bool = True,
    required: Iterable[str] = (),
    required_any: Iterable[str] = (),
    unsafe: bool = False,
    category: str = "general",
) -> ActionSpec:
    return ActionSpec(
        actions=action_set(*groups),
        fields=frozenset(fields),
        selector_required=selector_required,
        required=tuple(required),
        required_any=tuple(required_any),
        unsafe=unsafe,
        category=category,
    )


CLICK_ACTIONS = action_set(
    StepAction.CLICK,
    StepAction.HOVER,
    StepAction.DOUBLE_CLICK,
    StepAction.RIGHT_CLICK,
    StepAction.CHECK,
    StepAction.UNCHECK,
    StepAction.TAP,
    StepAction.SELECT_TEXT,
    StepAction.CLEAR,
    StepAction.SCROLL_INTO_VIEW,
    StepAction.FOCUS,
    StepAction.BLUR,
    StepAction.ENTER_FRAME,
    StepAction.DISMISS_ALERT,
)
VALUE_ACTIONS = action_set(
    StepAction.FILL,
    StepAction.TYPE,
    StepAction.PRESS_SEQUENTIALLY,
    StepAction.SELECT,
    StepAction.UPLOAD,
)
TEXT_ASSERT_ACTIONS = action_set(
    StepAction.ASSERT_TEXT,
    StepAction.HARD_ASSERT_TEXT,
    StepAction.ASSERT_TEXT_CONTAINS,
    StepAction.ASSERT_VALUE,
)
PAGE_ASSERT_ACTIONS = action_set(
    StepAction.ASSERT_URL,
    StepAction.ASSERT_URL_CONTAINS,
    StepAction.ASSERT_TITLE,
    StepAction.ASSERT_TITLE_CONTAINS,
)
PRESENCE_ASSERT_ACTIONS = action_set(
    StepAction.ASSERT_VISIBLE,
    StepAction.ASSERT_BE_HIDDEN,
    StepAction.ASSERT_EXISTS,
    StepAction.ASSERT_NOT_EXISTS,
    StepAction.ASSERT_ENABLED,
    StepAction.ASSERT_DISABLED,
    StepAction.ASSERT_CHECKED,
)
FLOW_ACTIONS = action_set(
    StepAction.USE_MODULE,
    StepAction.IF_CONDITION,
    StepAction.FOR_EACH,
    StepAction.AI_STEP,
)


ACTION_SPECS: tuple[ActionSpec, ...] = (
    _spec(CLICK_ACTIONS | PRESENCE_ASSERT_ACTIONS, category="interaction"),
    _spec(
        StepAction.SCROLL_TO,
        fields={"x", "y"},
        selector_required=False,
        category="interaction",
    ),
    _spec(VALUE_ACTIONS, fields={"delay"}, required=("value",), category="input"),
    _spec(
        StepAction.SET_CHECKED, fields={"checked"}, required_any=("checked", "value")
    ),
    _spec(StepAction.PRESS_KEY, fields={"key"}, required_any=("key", "value")),
    _spec(TEXT_ASSERT_ACTIONS, fields={"expected"}, required_any=("expected", "value")),
    _spec(
        PAGE_ASSERT_ACTIONS,
        fields={"expected"},
        selector_required=False,
        required_any=("expected", "value"),
        category="assertion",
    ),
    _spec(
        StepAction.ASSERT_ATTRIBUTE,
        fields={"attribute", "expected"},
        required=("attribute",),
        required_any=("expected", "value"),
        category="assertion",
    ),
    _spec(
        StepAction.ASSERT_ELEMENT_COUNT,
        fields={"expected", "expression"},
        required_any=("expected", "value", "expression"),
        category="assertion",
    ),
    _spec(
        StepAction.ASSERT_HAVE_VALUES,
        fields={"expected_values"},
        required_any=("expected_values", "value"),
        category="assertion",
    ),
    _spec(
        StepAction.NAVIGATE,
        StepAction.WAIT,
        StepAction.WAIT_FOR_NETWORK_IDLE,
        StepAction.REFRESH,
        StepAction.PAUSE,
        StepAction.WAIT_FOR_NEW_WINDOW,
        StepAction.GO_BACK,
        StepAction.GO_FORWARD,
        StepAction.SWITCH_WINDOW,
        StepAction.TAB_SWITCH,
        StepAction.CLOSE_WINDOW,
        fields={
            "x",
            "y",
            "wait_until",
            "variable_name",
            "scope",
        },
        selector_required=False,
        category="page",
    ),
    _spec(
        StepAction.WAIT_FOR_URL,
        fields={"url", "wait_until"},
        selector_required=False,
        required_any=("url", "value"),
        category="page",
    ),
    _spec(
        StepAction.SET_VIEWPORT_SIZE,
        fields={"width", "height"},
        selector_required=False,
        required=("width", "height"),
        category="page",
    ),
    _spec(
        StepAction.EMULATE_MEDIA,
        fields={"media", "color_scheme", "reduced_motion", "forced_colors"},
        selector_required=False,
        required_any=("media", "color_scheme", "reduced_motion", "forced_colors"),
        category="page",
    ),
    _spec(
        StepAction.EXECUTE_PYTHON,
        fields={"path"},
        selector_required=False,
        required=("value",),
        unsafe=True,
        category="unsafe",
    ),
    _spec(
        StepAction.EXECUTE_SCRIPT,
        fields={"script"},
        selector_required=False,
        required_any=("script", "value"),
        unsafe=True,
        category="unsafe",
    ),
    _spec(
        StepAction.WAIT_FOR_FUNCTION,
        fields={"expression", "arg"},
        selector_required=False,
        required_any=("expression", "value"),
        unsafe=True,
        category="unsafe",
    ),
    _spec(
        StepAction.ADD_INIT_SCRIPT,
        fields={"script", "path"},
        selector_required=False,
        required_any=("script", "path", "value"),
        unsafe=True,
        category="unsafe",
    ),
    _spec(StepAction.ACCEPT_ALERT, category="dialog"),
    _spec(
        StepAction.DRAG_AND_DROP,
        fields={"target_selector"},
        required_any=("target_selector", "value"),
        category="interaction",
    ),
    _spec(
        StepAction.DISPATCH_EVENT,
        fields={"event_type", "event", "event_init"},
        required_any=("event_type", "event", "value"),
        category="interaction",
    ),
    _spec(
        StepAction.CAPTURE_SCREENSHOT,
        fields={"path", "full_page"},
        selector_required=False,
        required_any=("path", "value"),
        category="evidence",
    ),
    _spec(
        StepAction.MANAGE_COOKIES,
        fields={
            "cookie_action",
            "operation",
            "cookies",
            "name",
            "url",
            "domain",
            "path",
            "expires",
            "httpOnly",
            "secure",
            "sameSite",
            "variable_name",
            "scope",
        },
        selector_required=False,
        required_any=("cookie_action", "operation", "value"),
        category="storage",
    ),
    _spec(
        StepAction.GET_VALUE,
        StepAction.GET_TEXT,
        StepAction.GET_BOUNDING_BOX,
        StepAction.GET_ALL_ELEMENTS,
        fields={"variable_name", "scope", "attribute"},
        category="read",
    ),
    _spec(
        StepAction.GET_ATTRIBUTE,
        fields={"variable_name", "scope", "attribute"},
        required_any=("attribute", "value"),
        category="read",
    ),
    _spec(
        StepAction.STORE_TEXT,
        StepAction.STORE_INPUT_VALUE,
        fields={"variable_name", "scope"},
        required=("variable_name",),
        category="storage",
    ),
    _spec(
        StepAction.SAVE_ELEMENT_COUNT,
        fields={"variable_name", "scope"},
        required=("variable_name",),
        category="storage",
    ),
    _spec(
        StepAction.DOWNLOAD_FILE,
        fields={"save_path"},
        category="download",
    ),
    _spec(
        StepAction.DOWNLOAD_VERIFY,
        fields={"file_pattern"},
        selector_required=False,
        required_any=("file_pattern", "value"),
        category="download",
    ),
    _spec(
        StepAction.STORE_ATTRIBUTE,
        fields={"variable_name", "attribute", "scope"},
        required=("variable_name", "attribute"),
        category="storage",
    ),
    _spec(
        StepAction.STORE_VARIABLE,
        fields={"name", "scope", "expression"},
        selector_required=False,
        required=("name",),
        required_any=("value", "expression"),
        category="storage",
    ),
    _spec(
        StepAction.WAIT_FOR_ELEMENT_HIDDEN,
        StepAction.WAIT_FOR_ELEMENT_CLICKABLE,
        category="wait",
    ),
    _spec(
        StepAction.WAIT_FOR_ELEMENT_TEXT,
        fields={"expected_text"},
        required_any=("expected_text", "value"),
        category="wait",
    ),
    _spec(
        StepAction.WAIT_FOR_ELEMENT_COUNT,
        fields={"expected_count"},
        required_any=("expected_count", "value"),
        category="wait",
    ),
    _spec(
        StepAction.EXPECT_POPUP,
        fields={"real_action", "variable_name"},
        category="window",
    ),
    _spec(
        StepAction.FAKER,
        fields={"data_type", "variable_name", "scope", "locale"},
        selector_required=False,
        required=("data_type", "variable_name"),
        category="data",
    ),
    _spec(
        StepAction.KEYBOARD_SHORTCUT,
        fields={"key_combination"},
        selector_required=False,
        required_any=("key_combination", "value"),
        category="keyboard",
    ),
    _spec(
        StepAction.KEYBOARD_PRESS,
        fields={"key"},
        selector_required=False,
        required_any=("key", "value"),
        category="keyboard",
    ),
    _spec(
        StepAction.KEYBOARD_TYPE,
        fields={"text", "delay"},
        selector_required=False,
        required_any=("text", "value"),
        category="keyboard",
    ),
    _spec(
        StepAction.AI_STEP,
        fields={"instruction"},
        selector_required=False,
        required_any=("instruction", "value", "target"),
        category="ai",
    ),
    _spec(
        StepAction.MONITOR_REQUEST,
        StepAction.MONITOR_RESPONSE,
        fields={
            "url_pattern",
            "action_type",
            "assert_params",
            "save_params",
            "variable_name",
            "scope",
            "key",
            "checked",
            "event",
            "event_type",
            "event_init",
        },
        required_any=("url_pattern", "value"),
        category="network",
    ),
    _spec(
        StepAction.WAIT_FOR_REQUEST,
        StepAction.WAIT_FOR_RESPONSE,
        fields={"url_pattern", "variable_name", "scope"},
        selector_required=False,
        required_any=("url_pattern", "value"),
        category="network",
    ),
    _spec(
        StepAction.MOCK_ROUTE,
        fields={"url_pattern", "status", "body", "json", "headers", "content_type"},
        selector_required=False,
        required_any=("url_pattern", "value"),
        category="network",
    ),
    _spec(
        StepAction.ABORT_ROUTE,
        fields={"url_pattern", "error_code"},
        selector_required=False,
        required_any=("url_pattern", "value"),
        category="network",
    ),
    _spec(
        StepAction.UNROUTE,
        fields={"url_pattern"},
        selector_required=False,
        required_any=("url_pattern", "value"),
        category="network",
    ),
    _spec(
        StepAction.SET_OFFLINE,
        fields={"offline"},
        selector_required=False,
        required_any=("offline", "value"),
        category="context",
    ),
    _spec(
        StepAction.SET_EXTRA_HTTP_HEADERS,
        fields={"headers"},
        selector_required=False,
        required=("headers",),
        category="context",
    ),
    _spec(
        StepAction.GRANT_PERMISSIONS,
        fields={"permissions", "origin"},
        selector_required=False,
        required=("permissions",),
        category="context",
    ),
    _spec(
        StepAction.CLEAR_PERMISSIONS,
        selector_required=False,
        category="context",
    ),
    _spec(
        StepAction.SAVE_STORAGE_STATE,
        fields={"path", "variable_name", "scope"},
        selector_required=False,
        category="storage",
    ),
)

ACTION_SPEC_BY_NAME = {action: spec for spec in ACTION_SPECS for action in spec.actions}
VALID_ACTIONS = frozenset(ACTION_SPEC_BY_NAME) | FLOW_ACTIONS
NO_SELECTOR_ACTIONS = (
    frozenset(
        action
        for action, spec in ACTION_SPEC_BY_NAME.items()
        if not spec.selector_required
    )
    | FLOW_ACTIONS
)
ACTION_ALLOWED_FIELDS = {
    action: spec.allowed_fields for action, spec in ACTION_SPEC_BY_NAME.items()
}
