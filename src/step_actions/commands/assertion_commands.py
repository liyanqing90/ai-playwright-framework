"""
断言相关的命令
"""

import json
from typing import Dict, Any

from src.step_actions.action_types import StepAction
from src.step_actions.commands.base_command import Command, CommandFactory
from src.step_actions.expression_evaluator import evaluate_math_expression
from utils.logger import logger


def _log_assertion_success(
    action: str,
    *,
    selector: str | None = None,
    expected: Any = None,
    actual: Any = None,
    **extra: Any,
) -> None:
    parts = [f"action={action}"]
    if selector:
        parts.append(f"selector={selector}")
    if expected is not None:
        parts.append(f"预期结果={expected}")
    if actual is not None:
        parts.append(f"实际结果={actual}")
    for key, value in extra.items():
        if value is not None:
            parts.append(f"{key}={value}")
    logger.info("断言通过: " + " | ".join(parts))


def _resolve_expected(ui_helper, value: Any) -> Any:
    variable_manager = getattr(ui_helper, "variable_manager", None)
    if variable_manager and hasattr(variable_manager, "replace_variables_refactored"):
        return variable_manager.replace_variables_refactored(value)
    return value


def _page_locator(ui_helper, selector: str):
    page = getattr(ui_helper, "page", None)
    if page and hasattr(page, "locator"):
        return page.locator(selector)

    locator_func = getattr(ui_helper, "_locator", None)
    if callable(locator_func):
        return locator_func(selector)

    raise AttributeError("ui_helper 不支持 locator 读取实际结果")


def _first_locator(ui_helper, selector: str):
    return _page_locator(ui_helper, selector).first


def _actual_text(ui_helper, selector: str) -> str:
    return _first_locator(ui_helper, selector).inner_text()


def _actual_url(ui_helper) -> str:
    return ui_helper.page.url


def _actual_title(ui_helper) -> str:
    return ui_helper.page.title()


def _actual_count(ui_helper, selector: str) -> int:
    return _page_locator(ui_helper, selector).count()


def _actual_visible(ui_helper, selector: str) -> bool:
    return _first_locator(ui_helper, selector).is_visible()


def _actual_hidden(ui_helper, selector: str) -> bool:
    return _first_locator(ui_helper, selector).is_hidden()


def _actual_exists(ui_helper, selector: str) -> bool:
    return _actual_count(ui_helper, selector) > 0


def _actual_enabled(ui_helper, selector: str) -> bool:
    return not _first_locator(ui_helper, selector).is_disabled()


def _actual_disabled(ui_helper, selector: str) -> bool:
    return _first_locator(ui_helper, selector).is_disabled()


def _actual_attribute(ui_helper, selector: str, attribute: str) -> Any:
    return _first_locator(ui_helper, selector).get_attribute(attribute)


def _actual_value(ui_helper, selector: str) -> str:
    return _first_locator(ui_helper, selector).input_value()


def _actual_values(ui_helper, selector: str) -> list[Any]:
    return [
        option.get_attribute("value")
        for option in _page_locator(ui_helper, selector).locator("option:checked").all()
    ]


@CommandFactory.register(StepAction.ASSERT_TEXT)
class AssertTextCommand(Command):
    """断言文本命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        expected = step.get("expected", value)
        ui_helper.assert_text(selector=selector, expected=expected)
        _log_assertion_success(
            "assert_text",
            selector=selector,
            expected=_resolve_expected(ui_helper, expected),
            actual=_actual_text(ui_helper, selector),
        )


@CommandFactory.register(StepAction.HARD_ASSERT_TEXT)
class HardAssertTextCommand(Command):
    """硬断言文本命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        expected = step.get("expected", value)
        ui_helper.hard_assert_text(selector=selector, expected=expected)
        _log_assertion_success(
            "hard_assert",
            selector=selector,
            expected=_resolve_expected(ui_helper, expected),
            actual=_actual_text(ui_helper, selector),
        )


@CommandFactory.register(StepAction.ASSERT_TEXT_CONTAINS)
class AssertTextContainsCommand(Command):
    """断言文本包含命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        expected = step.get("expected", str(value))
        ui_helper.assert_text_contains(selector=selector, expected=expected)
        _log_assertion_success(
            "assert_text_contains",
            selector=selector,
            expected=_resolve_expected(ui_helper, expected),
            actual=_actual_text(ui_helper, selector),
        )


@CommandFactory.register(StepAction.ASSERT_URL)
class AssertUrlCommand(Command):
    """断言URL命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        expected = step.get("expected", value)
        ui_helper.assert_url(expected=expected)
        _log_assertion_success(
            "assert_url",
            expected=_resolve_expected(ui_helper, expected),
            actual=_actual_url(ui_helper),
        )


@CommandFactory.register(StepAction.ASSERT_URL_CONTAINS)
class AssertUrlContainsCommand(Command):
    """断言URL包含命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        expected = step.get("expected", value)
        ui_helper.assert_url_contains(expected=expected)
        _log_assertion_success(
            "assert_url_contains",
            expected=_resolve_expected(ui_helper, expected),
            actual=_actual_url(ui_helper),
        )


@CommandFactory.register(StepAction.ASSERT_TITLE)
class AssertTitleCommand(Command):
    """断言标题命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        expected = step.get("expected", value)
        ui_helper.assert_title(expected=expected)
        _log_assertion_success(
            "assert_title",
            expected=_resolve_expected(ui_helper, expected),
            actual=_actual_title(ui_helper),
        )


@CommandFactory.register(StepAction.ASSERT_TITLE_CONTAINS)
class AssertTitleContainsCommand(Command):
    """断言标题包含命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        expected = step.get("expected", value)
        ui_helper.assert_title_contains(expected=expected)
        _log_assertion_success(
            "assert_title_contains",
            expected=_resolve_expected(ui_helper, expected),
            actual=_actual_title(ui_helper),
        )


@CommandFactory.register(StepAction.ASSERT_ELEMENT_COUNT)
class AssertElementCountCommand(Command):
    """断言元素数量命令，支持数学表达式"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        expected = step.get("expected", value)
        expression = step.get("expression")

        # 如果提供了表达式，则计算表达式的值
        if expression:
            try:
                expected = evaluate_math_expression(
                    expression, ui_helper.variable_manager
                )
                from utils.logger import logger

                logger.info(f"计算表达式: {expression} = {expected}")
            except Exception as e:
                from utils.logger import logger

                logger.error(f"计算表达式错误: {expression} - {e}")
                raise

        ui_helper.assert_element_count(selector=selector, expected=expected)
        _log_assertion_success(
            "assert_element_count",
            selector=selector,
            expected=_resolve_expected(ui_helper, expected),
            actual=_actual_count(ui_helper, selector),
        )


@CommandFactory.register(StepAction.ASSERT_VISIBLE)
class AssertVisibleCommand(Command):
    """断言可见命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.assert_visible(selector=selector)
        _log_assertion_success(
            "assert_visible",
            selector=selector,
            expected=True,
            actual=_actual_visible(ui_helper, selector),
        )


@CommandFactory.register(StepAction.ASSERT_BE_HIDDEN)
class AssertBeHiddenCommand(Command):
    """断言隐藏命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.assert_be_hidden(selector=selector)
        _log_assertion_success(
            "assert_be_hidden",
            selector=selector,
            expected=True,
            actual=_actual_hidden(ui_helper, selector),
        )


@CommandFactory.register(StepAction.ASSERT_EXISTS)
class AssertExistsCommand(Command):
    """断言存在命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.assert_exists(selector=selector)
        _log_assertion_success(
            "assert_exists",
            selector=selector,
            expected=True,
            actual=_actual_exists(ui_helper, selector),
            actual_count=_actual_count(ui_helper, selector),
        )


@CommandFactory.register(StepAction.ASSERT_NOT_EXISTS)
class AssertNotExistsCommand(Command):
    """断言不存在命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.assert_not_exists(selector=selector)
        actual_count = _actual_count(ui_helper, selector)
        _log_assertion_success(
            "assert_not_exists",
            selector=selector,
            expected=True,
            actual=actual_count == 0,
            actual_count=actual_count,
        )


@CommandFactory.register(StepAction.ASSERT_ENABLED)
class AssertEnabledCommand(Command):
    """断言启用命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.assert_element_enabled(selector=selector)
        _log_assertion_success(
            "assert_enabled",
            selector=selector,
            expected=True,
            actual=_actual_enabled(ui_helper, selector),
        )


@CommandFactory.register(StepAction.ASSERT_DISABLED)
class AssertDisabledCommand(Command):
    """断言禁用命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.assert_element_disabled(selector=selector)
        _log_assertion_success(
            "assert_disabled",
            selector=selector,
            expected=True,
            actual=_actual_disabled(ui_helper, selector),
        )


@CommandFactory.register(StepAction.ASSERT_ATTRIBUTE)
class AssertAttributeCommand(Command):
    """断言属性命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        attribute = step.get("attribute")
        expected = step.get("expected", value)
        ui_helper.assert_attribute(
            selector=selector, attribute=attribute, expected=expected
        )
        _log_assertion_success(
            "assert_attribute",
            selector=selector,
            expected=_resolve_expected(ui_helper, expected),
            actual=_actual_attribute(ui_helper, selector, attribute),
            attribute=attribute,
        )


@CommandFactory.register(StepAction.ASSERT_VALUE)
class AssertValueCommand(Command):
    """断言值命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        expected = step.get("expected", value)
        ui_helper.assert_value(selector=selector, expected=expected)
        _log_assertion_success(
            "assert_value",
            selector=selector,
            expected=_resolve_expected(ui_helper, expected),
            actual=_actual_value(ui_helper, selector),
        )


@CommandFactory.register(StepAction.ASSERT_HAVE_VALUES)
class AssertHaveValuesCommand(Command):
    """断言多个值命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        expected = step.get("expected_values", value)
        if isinstance(expected, str):
            # 尝试解析为JSON数组
            try:
                expected = json.loads(expected)
            except Exception:
                # 如果不是JSON，则分割字符串
                expected = expected.split(",")
        ui_helper.assert_values(selector=selector, expected=expected)
        _log_assertion_success(
            "assert_have_values",
            selector=selector,
            expected=_resolve_expected(ui_helper, expected),
            actual=_actual_values(ui_helper, selector),
        )
