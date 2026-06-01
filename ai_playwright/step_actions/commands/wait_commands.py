"""
等待相关的命令
"""

from typing import Dict, Any

from ai_playwright.constants import DEFAULT_TIMEOUT
from ai_playwright.step_actions.action_types import StepAction
from ai_playwright.step_actions.commands.base_command import Command, CommandFactory


@CommandFactory.register(StepAction.WAIT)
class WaitCommand(Command):
    """等待命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        wait_time = (
            int(float(step.get("value", 1)) * 1000) if step.get("value") else 1000
        )
        ui_helper.wait_for_timeout(timeout=wait_time)


@CommandFactory.register(StepAction.WAIT_FOR_NETWORK_IDLE)
class WaitForNetworkIdleCommand(Command):
    """等待网络空闲命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        timeout = int(step.get("timeout", DEFAULT_TIMEOUT))
        ui_helper.wait_for_network_idle(timeout=timeout)


@CommandFactory.register(StepAction.WAIT_FOR_ELEMENT_HIDDEN)
class WaitForElementHiddenCommand(Command):
    """等待元素消失命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        timeout = int(step.get("timeout", DEFAULT_TIMEOUT))
        ui_helper.wait_for_element_hidden(selector=selector, timeout=timeout)


@CommandFactory.register(StepAction.WAIT_FOR_ELEMENT_CLICKABLE)
class WaitForElementClickableCommand(Command):
    """等待元素可点击命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        timeout = int(step.get("timeout", DEFAULT_TIMEOUT))
        ui_helper.wait_for_element_clickable(selector=selector, timeout=timeout)


@CommandFactory.register(StepAction.WAIT_FOR_ELEMENT_TEXT)
class WaitForElementTextCommand(Command):
    """等待元素文本命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        timeout = int(step.get("timeout", DEFAULT_TIMEOUT))
        expected_text = step.get("expected_text", value)
        ui_helper.wait_for_element_text(
            selector=selector, expected=expected_text, timeout=timeout
        )


@CommandFactory.register(StepAction.WAIT_FOR_ELEMENT_COUNT)
class WaitForElementCountCommand(Command):
    """等待元素数量命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        timeout = int(step.get("timeout", DEFAULT_TIMEOUT))
        expected_count = int(step.get("expected_count", value))
        ui_helper.wait_for_element_count(
            selector=selector, expected_count=expected_count, timeout=timeout
        )


@CommandFactory.register(StepAction.WAIT_FOR_NEW_WINDOW)
class WaitForNewWindowCommand(Command):
    """等待新窗口命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        timeout = int(step.get("timeout", DEFAULT_TIMEOUT))
        new_page = ui_helper.wait_for_new_window(timeout=timeout)
        if "variable_name" in step:
            ui_helper.store_variable(
                step["variable_name"], new_page, step.get("scope", "global")
            )
