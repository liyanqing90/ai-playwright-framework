"""
导航相关的命令
"""

from typing import Dict, Any

from ai_playwright.page_objects.base_page import base_url
from ai_playwright.step_actions.action_types import StepAction
from ai_playwright.step_actions.commands.base_command import Command, CommandFactory


@CommandFactory.register(StepAction.NAVIGATE)
class NavigateCommand(Command):
    """导航命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        url = base_url()
        if not value:
            value = url
        if "http" not in value:
            value = url + value
        ui_helper.navigate(
            url=value,
            wait_until=step.get("wait_until", "domcontentloaded"),
        )


@CommandFactory.register(StepAction.REFRESH)
class RefreshCommand(Command):
    """刷新页面命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.refresh()


@CommandFactory.register(StepAction.GO_BACK)
class GoBackCommand(Command):
    """后退命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.go_back(wait_until=step.get("wait_until", "domcontentloaded"))


@CommandFactory.register(StepAction.GO_FORWARD)
class GoForwardCommand(Command):
    """前进命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.go_forward(wait_until=step.get("wait_until", "domcontentloaded"))


@CommandFactory.register(StepAction.WAIT_FOR_URL)
class WaitForUrlCommand(Command):
    """等待 URL 命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        expected_url = step.get("url", value)
        if not expected_url:
            raise ValueError("wait_for_url 缺少 url 或 value")
        ui_helper.wait_for_url(
            expected_url,
            timeout=int(step.get("timeout", 0) or 0) or None,
            wait_until=step.get("wait_until", "domcontentloaded"),
        )


@CommandFactory.register(StepAction.PAUSE)
class PauseCommand(Command):
    """暂停命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.pause()
