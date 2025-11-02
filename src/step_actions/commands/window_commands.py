"""
窗口相关的命令
"""

from typing import Dict, Any

from src.step_actions.action_types import StepAction
from src.step_actions.commands.base_command import Command, CommandFactory


@CommandFactory.register(StepAction.EXPECT_POPUP)
class ExpectPopupCommand(Command):
    """弹出tab命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        action = step.get("real_action", "click")
        variable_name = step.get("variable_name", value)
        ui_helper.expect_popup(
            action=action, selector=selector, variable_name=variable_name
        )


@CommandFactory.register(StepAction.SWITCH_WINDOW)
class SwitchWindowCommand(Command):
    """切换窗口命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.switch_window(value=value)


@CommandFactory.register(StepAction.CLOSE_WINDOW)
class CloseWindowCommand(Command):
    """关闭窗口命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.close_window()


@CommandFactory.register(StepAction.TAB_SWITCH)
class TabSwitchCommand(Command):
    """切换标签页命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.switch_window(value=value)
