"""
交互相关的命令
"""

from typing import Dict, Any

from ai_playwright.constants import DEFAULT_TYPE_DELAY
from ai_playwright.step_actions.action_types import StepAction
from ai_playwright.step_actions.commands.base_command import Command, CommandFactory


@CommandFactory.register(StepAction.CLICK)
class ClickCommand(Command):
    """点击命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.click(selector=selector)


@CommandFactory.register(StepAction.FILL)
class FillCommand(Command):
    """填充命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.fill(selector=selector, value=value)


@CommandFactory.register(StepAction.PRESS_KEY)
class PressKeyCommand(Command):
    """按键命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.press_key(selector=selector, key=step.get("key", value))


@CommandFactory.register(StepAction.CHECK)
class CheckCommand(Command):
    """勾选命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.check(selector=selector)


@CommandFactory.register(StepAction.UNCHECK)
class UncheckCommand(Command):
    """取消勾选命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.uncheck(selector=selector)


@CommandFactory.register(StepAction.SET_CHECKED)
class SetCheckedCommand(Command):
    """设置勾选状态命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        checked = step.get("checked", value)
        if isinstance(checked, str):
            checked = checked.strip().lower() not in {"0", "false", "no", "off"}
        ui_helper.set_checked(selector=selector, checked=bool(checked))


@CommandFactory.register(StepAction.TYPE)
class TypeCommand(Command):
    """模拟输入命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        delay = int(step.get("delay", DEFAULT_TYPE_DELAY))
        ui_helper.type(selector=selector, text=value, delay=delay)


@CommandFactory.register(StepAction.PRESS_SEQUENTIALLY)
class PressSequentiallyCommand(Command):
    """顺序输入命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        delay = int(step.get("delay", DEFAULT_TYPE_DELAY))
        ui_helper.press_sequentially(selector=selector, text=value, delay=delay)


@CommandFactory.register(StepAction.CLEAR)
class ClearCommand(Command):
    """清空命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.clear(selector=selector)


@CommandFactory.register(StepAction.HOVER)
class HoverCommand(Command):
    """悬停命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.hover(selector=selector)


@CommandFactory.register(StepAction.TAP)
class TapCommand(Command):
    """触屏轻触命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.tap(selector=selector)


@CommandFactory.register(StepAction.DOUBLE_CLICK)
class DoubleClickCommand(Command):
    """双击命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.double_click(selector=selector)


@CommandFactory.register(StepAction.RIGHT_CLICK)
class RightClickCommand(Command):
    """右键点击命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.right_click(selector=selector)


@CommandFactory.register(StepAction.SELECT)
class SelectCommand(Command):
    """选择命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.select_option(selector=selector, value=value)


@CommandFactory.register(StepAction.SELECT_TEXT)
class SelectTextCommand(Command):
    """选择元素文本命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.select_text(selector=selector)


@CommandFactory.register(StepAction.UPLOAD)
class UploadCommand(Command):
    """上传文件命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.upload_file(selector=selector, file_path=value)


@CommandFactory.register(StepAction.DRAG_AND_DROP)
class DragAndDropCommand(Command):
    """拖拽命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        target = step.get("target_selector") or step.get("target") or value
        if not target:
            raise ValueError("拖拽操作缺少目标选择器: target_selector 或 value")
        ui_helper.drag_and_drop(source=selector, target=target)
