"""
其他杂项命令
"""

from typing import Dict, Any

from constants import DEFAULT_TYPE_DELAY
from src.step_actions.action_types import StepAction
from src.step_actions.commands.base_command import Command, CommandFactory
from src.step_actions.utils import generate_faker_data, run_dynamic_script_from_path


@CommandFactory.register(StepAction.SCROLL_INTO_VIEW)
class ScrollIntoViewCommand(Command):
    """滚动到元素命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.scroll_into_view(selector=selector)


@CommandFactory.register(StepAction.SCROLL_TO)
class ScrollToCommand(Command):
    """滚动到位置命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        action = str(step.get("action", "")).lower()
        if action in {"滑到顶部", "滑到底部"} and selector:
            top_expr = "0" if action == "滑到顶部" else "el.scrollHeight"
            ui_helper._locator(selector).evaluate(
                f"el => el.scrollTo({{top: {top_expr}, behavior: 'instant'}})"
            )
            return
        x = int(step.get("x", 0))
        y = int(step.get("y", 0))
        ui_helper.scroll_to(x=x, y=y)


@CommandFactory.register(StepAction.FOCUS)
class FocusCommand(Command):
    """聚焦命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.focus(selector=selector)


@CommandFactory.register(StepAction.BLUR)
class BlurCommand(Command):
    """失焦命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.blur(selector=selector)


@CommandFactory.register(StepAction.ENTER_FRAME)
class EnterFrameCommand(Command):
    """进入框架命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.enter_frame(selector=selector)


@CommandFactory.register(StepAction.ACCEPT_ALERT)
class AcceptAlertCommand(Command):
    """接受弹框命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.accept_alert(selector=selector, prompt_text=value)


@CommandFactory.register(StepAction.DISMISS_ALERT)
class DismissAlertCommand(Command):
    """取消弹框命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.dismiss_alert(selector=selector)


@CommandFactory.register(StepAction.EXECUTE_PYTHON)
class ExecutePythonCommand(Command):
    """执行Python命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        run_dynamic_script_from_path(value)


@CommandFactory.register(StepAction.FAKER)
class FakerCommand(Command):
    """生成数据命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        data_type = step.get("data_type")
        kwargs = {
            k: v
            for k, v in step.items()
            if k not in ["action", "data_type", "variable_name", "scope"]
        }

        if "variable_name" not in step:
            raise ValueError("步骤缺少必要参数: variable_name")

        # 生成数据
        value = generate_faker_data(data_type, **kwargs)
        ui_helper.store_variable(
            step["variable_name"], value, step.get("scope", "global")
        )


@CommandFactory.register(StepAction.KEYBOARD_SHORTCUT)
class KeyboardShortcutCommand(Command):
    """键盘快捷键命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        key_combination = step.get("key_combination", value)
        ui_helper.press_keyboard_shortcut(key_combination)


@CommandFactory.register(StepAction.KEYBOARD_PRESS)
class KeyboardPressCommand(Command):
    """全局按键命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        key = step.get("key", value)
        ui_helper.keyboard_press(key)


@CommandFactory.register(StepAction.KEYBOARD_TYPE)
class KeyboardTypeCommand(Command):
    """全局输入命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        text = step.get("text", value)
        delay = int(step.get("delay", DEFAULT_TYPE_DELAY))
        ui_helper.keyboard_type(text, delay)


@CommandFactory.register(StepAction.EXECUTE_SCRIPT)
class ExecuteScriptCommand(Command):
    """执行JavaScript脚本命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.execute_script(script=value)
