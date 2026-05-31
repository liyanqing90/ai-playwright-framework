"""
其他杂项命令
"""

import os
from typing import Dict, Any

from ai_playwright.constants import DEFAULT_TYPE_DELAY
from ai_playwright.step_actions.action_types import StepAction
from ai_playwright.step_actions.commands.base_command import Command, CommandFactory
from ai_playwright.step_actions.utils import (
    generate_faker_data,
    run_dynamic_script_from_path,
)


def _unsafe_actions_enabled() -> bool:
    value = os.environ.get("AI_PLAYWRIGHT_ALLOW_UNSAFE_ACTIONS", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
        ui_helper.accept_alert(selector=selector, value=value)


@CommandFactory.register(StepAction.DISMISS_ALERT)
class DismissAlertCommand(Command):
    """取消弹框命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.dismiss_alert(selector=selector, value=value)


@CommandFactory.register(StepAction.EXECUTE_PYTHON)
class ExecutePythonCommand(Command):
    """执行Python命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        if not _unsafe_actions_enabled():
            raise PermissionError(
                "execute_python 默认禁用；设置 AI_PLAYWRIGHT_ALLOW_UNSAFE_ACTIONS=1 后再执行"
            )
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
        if not _unsafe_actions_enabled():
            raise PermissionError(
                "execute_script 默认禁用；设置 AI_PLAYWRIGHT_ALLOW_UNSAFE_ACTIONS=1 后再执行"
            )
        script = step.get("script", value)
        if not script:
            raise ValueError("execute_script 缺少 script 或 value")
        ui_helper.execute_script(script=script)


@CommandFactory.register(StepAction.WAIT_FOR_FUNCTION)
class WaitForFunctionCommand(Command):
    """等待 JavaScript 条件命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        if not _unsafe_actions_enabled():
            raise PermissionError(
                "wait_for_function 默认禁用；设置 AI_PLAYWRIGHT_ALLOW_UNSAFE_ACTIONS=1 后再执行"
            )
        expression = step.get("expression", value)
        if not expression:
            raise ValueError("wait_for_function 缺少 expression 或 value")
        ui_helper.wait_for_function(
            expression=expression,
            arg=step.get("arg"),
            timeout=int(step.get("timeout", 0) or 0) or None,
        )


@CommandFactory.register(StepAction.ADD_INIT_SCRIPT)
class AddInitScriptCommand(Command):
    """添加初始化脚本命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        if not _unsafe_actions_enabled():
            raise PermissionError(
                "add_init_script 默认禁用；设置 AI_PLAYWRIGHT_ALLOW_UNSAFE_ACTIONS=1 后再执行"
            )
        ui_helper.add_init_script(
            script=step.get("script", value),
            path=step.get("path"),
        )


@CommandFactory.register(StepAction.DISPATCH_EVENT)
class DispatchEventCommand(Command):
    """触发 DOM 事件命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        event_type = step.get("event_type") or step.get("event") or value
        if not event_type:
            raise ValueError("dispatch_event 缺少 event_type 或 value")
        ui_helper.dispatch_event(
            selector=selector,
            event_type=event_type,
            event_init=step.get("event_init"),
        )


@CommandFactory.register(StepAction.CAPTURE_SCREENSHOT)
class CaptureScreenshotCommand(Command):
    """截图命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        path = step.get("path") or value
        if not path:
            raise ValueError("截图操作缺少 path 或 value")
        full_page = step.get("full_page", False)
        if isinstance(full_page, str):
            full_page = full_page.strip().lower() in {"1", "true", "yes", "on"}
        ui_helper.capture_screenshot(
            path=path,
            selector=selector,
            full_page=bool(full_page),
        )


@CommandFactory.register(StepAction.SET_VIEWPORT_SIZE)
class SetViewportSizeCommand(Command):
    """设置视口大小命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        width = step.get("width")
        height = step.get("height")
        if not width or not height:
            raise ValueError("set_viewport_size 缺少 width 或 height")
        ui_helper.set_viewport_size(width=int(width), height=int(height))


@CommandFactory.register(StepAction.EMULATE_MEDIA)
class EmulateMediaCommand(Command):
    """模拟媒体环境命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.emulate_media(
            media=step.get("media"),
            color_scheme=step.get("color_scheme"),
            reduced_motion=step.get("reduced_motion"),
            forced_colors=step.get("forced_colors"),
        )


@CommandFactory.register(StepAction.MANAGE_COOKIES)
class ManageCookiesCommand(Command):
    """Cookie 操作命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        cookie_action = step.get("cookie_action") or step.get("operation") or value
        if not cookie_action:
            raise ValueError("Cookie 操作缺少 cookie_action 或 value")
        kwargs = {
            key: item
            for key, item in step.items()
            if key
            not in {
                "action",
                "cookie_action",
                "operation",
                "selector",
                "target",
                "value",
                "variable_name",
                "scope",
                "description",
                "timeout",
                "mode",
                "nth",
            }
        }
        result = ui_helper.manage_cookies(str(cookie_action), **kwargs)
        if "variable_name" in step:
            ui_helper.store_variable(
                step["variable_name"], result, step.get("scope", "global")
            )


@CommandFactory.register(StepAction.GET_VALUE)
class GetValueCommand(Command):
    """获取元素值命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> Any:
        result = ui_helper.get_value(selector=selector)
        if "variable_name" in step:
            ui_helper.store_variable(
                step["variable_name"], result, step.get("scope", "global")
            )
        return result


@CommandFactory.register(StepAction.GET_TEXT)
class GetTextCommand(Command):
    """获取元素文本命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> Any:
        result = ui_helper.get_text(selector=selector)
        if "variable_name" in step:
            ui_helper.store_variable(
                step["variable_name"], result, step.get("scope", "global")
            )
        return result


@CommandFactory.register(StepAction.GET_ATTRIBUTE)
class GetAttributeCommand(Command):
    """获取元素属性命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> Any:
        attribute = step.get("attribute", value)
        if not attribute:
            raise ValueError("get_attribute 缺少 attribute 或 value")
        result = ui_helper.get_element_attribute(selector=selector, attribute=attribute)
        if "variable_name" in step:
            ui_helper.store_variable(
                step["variable_name"], result, step.get("scope", "global")
            )
        return result


@CommandFactory.register(StepAction.GET_BOUNDING_BOX)
class GetBoundingBoxCommand(Command):
    """获取元素边界框命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> Any:
        result = ui_helper.get_bounding_box(selector=selector)
        if "variable_name" in step:
            ui_helper.store_variable(
                step["variable_name"], result, step.get("scope", "global")
            )
        return result


@CommandFactory.register(StepAction.GET_ALL_ELEMENTS)
class GetAllElementsCommand(Command):
    """获取所有匹配元素命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> Any:
        result = ui_helper.get_all_elements(selector=selector)
        if "variable_name" in step:
            ui_helper.store_variable(
                step["variable_name"], result, step.get("scope", "global")
            )
        return result
