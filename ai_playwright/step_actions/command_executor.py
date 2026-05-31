"""
使用命令模式的步骤执行器
"""

from typing import Dict, Any

from ai_playwright.step_actions.commands.base_command import CommandFactory
from ai_playwright.utils.logger import logger


def execute_action_with_command(
    ui_helper,
    action: str,
    selector: str,
    value: Any = None,
    step: Dict[str, Any] = None,
) -> Any:
    """
    使用命令模式执行具体操作

    Args:
        ui_helper: UI操作帮助类
        action: 操作类型
        selector: 元素选择器
        value: 操作值
        step: 步骤定义
    """
    action = action.lower()

    # 使用命令工厂获取命令
    command = CommandFactory.get_command(action)

    if command:
        try:
            # 如果找到命令，执行命令
            return command.execute(ui_helper, selector, value, step)
        except Exception as e:
            # 检查异常是否已被记录
            if not hasattr(e, "_logged"):
                logger.error(f"使用命令模式执行操作失败: {e}")
                setattr(e, "_logged", True)
            raise
    else:
        # 如果没有找到命令，记录错误
        logger.error(f"不支持的操作类型: {action}")
        raise ValueError(f"不支持的操作类型: {action}")
