"""
命令模式的基础类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class Command(ABC):
    """命令接口"""

    @abstractmethod
    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> Any:
        """
        执行命令

        Args:
            ui_helper: UI操作帮助类
            selector: 元素选择器
            value: 操作值
            step: 步骤定义

        Returns:
            操作结果
        """
        pass


class CommandFactory:
    """命令工厂类"""

    _commands = {}

    @classmethod
    def register(cls, action_types):
        """
        注册命令的装饰器

        Args:
            action_types: 操作类型列表
        """

        def decorator(command_class):
            for action_type in action_types:
                cls._commands[action_type.lower()] = command_class
            return command_class

        return decorator

    @classmethod
    def get_command(cls, action: str) -> Optional[Command]:
        """
        获取命令实例

        Args:
            action: 操作类型

        Returns:
            命令实例，如果不存在则返回None
        """
        command_class = cls._commands.get(action.lower())
        if command_class:
            return command_class()
        return None
