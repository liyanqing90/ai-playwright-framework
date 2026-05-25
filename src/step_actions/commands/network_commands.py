"""
网络相关的命令
"""

from typing import Dict, Any

from constants import DEFAULT_TIMEOUT
from src.step_actions.action_types import StepAction
from src.step_actions.commands.base_command import Command, CommandFactory
from src.step_actions.network_monitor import (
    monitor_action_request,
    monitor_action_response,
)
from utils.logger import logger


@CommandFactory.register(StepAction.MONITOR_REQUEST)
class MonitorRequestCommand(Command):
    """监测请求命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        # 获取参数
        url_pattern = step.get("url_pattern", value)
        action_type = step.get("action_type", "click")
        assert_params = step.get("assert_params")
        variable_name = step.get("variable_name")
        timeout = int(step.get("timeout", DEFAULT_TIMEOUT))
        scope = step.get("scope", "global")

        # 其他可能的参数
        kwargs = {}
        if action_type == "fill" and "value" in step:
            kwargs["value"] = step.get("value")
        elif action_type == "press_key" and "key" in step:
            kwargs["key"] = step.get("key")
        elif action_type == "select" and "value" in step:
            kwargs["value"] = step.get("value")

        # 检查URL格式
        if (
            url_pattern
            and "http" not in url_pattern
            and not url_pattern.startswith("*")
        ):
            if url_pattern.startswith("/"):
                url_pattern = f"**{url_pattern}**"
            else:
                url_pattern = f"**/{url_pattern}**"

        # 创建一个模拟的StepExecutor对象，只包含必要的属性
        class MockStepExecutor:
            def __init__(self, page, ui_helper, variable_manager):
                self.page = page
                self.ui_helper = ui_helper
                self.variable_manager = variable_manager

        mock_executor = MockStepExecutor(
            ui_helper.page, ui_helper, ui_helper.variable_manager
        )

        # 调用监测方法
        request_data = monitor_action_request(
            mock_executor,
            url_pattern=url_pattern,
            selector=selector,
            action=action_type,
            assert_params=assert_params,
            timeout=timeout,
            value=value,
            **kwargs,
        )

        # 如果提供了变量名，存储捕获数据
        if variable_name:
            ui_helper.variable_manager.set_variable(variable_name, request_data, scope)
            logger.info(f"已存储请求数据到变量 {variable_name}")


@CommandFactory.register(StepAction.MONITOR_RESPONSE)
class MonitorResponseCommand(Command):
    """监测响应命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        # 获取参数
        url_pattern = step.get("url_pattern", value)
        action_type = step.get("action_type", "click")
        assert_params = step.get("assert_params")
        save_params = step.get("save_params")
        timeout = int(step.get("timeout", DEFAULT_TIMEOUT))
        scope = step.get("scope", "global")
        variable_name = step.get("variable_name")

        # 其他可能的参数
        kwargs = {}
        if action_type == "fill" and "value" in step:
            kwargs["value"] = step.get("value")
        elif action_type == "press_key" and "key" in step:
            kwargs["key"] = step.get("key")
        elif action_type == "select" and "value" in step:
            kwargs["value"] = step.get("value")

        # 检查URL格式
        if (
            url_pattern
            and "http" not in url_pattern
            and not url_pattern.startswith("*")
        ):
            if url_pattern.startswith("/"):
                url_pattern = f"**{url_pattern}**"
            else:
                url_pattern = f"**/{url_pattern}**"

        # 创建一个模拟的StepExecutor对象，只包含必要的属性
        class MockStepExecutor:
            def __init__(self, page, ui_helper, variable_manager):
                self.page = page
                self.ui_helper = ui_helper
                self.variable_manager = variable_manager

        mock_executor = MockStepExecutor(
            ui_helper.page, ui_helper, ui_helper.variable_manager
        )

        # 调用监测方法
        response_data = monitor_action_response(
            mock_executor,
            url_pattern=url_pattern,
            selector=selector,
            action=action_type,
            assert_params=assert_params,
            save_params=save_params,
            timeout=timeout,
            value=value,
            **kwargs,
        )

        # 如果提供了变量名，存储捕获数据
        if variable_name:
            ui_helper.variable_manager.set_variable(variable_name, response_data, scope)
            logger.info(f"已存储响应数据到变量 {variable_name}")
