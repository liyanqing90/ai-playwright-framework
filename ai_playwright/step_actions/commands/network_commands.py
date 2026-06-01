"""
网络相关的命令
"""

from typing import Dict, Any

from ai_playwright.constants import DEFAULT_TIMEOUT
from ai_playwright.step_actions.action_types import StepAction
from ai_playwright.step_actions.commands.base_command import Command, CommandFactory
from ai_playwright.step_actions.network_monitor import (
    monitor_action_request,
    monitor_action_response,
)
from ai_playwright.utils.logger import logger


def _step_url_pattern(step: Dict[str, Any], value: Any) -> Any:
    url_pattern = step.get("url_pattern", value)
    if (
        url_pattern
        and "http" not in str(url_pattern)
        and not str(url_pattern).startswith("*")
    ):
        if str(url_pattern).startswith("/"):
            url_pattern = f"**{url_pattern}**"
        else:
            url_pattern = f"**/{url_pattern}**"
    return url_pattern


def _store_if_requested(ui_helper, step: Dict[str, Any], value: Any) -> None:
    variable_name = step.get("variable_name")
    if variable_name:
        ui_helper.variable_manager.set_variable(
            variable_name, value, step.get("scope", "global")
        )


@CommandFactory.register(StepAction.MONITOR_REQUEST)
class MonitorRequestCommand(Command):
    """监测请求命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        # 获取参数
        url_pattern = _step_url_pattern(step, value)
        action_type = step.get("action_type", "click")
        assert_params = step.get("assert_params")
        variable_name = step.get("variable_name")
        timeout = int(step.get("timeout", DEFAULT_TIMEOUT))
        scope = step.get("scope", "global")

        kwargs = {
            key: step[key]
            for key in ("value", "key", "checked", "event_type", "event", "event_init")
            if key in step
        }

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
        url_pattern = _step_url_pattern(step, value)
        action_type = step.get("action_type", "click")
        assert_params = step.get("assert_params")
        save_params = step.get("save_params")
        timeout = int(step.get("timeout", DEFAULT_TIMEOUT))
        scope = step.get("scope", "global")
        variable_name = step.get("variable_name")

        kwargs = {
            key: step[key]
            for key in ("value", "key", "checked", "event_type", "event", "event_init")
            if key in step
        }

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


@CommandFactory.register(StepAction.WAIT_FOR_REQUEST)
class WaitForRequestCommand(Command):
    """等待请求命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        result = ui_helper.wait_for_request(
            _step_url_pattern(step, value),
            timeout=int(step.get("timeout", DEFAULT_TIMEOUT)),
        )
        _store_if_requested(ui_helper, step, result)


@CommandFactory.register(StepAction.WAIT_FOR_RESPONSE)
class WaitForResponseCommand(Command):
    """等待响应命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        result = ui_helper.wait_for_response(
            _step_url_pattern(step, value),
            timeout=int(step.get("timeout", DEFAULT_TIMEOUT)),
        )
        _store_if_requested(ui_helper, step, result)


@CommandFactory.register(StepAction.MOCK_ROUTE)
class MockRouteCommand(Command):
    """模拟路由命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.mock_route(
            url_pattern=_step_url_pattern(step, value),
            status=int(step.get("status", 200)),
            body=step.get("body", ""),
            json_data=step.get("json"),
            headers=step.get("headers"),
            content_type=step.get("content_type"),
        )


@CommandFactory.register(StepAction.ABORT_ROUTE)
class AbortRouteCommand(Command):
    """中止路由命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.abort_route(
            url_pattern=_step_url_pattern(step, value),
            error_code=step.get("error_code", "failed"),
        )


@CommandFactory.register(StepAction.UNROUTE)
class UnrouteCommand(Command):
    """取消路由命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.unroute(url_pattern=_step_url_pattern(step, value))


@CommandFactory.register(StepAction.SET_OFFLINE)
class SetOfflineCommand(Command):
    """设置离线模式命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        offline = step.get("offline", value)
        if isinstance(offline, str):
            offline = offline.strip().lower() in {"1", "true", "yes", "on"}
        ui_helper.set_offline(bool(offline))


@CommandFactory.register(StepAction.SET_EXTRA_HTTP_HEADERS)
class SetExtraHttpHeadersCommand(Command):
    """设置额外请求头命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        headers = step.get("headers")
        if not isinstance(headers, dict):
            raise ValueError("set_extra_http_headers 需要 headers 对象")
        ui_helper.set_extra_http_headers(headers)


@CommandFactory.register(StepAction.GRANT_PERMISSIONS)
class GrantPermissionsCommand(Command):
    """授予浏览器权限命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        permissions = step.get("permissions")
        if isinstance(permissions, str):
            permissions = [permissions]
        if not isinstance(permissions, list) or not permissions:
            raise ValueError("grant_permissions 需要 permissions 列表")
        ui_helper.grant_permissions(permissions, origin=step.get("origin"))


@CommandFactory.register(StepAction.CLEAR_PERMISSIONS)
class ClearPermissionsCommand(Command):
    """清除浏览器权限命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        ui_helper.clear_permissions()


@CommandFactory.register(StepAction.SAVE_STORAGE_STATE)
class SaveStorageStateCommand(Command):
    """保存 storage state 命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        result = ui_helper.save_storage_state(path=step.get("path", value))
        _store_if_requested(ui_helper, step, result)
