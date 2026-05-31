"""
处理网络监控相关的操作
"""

import json
from typing import Dict, Any

import allure
from jsonpath_ng import parse

from ai_playwright.utils.logger import logger


def _execute_monitored_action(
    step_executor, action: str, selector: str, **kwargs
) -> None:
    ui_helper = step_executor.ui_helper
    action = str(action or "click").lower()
    if action == "click":
        ui_helper.click(selector)
    elif action == "fill":
        ui_helper.fill(selector, kwargs.get("value", ""))
    elif action == "press_key":
        ui_helper.press_key(selector, kwargs.get("key", ""))
    elif action == "select":
        ui_helper.select_option(selector, kwargs.get("value", ""))
    elif action == "check":
        ui_helper.check(selector)
    elif action == "uncheck":
        ui_helper.uncheck(selector)
    elif action == "set_checked":
        checked = kwargs.get("checked", kwargs.get("value", True))
        if isinstance(checked, str):
            checked = checked.strip().lower() not in {"0", "false", "no", "off"}
        ui_helper.set_checked(selector, bool(checked))
    elif action == "tap":
        ui_helper.tap(selector)
    elif action == "dispatch_event":
        event_type = (
            kwargs.get("event_type") or kwargs.get("event") or kwargs.get("value")
        )
        if not event_type:
            raise ValueError("监测 dispatch_event 缺少 event_type 或 value")
        ui_helper.dispatch_event(selector, event_type, kwargs.get("event_init"))
    elif action == "goto":
        ui_helper.navigate(kwargs.get("value", ""))
    else:
        raise ValueError(f"不支持的监测操作类型: {action}")


def monitor_action_request(
    step_executor,
    url_pattern: str,
    selector: str,
    action: str = "click",
    assert_params: Dict[str, Any] = None,
    timeout: int = None,
    **kwargs,
):
    """
    监测操作触发的请求并验证参数

    Args:
        step_executor: StepExecutor实例
        url_pattern: URL匹配模式，如 "**/api/user/**"
        selector: 要操作的元素选择器
        action: 要执行的操作，如 "click", "goto" 等
        assert_params: 要验证的参数列表，格式为 [{"$.path.to.field": expected_value}, ...]
        timeout: 等待超时时间(毫秒)
        **kwargs: 其他操作参数，如 goto 操作的 value

    Returns:
        捕获的请求数据
    """
    from ai_playwright.constants import DEFAULT_TIMEOUT

    if timeout is None:
        timeout = DEFAULT_TIMEOUT

    logger.info(f"开始监测请求: {url_pattern}, 操作: {action} 元素: {selector}")

    try:
        with step_executor.page.expect_request(
            url_pattern, timeout=timeout
        ) as request_info:
            _execute_monitored_action(step_executor, action, selector, **kwargs)

            # 等待请求完成
            request = request_info.value
            logger.info(f"捕获到请求: {request.url}")

            # 获取请求数据
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    request_data = request.post_data_json()
                except Exception:
                    request_data = json.loads(request.post_data)
                logger.info(f"请求数据 (解析为JSON): {request_data}")
            else:
                # 对于GET请求，获取URL参数
                from urllib.parse import urlparse, parse_qs

                parsed_url = urlparse(request.url)
                request_data = parse_qs(parsed_url.query)

                # 将单项列表值转换为单个值
                for key, value in request_data.items():
                    if isinstance(value, list) and len(value) == 1:
                        request_data[key] = value[0]

                logger.info(f"请求参数: {request_data}")

            # 构建完整的请求信息
            captured_data = {
                "url": request.url,
                "method": request.method,
                "data": request_data,
                "headers": {k: v for k, v in request.headers.items()},
            }

            # 验证参数（如果需要）
            if assert_params and request_data:
                # 处理断言参数
                for jsonpath_expr, expected_value in assert_params.items():
                    verify_jsonpath(
                        step_executor, request_data, jsonpath_expr, expected_value
                    )

        return captured_data

    except Exception as e:
        logger.error(f"监测请求失败: {e}")
        screenshot = step_executor.page.screenshot()
        allure.attach(
            screenshot,
            name="请求捕获失败截图",
            attachment_type=allure.attachment_type.PNG,
        )
        raise


def monitor_action_response(
    step_executor,
    url_pattern: str,
    selector: str,
    action: str = "click",
    assert_params: Dict[str, Any] = None,
    save_params: Dict[str, Any] = None,
    timeout: int = None,
    **kwargs,
):
    """
    监测操作触发的响应并验证参数

    Args:
        step_executor: StepExecutor实例
        url_pattern: URL匹配模式，如 "**/api/user/**"
        selector: 要操作的元素选择器
        action: 要执行的操作，如 "click", "fill" 等
        assert_params: 要验证的参数列表，格式为 [{"$.path.to.field": expected_value}, ...]
        save_params: 要保存的参数列表，格式为 [{"$.path.to.field": viable_name}, ...]
        timeout: 等待超时时间(毫秒)
        **kwargs: 其他操作参数，如 fill 操作的 value

    Returns:
        捕获的响应数据
    """
    from ai_playwright.constants import DEFAULT_TIMEOUT

    if timeout is None:
        timeout = DEFAULT_TIMEOUT

    logger.info(f"开始监测响应: {url_pattern}, 操作: {action} 元素: {selector}")

    try:
        with step_executor.page.expect_response(
            url_pattern, timeout=timeout
        ) as response_info:
            _execute_monitored_action(step_executor, action, selector, **kwargs)

            # 等待响应完成
            response = response_info.value
            logger.info(f"捕获到响应: {response.url}, 状态码: {response.status}")

            # 获取响应数据
            try:
                response_data = response.json()
                logger.info(f"响应数据: {response_data}")

                # 验证参数（如果需要）
                if response_data:
                    if assert_params:
                        # 处理断言参数
                        for jsonpath_expr, expected_value in assert_params.items():
                            verify_jsonpath(
                                step_executor,
                                response_data,
                                jsonpath_expr,
                                expected_value,
                            )

                    if save_params:
                        # 处理保存参数
                        for jsonpath_expr, viable_name in save_params.items():
                            save_jsonpath(
                                step_executor, response_data, jsonpath_expr, viable_name
                            )
                return response_data

            except Exception as e:
                logger.error(f"处理响应数据失败: {e}")
                raise

    except Exception as e:
        logger.error(f"监测响应失败: {e}")
        screenshot = step_executor.page.screenshot()
        allure.attach(
            screenshot,
            name="响应捕获失败截图",
            attachment_type=allure.attachment_type.PNG,
        )
        raise


def save_jsonpath(step_executor, data, jsonpath_expr, viable_name):
    """
    保存JSONPath表达式的值到变量

    Args:
        step_executor: StepExecutor实例
        data: 要保存的数据
        jsonpath_expr: JSONPath表达式
        viable_name: 变量名称
    """
    # 解析 jsonpath 表达式
    jsonpath_expr = jsonpath_expr.strip()
    expr = parse(jsonpath_expr)
    logger.debug(f"JSONPath表达式: {jsonpath_expr}")
    logger.debug(f"变量名称: {viable_name}")

    # 查找匹配的值
    matches = [value.value for value in expr.find(data)]
    if matches:
        value = matches[0]
        logger.debug(f"匹配的值: {value}")
        step_executor.variable_manager.set_variable(viable_name, value)
    else:
        logger.warning(f"JSONPath {jsonpath_expr} 未找到匹配项")


def verify_jsonpath(step_executor, data, jsonpath_expr, expected_value):
    """
    验证JSONPath表达式的值是否符合预期

    Args:
        step_executor: StepExecutor实例
        data: 要验证的数据
        jsonpath_expr: JSONPath表达式
        expected_value: 期望值
    """
    from pytest_check import check

    # 解析 jsonpath 表达式
    jsonpath_expr = jsonpath_expr.strip()
    expr = parse(jsonpath_expr)

    # 查找匹配的值
    matches = [value.value for value in expr.find(data)]
    if not matches:
        logger.error(f"JSONPath {jsonpath_expr} 未找到匹配项")
        raise ValueError(f"JSONPath {jsonpath_expr} 未找到匹配项，当前数据: {data}")

    matches_value = matches[0]

    expected_value = step_executor.variable_manager.replace_variables_refactored(
        expected_value
    )

    # 执行断言
    with check, allure.step(f"验证参数 {jsonpath_expr}"):
        if isinstance(matches_value, list) and isinstance(expected_value, list):
            # 列表比较
            assert sorted([str(x) for x in matches_value]) == sorted(
                [str(x) for x in expected_value]
            ), f"断言失败: 参数 {jsonpath_expr} 期望值为 '{expected_value}', 实际值为 '{matches_value}'"
        elif isinstance(matches_value, list):
            # 检查列表中是否包含期望值
            expected_str = str(expected_value)
            found = any(str(item) == expected_str for item in matches_value)
            assert (
                found
            ), f"断言失败: 参数 {jsonpath_expr} 期望包含值 '{expected_value}', 实际值为 '{matches_value}'"
        else:
            # 单值比较
            assert str(matches_value) == str(
                expected_value
            ), f"断言失败: 参数 {jsonpath_expr} 期望值为 '{expected_value}', 实际值为 '{matches_value}'"

    allure.attach(
        f"断言成功: 参数 {jsonpath_expr} 匹配期望值 {expected_value}",
        name="断言结果",
        attachment_type=allure.attachment_type.TEXT,
    )

    logger.info(f"参数验证成功: {jsonpath_expr} 匹配期望值 {expected_value}")
