"""
处理流程控制相关的操作（条件和循环）
"""

import json
from typing import Dict, Any

import allure

from utils.logger import logger


def execute_condition(step_executor, step: Dict[str, Any]) -> None:
    """
    执行条件分支

    Args:
        step_executor: StepExecutor实例
        step: 包含if字段的步骤
    """
    condition = step["if"]
    then_steps = step.get("then", [])
    else_steps = step.get("else", [])
    description = step.get("description", "条件分支")

    # 计算条件表达式
    # 先获取原始表达式内容用于日志
    original_condition = condition

    # 提取表达式内容（如果是${{...}}格式）
    if condition.startswith("${{") and condition.endswith("}}"):
        expr_content = condition[3:-2].strip()
        # 替换变量得到可读的表达式
        readable_expr = step_executor._replace_variables(expr_content)
    else:
        readable_expr = step_executor._replace_variables(condition)

    # 计算条件结果
    condition_result = step_executor._evaluate_expression(condition)

    with allure.step(f"条件分支: {description} ({readable_expr} = {condition_result})"):
        if condition_result:
            logger.info(f"条件 '{readable_expr}' 为真，执行THEN分支")
            for then_step in then_steps:
                step_executor.execute_step(then_step)
        else:
            logger.info(f"条件 '{readable_expr}' 为假，执行ELSE分支")
            for else_step in else_steps:
                step_executor.execute_step(else_step)


def execute_loop(step_executor, step: Dict[str, Any]) -> None:
    """
    执行循环

    Args:
        step_executor: StepExecutor实例
        step: 包含for_each字段的步骤
    """
    items = step["for_each"]
    as_var = step.get("as", "item")
    do_steps = step.get("do", [])
    description = step.get("description", "循环")

    # 处理循环项，可能是变量引用或直接值
    if isinstance(items, str) and items.startswith("${") and items.endswith("}"):
        var_name = items[2:-1]
        items_value = step_executor.variable_manager.get_variable(var_name)
    else:
        items_value = items

    # 确保循环项是可迭代的
    if not isinstance(items_value, (list, tuple, dict)):
        if isinstance(items_value, str):
            try:
                # 尝试解析为JSON
                items_value = json.loads(items_value)
            except json.JSONDecodeError:
                # 如果不是JSON，则转为列表
                items_value = [items_value]
        else:
            items_value = [items_value]

    # 如果是字典，则遍历键
    if isinstance(items_value, dict):
        items_value = list(items_value.keys())

    with allure.step(f"循环: {description} (迭代 {len(items_value)} 个项)"):
        for i, item in enumerate(items_value):
            logger.info(f"循环项 {i + 1}/{len(items_value)}: {item}")

            # 设置循环变量
            step_executor.variable_manager.set_variable(as_var, item, "test_case")

            # 执行循环体
            for do_step in do_steps:
                step_executor.execute_step(do_step)


def evaluate_expression(step_executor, expression: str) -> bool:
    """
    计算表达式的值，支持数学运算、比较操作和UI元素判断

    Args:
        step_executor: 步骤执行器实例
        expression: 表达式字符串，格式为 ${{ expression }}

    Returns:
        bool: 表达式计算结果

    支持的UI元素判断函数：
        - element_exists(selector): 检查元素是否存在
        - element_visible(selector): 检查元素是否可见
        - element_enabled(selector): 检查元素是否启用
        - element_text(selector): 获取元素文本内容
        - element_attribute(selector, attr_name): 获取元素属性值
        - element_count(selector): 获取匹配元素的数量
    """
    if not (expression.startswith("${{") and expression.endswith("}}")):
        return bool(step_executor._replace_variables(expression))

    expr_content = expression[3:-2].strip()

    # 先替换表达式中的选择器（通过elements映射）
    import re

    def replace_selector_in_expression(match):
        """替换表达式中的选择器引用，使用与step_executor相同的逻辑"""
        func_name = match.group(1)
        params_str = match.group(2)

        # 解析函数参数
        if func_name in ['element_exists', 'element_visible', 'element_enabled', 'element_text', 'element_count']:
            # 单参数函数：element_exists('selector')
            selector_match = re.match(r"'([^']+)'|\"([^\"]+)\"", params_str.strip())
            if selector_match:
                pre_selector = selector_match.group(1) or selector_match.group(2)
                # 使用与step_executor相同的替换逻辑
                selector = step_executor.variable_manager.replace_variables_refactored(
                    step_executor.elements.get(pre_selector, pre_selector)
                )
                # 转义选择器中的单引号
                escaped_selector = selector.replace("'", "\\'")
                return f"{func_name}('{escaped_selector}')"
        elif func_name == 'element_attribute':
            # 双参数函数：element_attribute('selector', 'attr')
            params = params_str.split(',', 1)
            if len(params) == 2:
                selector_match = re.match(r"'([^']+)'|\"([^\"]+)\"", params[0].strip())
                if selector_match:
                    pre_selector = selector_match.group(1) or selector_match.group(2)
                    # 使用与step_executor相同的替换逻辑
                    selector = step_executor.variable_manager.replace_variables_refactored(
                        step_executor.elements.get(pre_selector, pre_selector)
                    )
                    # 转义选择器中的单引号
                    escaped_selector = selector.replace("'", "\\'")
                    attr_part = params[1].strip()
                    return f"{func_name}('{escaped_selector}', {attr_part})"

        return match.group(0)  # 如果无法解析，返回原始内容

    # 替换表达式中的选择器引用
    pattern = r'(element_(?:exists|visible|enabled|text|attribute|count))\(([^)]+)\)'
    expr_content = re.sub(pattern, replace_selector_in_expression, expr_content)

    # 然后替换其他变量
    expr_content = step_executor._replace_variables(expr_content)

    try:
        import math
        import operator

        # 定义UI元素检查函数，直接使用ui_helper的_locator方法
        def element_exists(selector):
            """检查元素是否存在"""
            try:
                return step_executor.ui_helper._locator(selector).first.is_attached()
            except Exception as e:
                logger.debug(f"元素存在性检查失败 {selector}: {e}")
                return False

        def element_visible(selector):
            """检查元素是否可见"""
            try:
                return step_executor.ui_helper._locator(selector).first.is_visible()
            except Exception as e:
                logger.debug(f"元素可见性检查失败 {selector}: {e}")
                return False

        def element_enabled(selector):
            """检查元素是否启用"""
            try:
                return step_executor.ui_helper._locator(selector).first.is_enabled()
            except Exception as e:
                logger.debug(f"元素启用状态检查失败 {selector}: {e}")
                return False

        def element_text(selector):
            """获取元素文本内容"""
            try:
                return step_executor.ui_helper._locator(selector).first.inner_text()
            except Exception as e:
                logger.debug(f"元素文本获取失败 {selector}: {e}")
                return ""

        def element_attribute(selector, attr_name):
            """获取元素属性值"""
            try:
                return step_executor.ui_helper._locator(selector).first.get_attribute(attr_name)
            except Exception as e:
                logger.debug(f"元素属性获取失败 {selector}.{attr_name}: {e}")
                return None

        def element_count(selector):
            """获取匹配元素的数量"""
            try:
                return step_executor.ui_helper._locator(selector).count()
            except Exception as e:
                logger.debug(f"元素计数失败 {selector}: {e}")
                return 0

        # 扩展安全函数集合
        safe_math_functions = {
            # 原有数学函数
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "pow": math.pow,
            "int": int, "float": float, "str": str, "bool": bool,
            "len": len,

            # 新增UI元素检查函数
            "element_exists": element_exists,
            "element_visible": element_visible,
            "element_enabled": element_enabled,
            "element_text": element_text,
            "element_attribute": element_attribute,
            "element_count": element_count,
        }

        safe_globals = {
            "__builtins__": {},
            **safe_math_functions,
        }

        processed_expr = preprocess_expression(expr_content)
        result = eval(processed_expr, safe_globals)
        logger.debug(f"表达式计算: {expr_content} = {result}")
        return bool(result)
    except Exception as e:
        logger.error(f"表达式计算错误: {expr_content} - {e}")
        return False


def preprocess_expression(expr: str) -> str:
    operators = ["==", "!=", ">=", "<=", ">", "<"]
    for op in operators:
        if op in expr:
            # 分割表达式
            parts = expr.split(op, 1)  # 只分割一次，处理第一个操作符
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()

                # 处理左侧
                left = process_operand(left)

                # 处理右侧
                right = process_operand(right)

                # 重新组合表达式
                return f"{left} {op} {right}"

    # 处理数学运算表达式
    # 这里我们假设如果没有比较操作符，那么整个表达式就是一个数学运算
    return process_operand(expr)


def process_operand(operand: str) -> str:
    """
    处理操作数，确保字符串和数字格式正确

    Args:
        operand: 操作数字符串

    Returns:
        处理后的操作数字符串
    """
    # 去除首尾空格
    operand = operand.strip()

    # 如果已经是引号括起来的字符串，直接返回
    if (operand.startswith('"') and operand.endswith('"')) or (
        operand.startswith("'") and operand.endswith("'")
    ):
        return operand

    # 尝试解析为数字
    try:
        # 尝试解析为整数
        int(operand)
        return operand  # 是整数，直接返回
    except ValueError:
        try:
            # 尝试解析为浮点数
            float(operand)
            return operand  # 是浮点数，直接返回
        except ValueError:
            # 不是数字，也不是已引用的字符串，添加引号
            # 检查是否包含数学表达式的特殊字符
            if any(c in operand for c in "+-*/()%"):
                # 可能是复杂表达式，不添加引号
                return operand
            else:
                # 普通字符串，添加引号
                return f"'{operand}'"
