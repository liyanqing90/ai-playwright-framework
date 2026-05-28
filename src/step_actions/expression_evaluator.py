"""
表达式计算器，用于计算数学表达式
"""

import math
from typing import Any

from src.step_actions.flow_control import preprocess_expression
from src.step_actions.safe_expression import safe_eval_expression
from utils.logger import logger


def evaluate_math_expression(expression: str, variable_manager) -> Any:
    """
    计算数学表达式的值

    Args:
        expression: 表达式字符串，如 "${num1} + ${num2} * 2"
        variable_manager: 变量管理器实例，用于替换表达式中的变量

    Returns:
        表达式的计算结果
    """
    try:
        # 如果表达式不是${{...}}格式，则包装为该格式
        if not (expression.startswith("${{") and expression.endswith("}}")):
            expression = f"${{{{{expression}}}}}"

        # 先替换变量
        expr_content = expression[3:-2].strip()
        expr_content = variable_manager.replace_variables_refactored(expr_content)

        # 定义安全的数学函数集合
        safe_math_functions = {
            # 基本数学函数
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            # 数学模块函数
            "sqrt": math.sqrt,
            "pow": math.pow,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "floor": math.floor,
            "ceil": math.ceil,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            # 常量
            "pi": math.pi,
            "e": math.e,
            # 类型转换
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
        }

        # 预处理表达式
        processed_expr = preprocess_expression(expr_content)

        result = safe_eval_expression(processed_expr, safe_math_functions)
        logger.debug(f"计算表达式: {expr_content} = {result}")
        return result
    except Exception as e:
        logger.error(f"计算表达式错误: {expression} - {e}")
        raise
