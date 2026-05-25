"""
处理模块相关的操作
"""

from typing import Dict, Any, List

import allure

from src.case_utils import load_modules
from utils.logger import logger


def _replace_module_params(
    steps: List[Dict[str, Any]], params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    替换模块步骤中的参数

    Args:
        steps: 模块步骤列表
        params: 参数字典

    Returns:
        处理后的步骤列表
    """
    import copy

    processed_steps = copy.deepcopy(steps)

    def replace_in_value(value):
        if isinstance(value, str):
            # 替换参数引用，格式为 ${param_name}
            for param_name, param_value in params.items():
                placeholder = "${" + param_name + "}"
                if placeholder in value:
                    value = value.replace(placeholder, str(param_value))
        return value

    def process_step_dict(step_dict):
        for key, value in step_dict.items():
            if isinstance(value, dict):
                process_step_dict(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        process_step_dict(item)
                    else:
                        value[i] = replace_in_value(item)
            else:
                step_dict[key] = replace_in_value(value)

    for step in processed_steps:
        process_step_dict(step)

    return processed_steps


def find_module(module_name: str) -> Dict[str, Any]:
    """
    查找并加载模块数据

    Args:
        module_name: 模块名称
    Returns:
        模块数据

    Raises:
        ValueError: 如果找不到模块
    """
    all_modules = load_modules()

    # 检查是否有匹配的模块名
    if module_name in all_modules:
        return {module_name: all_modules[module_name]}

    # 如果所有尝试都失败，抛出错误
    raise ValueError(f"找不到模块: {module_name}")


def execute_module(step_executor, step: Dict[str, Any]) -> None:
    """
    执行模块引用

    Args:
        step_executor: StepExecutor实例
        step: 包含use_module字段的步骤
    """
    module_name = step["use_module"]
    params = step.get("params", {})
    description = step.get("description", f"执行模块 {module_name}")

    logger.info(f"开始执行模块: {module_name} {description}")

    # 处理参数中的变量
    processed_params = {}
    for key, value in params.items():
        processed_params[key] = step_executor._replace_variables(value)

    # 加载模块步骤
    try:
        # 尝试从缓存中获取模块
        if step_executor.modules_cache.get(module_name):
            module_data = step_executor.modules_cache[module_name]
        else:
            # 获取模块路径和数据
            module_data = find_module(module_name)

            # 缓存模块数据
            step_executor.modules_cache[module_name] = module_data

        # 获取步骤列表
        if "steps" in module_data:
            steps = module_data["steps"]
        elif module_data:
            # 获取第一个键对应的值
            first_key = next(iter(module_data))
            steps = module_data[first_key]
        else:
            raise ValueError(f"模块 '{module_name}' 中没有找到步骤")

        # 替换参数
        processed_steps = _replace_module_params(steps, processed_params)

        # 执行模块步骤
        with allure.step(f"执行模块: {module_name}"):
            for module_step in processed_steps:
                step_executor.execute_step(module_step)

        logger.info(f"模块 '{module_name}' 执行完成")
    except Exception as e:
        logger.error(f"执行模块 '{module_name}' 失败: {e}")
        raise
