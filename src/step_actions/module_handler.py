from __future__ import annotations

import copy
from typing import Any

import allure

from src.case_utils import load_modules
from utils.logger import logger


def _replace_module_params(
    steps: list[dict[str, Any]],
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    processed_steps = copy.deepcopy(steps)

    def replace_in_value(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        for param_name, param_value in params.items():
            value = value.replace("${" + param_name + "}", str(param_value))
        return value

    def process_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: process_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [process_value(item) for item in value]
        return replace_in_value(value)

    return [process_value(step) for step in processed_steps]


def find_module(module_name: str) -> dict[str, Any]:
    all_modules = load_modules()
    if module_name in all_modules:
        return {module_name: all_modules[module_name]}
    raise ValueError(f"找不到模块: {module_name}")


def execute_module(step_executor, step: dict[str, Any]) -> None:
    module_name = step["use_module"]
    params = step.get("params", {})
    description = step.get("description", f"执行模块 {module_name}")
    step.pop("_module_executed_steps", None)

    logger.info(f"开始执行模块: {module_name} {description}")

    processed_params = {
        key: step_executor._replace_variables(value) for key, value in params.items()
    }

    try:
        # modules.yaml is the source of truth. Do not cache expanded module steps.
        module_data = find_module(module_name)

        if "steps" in module_data:
            steps = module_data["steps"]
        elif module_data:
            first_key = next(iter(module_data))
            steps = module_data[first_key]
        else:
            raise ValueError(f"模块 '{module_name}' 中没有找到步骤")

        processed_steps = _replace_module_params(steps, processed_params)

        with allure.step(f"执行模块: {module_name}"):
            executed_steps: list[dict[str, Any]] = []
            for module_step in processed_steps:
                step_executor.execute_step(module_step)
                executed_steps.append(copy.deepcopy(module_step))
        step["_module_executed_steps"] = executed_steps

        logger.info(f"模块 '{module_name}' 执行完成")
    except Exception as exc:
        logger.error(f"执行模块 '{module_name}' 失败: {exc}")
        raise
