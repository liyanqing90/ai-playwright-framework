"""
存储相关的命令
"""

from typing import Dict, Any

from src.step_actions.action_types import StepAction
from src.step_actions.commands.base_command import Command, CommandFactory
from src.step_actions.expression_evaluator import evaluate_math_expression
from utils.logger import logger


@CommandFactory.register(StepAction.STORE_VARIABLE)
class StoreVariableCommand(Command):
    """存储变量命令，支持表达式计算"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        var_name = step.get("name", "temp_var")
        var_value = step.get("value")
        scope = step.get("scope", "global")
        expression = step.get("expression")

        # 如果提供了表达式，则计算表达式的值
        if expression:
            try:
                var_value = evaluate_math_expression(
                    expression, ui_helper.variable_manager
                )
                logger.info(f"计算表达式: {expression} = {var_value}")
            except Exception as e:
                logger.error(f"计算表达式错误: {expression} - {e}")
                raise

        # 存储变量
        ui_helper.variable_manager.set_variable(var_name, var_value, scope)
        logger.info(f"已存储变量 {var_name}={var_value} (scope={scope})")


@CommandFactory.register(StepAction.STORE_TEXT)
class StoreTextCommand(Command):
    """存储文本命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        var_name = step.get("variable_name", "text_var")
        scope = step.get("scope", "global")
        # 获取元素文本
        text = ui_helper.get_text(selector=selector)
        # 存储文本
        ui_helper.variable_manager.set_variable(var_name, text, scope)
        logger.info(f"已存储元素文本 {var_name}={text} (scope={scope})")


@CommandFactory.register(StepAction.STORE_ATTRIBUTE)
class StoreAttributeCommand(Command):
    """存储属性命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        var_name = step.get("variable_name", "attr_var")
        attribute = step.get("attribute")
        scope = step.get("scope", "global")
        # 获取元素属性
        attr_value = ui_helper.get_element_attribute(
            selector=selector, attribute=attribute
        )
        # 存储属性
        ui_helper.variable_manager.set_variable(var_name, attr_value, scope)
        logger.info(f"已存储元素属性 {var_name}={attr_value} (scope={scope})")


@CommandFactory.register(StepAction.STORE_INPUT_VALUE)
class StoreInputValueCommand(Command):
    """存储输入框值命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        var_name = step.get("variable_name", "input_value_var")
        scope = step.get("scope", "global")
        # 获取输入框的value值
        input_value = ui_helper.get_value(selector=selector)
        # 存储输入框值
        ui_helper.variable_manager.set_variable(var_name, input_value, scope)
        logger.info(f"已存储输入框值 {var_name}={input_value} (scope={scope})")


@CommandFactory.register(StepAction.SAVE_ELEMENT_COUNT)
class SaveElementCountCommand(Command):
    """存储元素数量命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        count = ui_helper.get_element_count(selector=selector)
        if "variable_name" in step:
            ui_helper.store_variable(
                step["variable_name"], str(count), step.get("scope", "global")
            )
