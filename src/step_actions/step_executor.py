"""
步骤执行器的核心实现
"""

from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, Any, List

import allure

from constants import DEFAULT_TIMEOUT
from src.ai_runtime.config import load_ai_config, runtime_mode
from src.ai_runtime.smart_resolver import SmartResolver
from src.step_actions.action_types import StepAction

# 导入命令模式执行器
from src.step_actions.command_executor import execute_action_with_command
from src.step_actions.flow_control import (
    execute_condition,
    execute_loop,
    evaluate_expression,
)
from src.step_actions.module_handler import execute_module
from utils.logger import logger
from utils.variable_manager import VariableManager


# 导入所有命令类


class StepExecutor:

    def __init__(
        self,
        page,
        ui_helper,
        elements: Dict[str, Any],
        default_mode: str | None = None,
    ):
        self.has_error = None
        self.page = page
        self.ui_helper = ui_helper
        self.elements = elements or {}
        self.default_mode = default_mode
        self.ai_config = load_ai_config()
        self.smart_resolver = None
        self.start_time = None
        self.step_has_error = False  # 步骤错误状态
        self._log_buffer = StringIO()  # 步骤日志缓存
        self._buffer_handler_id = None
        self._prepare_evidence_dir()
        self._VALID_ACTIONS = {
            a.lower()
            for attr in dir(StepAction)
            if isinstance((alist := getattr(StepAction, attr)), list)
            for a in alist
        }

        self._NO_SELECTOR_ACTIONS = {a.lower() for a in StepAction.NO_SELECTOR_ACTIONS}

        # 初始化变量管理器
        self.variable_manager = VariableManager()

        # 初始化项目名称
        self.project_name = None

        # 已加载的模块缓存
        self.modules_cache = {}

    @staticmethod
    def _prepare_evidence_dir():
        """创建截图存储目录"""
        Path("./evidence/screenshots").mkdir(parents=True, exist_ok=True)

    def setup(self, elements: Dict[str, Any] = None):
        """设置元素定义，在测试开始前调用"""
        if elements:
            self.elements = elements

    def execute_steps(
        self, steps: List[Dict[str, Any]], project_name: str = None
    ) -> None:
        """
        执行多个测试步骤

        Args:
            steps: 测试步骤列表
            project_name: 项目名称，用于加载模块
        """
        self.project_name = project_name
        for step in steps:
            self.execute_step(step)

    def execute_step(self, step: Dict[str, Any]) -> None:
        try:
            self.start_time = datetime.now()
            self.step_has_error = False

            # 检查是否为流程控制步骤
            if "use_module" in step:
                execute_module(self, step)
                return
            elif "if" in step:
                execute_condition(self, step)
                return
            elif "for_each" in step:
                execute_loop(self, step)
                return

            action = step.get("action", "").lower()
            if action in {"observe", "ai操作", "智能操作"}:
                self._execute_observe(step)
                return

            mode = self._resolve_mode(step)
            pre_selector = step.get("selector")
            raw_selector = self.elements.get(pre_selector, pre_selector)
            selector = self.variable_manager.replace_variables_refactored(raw_selector)
            target = self.variable_manager.replace_variables_refactored(
                step.get("target")
            )
            if mode != "strict" and not target and pre_selector not in self.elements:
                target = self.variable_manager.replace_variables_refactored(
                    pre_selector
                )
            selector = self._resolve_selector(action, selector, target, mode, step)
            selector = self._apply_nth(selector, step)
            value = self.variable_manager.replace_variables_refactored(
                step.get("value")
            )  # 替换变量
            self._log_step_execution(action=action, selector=selector, value=value)
            self._log_ai_mode(
                mode=mode,
                action=action,
                target=target,
                selector=selector,
                step=step,
            )
            self._validate_step(action, selector, step)
            self._execute_action(action, selector, value, step)
        except AssertionError as e:
            self.has_error = True
            self.step_has_error = True
            # 检查是否为硬断言异常
            if hasattr(e, "_hard_assert") and getattr(e, "_hard_assert", False):
                logger.error(f"硬断言失败，终止测试执行: {e}")
                raise  # 硬断言失败时重新抛出异常，终止测试执行
            # 软断言失败时不抛出异常，继续执行
        except Exception as e:
            logger.error(f"步骤执行失败: {e}")
            self.has_error = True
            self.step_has_error = True
            raise e
        finally:
            self._finalize_step()

    def _validate_step(self, action, selector, step: Dict[str, Any] | None = None) -> None:
        if not action:
            raise ValueError("步骤缺少必要参数: action", f"原始输入: {action}")
        # 操作类型白名单校验
        if action not in self._VALID_ACTIONS:
            raise ValueError(f"不支持的操作类型: {action}")
        # 必要参数校验
        has_coordinate = bool((step or {}).get("_resolved_coordinate"))
        if action not in self._NO_SELECTOR_ACTIONS and not selector and not has_coordinate:
            raise ValueError(f"操作 {action} 需要提供selector参数")

    def _execute_action(
        self, action: str, selector: str, value: Any = None, step: Dict[str, Any] = None
    ) -> None:
        """执行具体操作，使用命令模式"""
        if step and step.get("_resolved_coordinate") and not selector:
            self._execute_coordinate_action(
                action, step["_resolved_coordinate"], value, step
            )
            return
        try:
            execute_action_with_command(self.ui_helper, action, selector, value, step)
        except AssertionError as e:
            # 标记异常为断言失败
            self.step_has_error = True
            raise e
        except Exception as e:
            # 统一处理所有异常，不区分超时与非超时
            # 标记步骤失败
            self.step_has_error = True
            self.has_error = True

            # 只添加必要的错误信息，不进行额外处理
            if not hasattr(e, "_logged"):
                logger.error(f"步骤执行失败: {e}")
                setattr(e, "_logged", True)

            # 添加关键信息
            setattr(e, "_action", action)
            setattr(e, "_selector", selector)
            setattr(e, "_value", value)

            # 直接抛出异常，不做其他任何处理
            raise

    def _execute_coordinate_action(
        self,
        action: str,
        coordinate: tuple[float, float] | list[float],
        value: Any = None,
        step: Dict[str, Any] | None = None,
    ) -> None:
        x, y = float(coordinate[0]), float(coordinate[1])
        logger.debug(f"UI Vision坐标兜底执行: {action} | x={x} | y={y}")
        if action == "click":
            self.page.mouse.click(x, y)
            return
        if action == "fill":
            self.page.mouse.click(x, y)
            self.page.keyboard.press("Control+A")
            self.page.keyboard.type(str(value or ""))
            return
        if action in {"press", "press_key"}:
            self.page.mouse.click(x, y)
            key = (step or {}).get("key") or value or "Enter"
            self.page.keyboard.press(str(key))
            return
        raise ValueError(f"当前action不支持UI Vision坐标兜底: {action}")

    def _resolve_mode(self, step: Dict[str, Any]) -> str:
        config_default = self.ai_config.get("runtime", {}).get("default_mode", "strict")
        mode = step.get("mode") or self.default_mode or runtime_mode(config_default)
        mode = str(mode or "strict").lower()
        if mode not in {"strict", "smart", "ai"}:
            raise ValueError(f"不支持的AI执行模式: {mode}")
        return mode

    def _resolve_selector(
        self,
        action: str,
        selector: str | None,
        target: str | None,
        mode: str,
        step: Dict[str, Any],
    ) -> str | None:
        if action in self._NO_SELECTOR_ACTIONS:
            return selector
        if mode == "strict":
            return selector
        resolver = self._get_smart_resolver()
        resolved = resolver.resolve(
            action=action,
            target=target,
            selector=selector,
            mode=mode,
            timeout=int(step.get("timeout", DEFAULT_TIMEOUT)),
        )
        step["_resolved_selector_source"] = resolved.source
        step["_resolved_by_ai"] = resolved.ai_called
        step["_resolved_confidence"] = resolved.confidence
        step["_resolved_prompt_version"] = resolved.prompt_version
        step["_resolved_schema_version"] = resolved.schema_version
        step["_resolved_model"] = resolved.model
        step["_resolved_candidate_count"] = resolved.candidate_count
        step["_resolved_candidate_hash"] = resolved.candidate_hash
        step["_resolved_coordinate"] = resolved.coordinate
        step["_resolved_vision_method"] = resolved.vision_method
        step["_resolved_vision_reason"] = resolved.vision_reason
        return resolved.selector

    @staticmethod
    def _apply_nth(selector: str | None, step: Dict[str, Any]) -> str | None:
        if selector is None or "nth" not in step:
            return selector
        nth = step["nth"]
        if nth is None or ">> nth=" in str(selector):
            return selector
        return f"{selector} >> nth={int(nth)}"

    def _execute_observe(self, step: Dict[str, Any]) -> None:
        instruction = self.variable_manager.replace_variables_refactored(
            step.get("instruction") or step.get("value") or step.get("target")
        )
        if not instruction:
            raise ValueError("observe步骤需要instruction、value或target")
        timeout = int(step.get("timeout", DEFAULT_TIMEOUT))
        resolver = self._get_smart_resolver()
        operation = resolver.observe_operation(instruction=instruction, timeout=timeout)
        value = operation.value or operation.key or operation.wait_ms
        self._log_step_execution(
            action=operation.action, selector=operation.selector, value=value
        )
        parts = [
            "AI执行模式: observe",
            "定位来源: AI观察",
            f"指令: {instruction}",
            f"解析动作: {operation.action}",
        ]
        parts.extend(
            self._ai_metadata_parts(
                {
                    "_resolved_prompt_version": operation.prompt_version,
                    "_resolved_schema_version": operation.schema_version,
                    "_resolved_model": operation.model,
                    "_resolved_candidate_count": operation.candidate_count,
                    "_resolved_candidate_hash": operation.candidate_hash,
                }
            )
        )
        logger.debug(" | ".join(parts))
        resolver.apply_operation(operation, timeout=timeout)

    def _get_smart_resolver(self) -> SmartResolver:
        if self.smart_resolver is None:
            self.smart_resolver = SmartResolver(self.page)
        return self.smart_resolver

    @staticmethod
    def _log_step_execution(action: str, selector: Any, value: Any) -> None:
        logger.debug(f"执行步骤: {action} | 选择器: {selector} | 值: {value}")

    def _log_ai_mode(
        self,
        *,
        mode: str,
        action: str,
        target: Any,
        selector: Any,
        step: Dict[str, Any],
    ) -> None:
        if mode == "strict":
            return
        source = step.get("_resolved_selector_source")
        ai_called = "是" if step.get("_resolved_by_ai") else "否"
        if action in self._NO_SELECTOR_ACTIONS:
            logger.debug(f"AI执行模式: {mode} | 定位来源: 无需定位")
            return
        parts = [
            f"AI执行模式: {mode}",
            f"定位来源: {self._source_label(source)}",
        ]
        if target:
            parts.append(f"目标: {target}")
        parts.extend([f"选择器: {selector}", f"AI兜底: {ai_called}"])
        parts.extend(self._ai_metadata_parts(step))
        logger.debug(" | ".join(parts))

    @staticmethod
    def _ai_metadata_parts(source: Dict[str, Any]) -> List[str]:
        metadata = [
            ("confidence", "_resolved_confidence"),
            ("prompt_version", "_resolved_prompt_version"),
            ("schema_version", "_resolved_schema_version"),
            ("model", "_resolved_model"),
            ("candidate_count", "_resolved_candidate_count"),
            ("candidate_hash", "_resolved_candidate_hash"),
            ("coordinate", "_resolved_coordinate"),
            ("vision_method", "_resolved_vision_method"),
            ("vision_reason", "_resolved_vision_reason"),
        ]
        parts: List[str] = []
        for label, key in metadata:
            value = source.get(key)
            if value is not None:
                parts.append(f"{label}: {value}")
        return parts

    @staticmethod
    def _source_label(source: Any) -> str:
        labels = {
            "explicit": "显式选择器",
            "registry": "历史定位",
            "heuristic": "规则定位",
            "ai_observe": "AI定位",
            "vision_dom": "UI Vision DOM兜底",
            "vision_coordinate": "UI Vision坐标兜底",
        }
        return labels.get(str(source or ""), str(source or "未知"))

    def _replace_variables(self, value: Any) -> Any:
        """
        替换值中的变量引用

        Args:
            value: 原始值，可能包含变量引用 ${var_name} 或 $<var_name> 或 $[[expression]]

        Returns:
            替换后的值
        """
        if value is None:
            return value

        if isinstance(value, (int, float, bool)):
            return value

        if isinstance(value, str):
            # 处理数学表达式引用，如 $[[1 + 2 * ${var}]]
            if (
                value.startswith("$[[")
                and value.endswith("]]")
                and value.count("$[[") == 1
            ):
                try:
                    from src.step_actions.expression_evaluator import (
                        evaluate_math_expression,
                    )

                    # 提取表达式内容
                    expr = value[3:-2].strip()
                    # 计算表达式
                    result = evaluate_math_expression(expr, self.variable_manager)
                    return result
                except Exception as e:
                    logger.error(f"计算表达式错误: {value} - {e}")
                    raise

            # 处理完整的变量引用，如 ${var_name} 或 $<var_name>
            if (
                value.startswith("${")
                and value.endswith("}")
                and value.count("${") == 1
            ) or (
                value.startswith("$<")
                and value.endswith(">")
                and value.count("$<") == 1
            ):

                if value.startswith("${"):
                    var_name = value[2:-1]
                else:  # value.startswith("$<")
                    var_name = value[2:-1]

                return self.variable_manager.get_variable(var_name)

            # 替换内嵌变量引用
            import re

            # 同时匹配 ${var_name} 和 $<var_name> 两种模式
            pattern = r"\${([^{}]+)}|\$<([^<>]+)>"

            def replace_var(match):
                # 获取匹配的组，第一个组是 ${} 形式，第二个组是 $<> 形式
                var_name = (
                    match.group(1) if match.group(1) is not None else match.group(2)
                )
                var_value = self.variable_manager.get_variable(var_name)
                return str(var_value) if var_value is not None else match.group(0)

            # 使用正则表达式替换所有变量引用
            result = re.sub(pattern, replace_var, value)

            # 处理内嵌的数学表达式引用，如 "Total: $[[1 + 2 * ${var}]]"
            pattern_expr = r"\$\[\[([^\[\]]+)\]\]"

            def replace_expr(match):
                try:
                    from src.step_actions.expression_evaluator import (
                        evaluate_math_expression,
                    )

                    # 提取表达式内容
                    expr = match.group(1).strip()
                    # 计算表达式
                    result = evaluate_math_expression(expr, self.variable_manager)
                    return str(result)
                except Exception as e:
                    logger.error(f"计算表达式错误: {match.group(0)} - {e}")
                    raise

            # 替换所有内嵌的数学表达式
            result = re.sub(pattern_expr, replace_expr, result)

            return result

        if isinstance(value, list):
            return [self._replace_variables(item) for item in value]

        if isinstance(value, dict):
            return {k: self._replace_variables(v) for k, v in value.items()}

        return value

    def _evaluate_expression(self, expression: str) -> bool:
        """
        计算表达式的值

        Args:
            expression: 表达式字符串，如 "${{ ${count} > 5 }}"

        Returns:
            表达式的布尔结果
        """
        return evaluate_expression(self, expression)

    def _finalize_step(self):
        """统一后处理逻辑"""
        # 移除日志handler
        if self._buffer_handler_id:
            logger.remove(self._buffer_handler_id)
            self._buffer_handler_id = None

        # 记录耗时
        self._log_step_duration()

        # 失败时采集证据
        if self.step_has_error:
            self._capture_failure_evidence()

    def _log_step_duration(self):
        """统一记录步骤耗时"""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            if self.step_has_error:
                logger.error(f"[失败] 步骤耗时: {duration:.2f}s")
            else:
                logger.info(f"[成功] 步骤耗时: {duration:.2f}s")

    def _capture_failure_evidence(self):
        """统一失败证据采集"""
        try:
            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            context_info = f"URL: {self.page.url}\n错误时间: {timestamp}"
            allure.attach(
                context_info,
                name="失败上下文",
                attachment_type=allure.attachment_type.TEXT,
            )

        except Exception as e:
            logger.error(f"证据采集失败: {str(e)}")
