from typing import Dict, Any, Set

import allure

# 导入重构后的StepExecutor
from src.step_actions.step_executor import StepExecutor
from utils.logger import logger

log = logger


def _cleanup_test_environment(case: Dict[str, Any]) -> None:
    with allure.step("测试环境清理"):
        log.debug(f"Cleaning up test environment for case: {case['name']}")
        # fixture 的清理会由 pytest 自动处理


def _setup_test_environment(case: Dict[str, Any]) -> None:
    with allure.step("测试环境准备"):
        log.debug(f"Setting up test environment for case: {case['name']}")
        # 添加环境准备代码


class CaseExecutor:
    def __init__(
        self,
        case_data: Dict[str, Any],
        elements: Dict[str, Any],
        case_metadata: Dict[str, Any] | None = None,
    ):
        self.case_data = case_data
        self.elements = elements
        self.case_metadata = case_metadata or {}
        self.executed_fixtures: Set[str] = set()

    def execute_test_case(self, page, ui_helper) -> None:
        """执行测试用例
        Args:
            page: Playwright页面对象
            ui_helper: UI操作帮助类
        """
        # 执行测试步骤
        step_executor = StepExecutor(
            page,
            ui_helper,
            self.elements,
            default_mode=self._default_mode(),
        )

        # 支持两种数据结构：直接的步骤列表或包含步骤的字典
        if isinstance(self.case_data, list):
            # 如果是列表，取第一个元素（兼容旧格式）
            if self.case_data and isinstance(self.case_data[0], dict):
                steps = self.case_data[0].get("steps", [])
            else:
                steps = []
        elif isinstance(self.case_data, dict):
            # 如果是字典，直接获取steps
            steps = self.case_data.get("steps", [])
        else:
            steps = []

        # 执行所有步骤
        for step in steps:
            step_executor.execute_step(step)

    def _default_mode(self) -> str | None:
        if isinstance(self.case_data, dict) and self.case_data.get("mode"):
            return self.case_data.get("mode")
        return self.case_metadata.get("mode")
