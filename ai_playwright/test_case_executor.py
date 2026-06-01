from typing import Dict, Any, Set

import allure

from ai_playwright.ai_runtime.agent_case_executor import AgentCaseExecutor
from ai_playwright.ai_generation.pipeline import execute_compiled_payload_steps

# 导入重构后的StepExecutor
from ai_playwright.step_actions.step_executor import StepExecutor
from ai_playwright.utils.logger import logger

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
        if self._is_agent_case():
            self.execute_agent_case(page, ui_helper)
            return
        steps, default_mode = self.resolve_steps_and_mode()
        self.execute_resolved_steps(page, ui_helper, steps, default_mode)

    def execute_resolved_steps(
        self,
        page,
        ui_helper,
        steps: list[dict[str, Any]],
        default_mode: str | None,
    ) -> None:
        step_executor = StepExecutor(
            page,
            ui_helper,
            self.elements,
            default_mode=default_mode,
        )

        # 执行所有步骤
        execute_compiled_payload_steps(
            step_executor=step_executor,
            payload={},
            case_name=self._case_name(default="anonymous_case"),
            steps=steps,
            elements=self.elements,
            source="static_case",
        )

    def resolve_steps_and_mode(self) -> tuple[list[dict[str, Any]], str | None]:
        # 支持列表和对象两种数据来源，由 data 层 schema 保证用例形态。
        if isinstance(self.case_data, list):
            if self.case_data and isinstance(self.case_data[0], dict):
                steps = self.case_data[0].get("steps", [])
            else:
                steps = []
            return steps, self._default_mode()
        elif isinstance(self.case_data, dict):
            if str(self.case_data.get("type") or "").lower() == "agent_case":
                raise ValueError(
                    "agent_case 不会编译为静态steps，请通过AgentCaseExecutor执行"
                )
            if str(self.case_data.get("type") or "").lower() == "ai_case":
                raise ValueError(
                    "run_case 不再支持运行时 ai_case 编译；请使用 gen 生成用例，或改用 agent_case"
                )
            # 如果是字典，直接获取steps
            steps = self.case_data.get("steps", [])
            return steps, self._default_mode()
        else:
            return [], self._default_mode()

    def execute_agent_case(self, page, ui_helper) -> None:
        case_name = self._case_name(default="anonymous_agent_case")
        result = AgentCaseExecutor(
            page=page,
            ui_helper=ui_helper,
            elements=self.elements,
        ).run(
            case_name=case_name,
            case_data=self.case_data,
        )
        log.info(
            "Agent用例执行结果: "
            f"case={case_name} | steps_executed={result.steps_executed} "
            f"| model_calls={result.model_calls} "
            f"| final_reason={result.final_reason}"
        )

    def _default_mode(self) -> str | None:
        if isinstance(self.case_data, dict) and self.case_data.get("mode"):
            return self.case_data.get("mode")
        return self.case_metadata.get("mode")

    def _case_name(self, *, default: str) -> str:
        return str(
            self.case_metadata.get("name")
            or (
                self.case_data.get("name") if isinstance(self.case_data, dict) else None
            )
            or default
        )

    def _is_agent_case(self) -> bool:
        return (
            isinstance(self.case_data, dict)
            and str(self.case_data.get("type") or "").lower() == "agent_case"
        )
