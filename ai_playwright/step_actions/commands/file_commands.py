"""
文件相关的命令
"""

from typing import Dict, Any

from ai_playwright.constants import DEFAULT_TIMEOUT
from ai_playwright.step_actions.action_types import StepAction
from ai_playwright.step_actions.commands.base_command import Command, CommandFactory


@CommandFactory.register(StepAction.DOWNLOAD_FILE)
class DownloadFileCommand(Command):
    """下载文件命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        save_path = step.get("save_path")
        file_path = ui_helper.download_file(selector=selector, save_path=save_path)
        if "variable_name" in step:
            ui_helper.store_variable(
                step["variable_name"], file_path, step.get("scope", "global")
            )


@CommandFactory.register(StepAction.DOWNLOAD_VERIFY)
class DownloadVerifyCommand(Command):
    """验证下载命令"""

    def execute(
        self, ui_helper, selector: str, value: Any, step: Dict[str, Any]
    ) -> None:
        file_pattern = step.get("file_pattern", value)
        timeout = int(step.get("timeout", DEFAULT_TIMEOUT))
        result = ui_helper.verify_download(file_pattern=file_pattern, timeout=timeout)
        if "variable_name" in step:
            ui_helper.store_variable(
                step["variable_name"], str(result), step.get("scope", "global")
            )
