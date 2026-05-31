from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Optional

import pytest
import typer
from rich.table import Table

from ai_playwright.cli.common import console, normalize_yaml_file_name
from ai_playwright.utils.config import Browser, Config, Environment
from ai_playwright.utils.logger import configure_file_logger, logger
from ai_playwright.utils.token_usage import get_token_usage_tracker


app = typer.Typer(
    add_completion=False,
    help="Run project test cases.",
    pretty_exceptions_show_locals=False,
)


def build_pytest_args(config: Config) -> list[str]:
    test_path = f"{config.test_dir}/cases"
    if config.test_file:
        test_path = f"{test_path}/{config.test_file}"

    pytest_args = [
        test_path,
        "-v",
        "--tb=line",
        "-p",
        "no:warnings",
        "-s",
        "--alluredir=reports/allure-results",
        "--clean-alluredir",
        "-p",
        "ai_playwright.pytest_plugin",
    ]
    if config.marker:
        pytest_args.extend(["-m", config.marker])
    if config.keyword:
        pytest_args.extend(["-k", config.keyword])
    return pytest_args


def display_run_configuration(config: Config) -> None:
    table = Table(title="测试运行配置")
    table.add_column("配置项", style="cyan")
    table.add_column("值", style="magenta")

    table.add_row("浏览器", config.browser.value)
    table.add_row("运行模式", "有头模式" if config.headed else "无头模式")
    table.add_row("运行环境", config.env.value)
    table.add_row("项目", config.project)
    if config.test_file:
        table.add_row("测试文件", config.test_file)
    if config.marker:
        table.add_row("用例标记", config.marker)
    if config.keyword:
        table.add_row("筛选关键字", config.keyword)
    if config.base_url:
        table.add_row("基础URL", config.base_url)
    console.print(table)


def show_test_summary(start_time: float) -> None:
    duration = time.time() - start_time
    table = Table(title="测试运行摘要")
    table.add_column("项目", style="cyan")
    table.add_column("值", style="magenta")
    table.add_row("运行时间", f"{duration:.2f} 秒")
    table.add_row("完成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    table.add_row("报告位置", "reports/allure-results")
    console.print(table)


@app.command()
def main(
    project: str = typer.Option("demo", "-p", "--project", help="项目"),
    test_file: Optional[str] = typer.Option(
        "", "-f", "--file", "--test-file", help="测试文件名，例如 smoke"
    ),
    ai_mode: str = typer.Option(
        "strict", "-m", "--ai-mode", help="执行模式: strict/smart"
    ),
    env: Environment = typer.Option(Environment.PROD, "-e", "--env", help="执行环境"),
    headed: bool = typer.Option(
        True, "--headed/--headless", help="是否以有头模式运行浏览器"
    ),
    browser: Browser = typer.Option(Browser.CHROMIUM, "--browser", help="浏览器"),
    keyword: Optional[str] = typer.Option(
        None, "-k", "--keyword", help="只运行匹配关键字的测试用例"
    ),
    marker: Optional[str] = typer.Option(
        None, "--marker", help="只运行特定标记的测试用例"
    ),
    base_url: Optional[str] = typer.Option("", "--base-url", help="指定基础 URL"),
):
    if ai_mode not in {"strict", "smart"}:
        raise typer.BadParameter("--ai-mode 仅支持 strict/smart")

    try:
        config = Config(
            marker=marker,
            keyword=keyword,
            headed=headed,
            browser=browser,
            env=env,
            project=project,
            base_url=base_url,
            test_file=normalize_yaml_file_name(test_file),
        )
    except Exception as exc:
        console.print(f"[red]配置加载失败: {exc}[/red]")
        raise typer.Exit(2) from exc

    config.configure_environment()
    configure_file_logger()
    os.environ["UI_AI_MODE"] = ai_mode

    display_run_configuration(config)
    start_time = time.time()
    tracker = get_token_usage_tracker()
    tracker.start_run(
        run_kind="pytest",
        metadata={
            "project": config.project,
            "env": config.env.value,
            "test_file": config.test_file,
            "ai_mode": ai_mode,
        },
    )

    try:
        pytest_args = build_pytest_args(config)
        logger.info(f"使用参数运行pytest: {' '.join(pytest_args)}")
        exit_code = pytest.main(pytest_args)
        show_test_summary(start_time)
        tracker.finish_run(
            status="passed" if exit_code == 0 else "failed",
            metadata={"exit_code": exit_code},
        )
        sys.exit(exit_code)
    except KeyboardInterrupt:
        tracker.finish_run(status="failed", metadata={"error": "KeyboardInterrupt"})
        logger.error("测试被用户中断")
        sys.exit(1)
    except Exception as exc:
        tracker.finish_run(status="failed", metadata={"error": str(exc)})
        logger.error(f"测试运行出错: {exc}")
        sys.exit(1)


def cli() -> None:
    app()


if __name__ == "__main__":
    cli()
