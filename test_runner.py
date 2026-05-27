import os
import sys
import time
from datetime import datetime
from typing import Optional, List

import pytest
import typer
from rich.console import Console
from rich.table import Table

from utils.config import Config, Browser, Environment, Project
from utils.logger import logger
from utils.token_usage import get_token_usage_tracker
from src.ai_generation import generate_case_files
from src.ai_runtime.provider import LLMConfigurationError

console = Console()
app = typer.Typer()


def build_pytest_args(config: Config) -> List[str]:
    """
    构建pytest运行参数

    Args:
        config: 测试配置对象

    Returns:
        pytest命令行参数列表
    """
    # 构建测试文件路径
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
    ]
    if config.marker:
        pytest_args.extend(["-m", config.marker])
    if config.keyword:
        pytest_args.extend(["-k", config.keyword])

    return pytest_args


def display_run_configuration(config: Config) -> None:
    """
    显示测试运行配置信息

    Args:
        config: 测试配置对象
    """
    table = Table(title="测试运行配置")
    table.add_column("配置项", style="cyan")
    table.add_column("值", style="magenta")

    table.add_row("浏览器", config.browser.value)
    table.add_row("运行模式", "有头模式" if config.headed else "无头模式")
    table.add_row("运行环境", config.env.value)
    table.add_row("项目", config.project.value)

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
    """
    显示测试运行摘要信息

    Args:
        start_time: 测试开始时间戳
    """
    duration = time.time() - start_time

    table = Table(title="测试运行摘要")
    table.add_column("项目", style="cyan")
    table.add_column("值", style="magenta")

    table.add_row("运行时间", f"{duration:.2f} 秒")
    table.add_row("完成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    table.add_row("报告位置", "reports/allure-results")

    # console.print(table)


def display_generation_result(result, dry_run: bool) -> None:
    table = Table(title="AI用例生成结果" if not dry_run else "AI用例生成预览")
    table.add_column("类型", style="cyan")
    table.add_column("路径", style="magenta")
    table.add_row("cases", str(result.case_file))
    table.add_row("data", str(result.data_file))
    if result.elements_file:
        table.add_row("elements", str(result.elements_file))
    if result.modules_file:
        table.add_row("modules", str(result.modules_file))
    if result.vars_file:
        table.add_row("vars", str(result.vars_file))
    console.print(table)
    for warning in result.warnings:
        console.print(f"[yellow]警告:[/yellow] {warning}")


def display_token_usage_summary(summary: dict | None) -> None:
    if not summary:
        return
    totals = summary.get("totals", {})
    table = Table(title="AI Token Usage")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Calls", str(totals.get("call_count", 0)))
    table.add_row("Input Tokens", str(totals.get("input_tokens", 0)))
    table.add_row("Output Tokens", str(totals.get("output_tokens", 0)))
    table.add_row("Total Tokens", str(totals.get("total_tokens", 0)))
    table.add_row("Cache Hit Input", str(totals.get("cached_input_tokens", 0)))
    table.add_row("Cache Miss Input", str(totals.get("uncached_input_tokens", 0)))
    table.add_row("Usage Unavailable", str(totals.get("calls_without_usage", 0)))
    table.add_row("Report", str(summary.get("summary_file", "")))
    console.print(table)


@app.command()
def main(
    marker: Optional[str] = typer.Option(
        None, "-m", "--marker", help="只运行特定标记的测试用例"
    ),
    keyword: Optional[str] = typer.Option(
        None, "-k", "--keyword", help="只运行匹配关键字的测试用例"
    ),
    headed: bool = typer.Option(True, "--headed", help="是否以有头模式运行浏览器"),
    browser: Browser = typer.Option(Browser.CHROMIUM, "--browser", help="指定浏览器"),
    env: Environment = typer.Option(Environment.PROD, "--env", help="指定环境"),
    project: Project = typer.Option(Project.DEMO, "--project", help="指定项目"),
    base_url: Optional[str] = typer.Option("", "--base-url", help="指定基础 URL"),
    test_file: Optional[str] = typer.Option("", "--test-file", help="指定测试文件"),
    no_parallel: bool = typer.Option(False, "--no-parallel", help="禁用并行执行"),
    ai_mode: str = typer.Option(
        "strict", "--ai-mode", help="执行模式: strict/smart，默认不影响历史用例"
    ),
    generate_case: Optional[str] = typer.Option(
        None, "--generate-case", help="根据YAML规格生成当前项目格式的用例"
    ),
    output_name: Optional[str] = typer.Option(
        None, "--output-name", help="生成文件名，不含扩展名"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="只预览生成结果，不写文件"),
    overwrite: bool = typer.Option(False, "--overwrite", help="允许覆盖已存在生成文件"),
    no_ai: bool = typer.Option(
        False,
        "--no-ai",
        help="兼容内部结构化规格：只转换显式 action steps，不调用模型",
    ),
):
    """
    测试运行入口函数
    """
    # 创建配置对象
    config = Config(
        marker=marker,
        keyword=keyword,
        headed=headed,
        browser=browser,
        env=env,
        project=project,
        base_url=base_url,
        test_file=(
            test_file + ".yaml"
            if test_file and not test_file.endswith((".yaml", ".yml"))
            else test_file
        ),
    )

    # 配置运行环境
    config.configure_environment()
    if ai_mode not in {"strict", "smart"}:
        console.print("[red]--ai-mode 仅支持 strict、smart[/red]")
        sys.exit(1)
    os.environ["UI_AI_MODE"] = ai_mode
    tracker = get_token_usage_tracker()

    if generate_case:
        tracker.start_run(
            run_kind="generate_case",
            metadata={
                "project": config.project.value,
                "env": config.env.value,
                "spec_path": generate_case,
                "dry_run": dry_run,
                "use_ai": not no_ai,
            },
        )
        try:
            result = generate_case_files(
                project=config.project.value,
                env=config.env.value,
                spec_path=generate_case,
                output_name=output_name,
                dry_run=dry_run,
                overwrite=overwrite,
                use_ai=not no_ai,
            )
            display_generation_result(result, dry_run=dry_run)
            display_token_usage_summary(
                tracker.finish_run(
                    status="passed",
                    metadata={"output_name": output_name, "overwrite": overwrite},
                )
            )
            return
        except LLMConfigurationError as e:
            tracker.finish_run(status="failed", metadata={"error": str(e)})
            logger.error(str(e))
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
        except Exception as e:
            tracker.finish_run(status="failed", metadata={"error": str(e)})
            logger.error(f"用例生成失败: {str(e)}")
            console.print(f"[red]用例生成失败: {e}[/red]")
            sys.exit(1)

    # 显示配置信息
    display_run_configuration(config)

    # 记录开始时间
    start_time = time.time()

    try:
        tracker.start_run(
            run_kind="pytest",
            metadata={
                "project": config.project.value,
                "env": config.env.value,
                "test_file": config.test_file,
                "ai_mode": ai_mode,
            },
        )

        pytest_args = build_pytest_args(config)
        logger.info(f"使用参数运行pytest: {' '.join(pytest_args)}")
        exit_code = pytest.main(pytest_args)

        # 显示运行摘要
        show_test_summary(start_time)
        tracker.finish_run(
            status="passed" if exit_code == 0 else "failed",
            metadata={"exit_code": exit_code},
        )

        # 返回退出码
        sys.exit(exit_code)

    except KeyboardInterrupt:
        tracker.finish_run(status="failed", metadata={"error": "KeyboardInterrupt"})
        logger.error("测试被用户中断")
        sys.exit(1)
    except Exception as e:
        tracker.finish_run(status="failed", metadata={"error": str(e)})
        logger.error(f"测试运行出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    typer.run(main)
