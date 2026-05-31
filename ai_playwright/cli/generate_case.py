from __future__ import annotations

import os

import typer

from ai_playwright.ai_generation import generate_case_files
from ai_playwright.ai_runtime.provider import LLMConfigurationError
from ai_playwright.cli.common import (
    console,
    display_generation_result,
    display_token_usage_summary,
)
from ai_playwright.utils.config import Environment
from ai_playwright.utils.logger import configure_file_logger, logger
from ai_playwright.utils.token_usage import get_token_usage_tracker


app = typer.Typer(
    add_completion=False,
    help="Generate project test cases.",
    pretty_exceptions_show_locals=False,
)


@app.command()
def main(
    spec: str = typer.Argument(..., help="生成规格名，例如 smoke"),
    project: str = typer.Option("demo", "-p", "--project", help="项目"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="已废弃：生成必须执行真实验证，不再支持只预览",
        hidden=True,
    ),
    no_overwrite: bool = typer.Option(
        False, "--no-overwrite", help="不覆盖已存在的生成文件"
    ),
    no_verify: bool = typer.Option(
        False,
        "--no-verify",
        help="已废弃：生成必须执行真实验证",
        hidden=True,
    ),
    context_env: Environment = typer.Option(
        Environment.PROD,
        "--context-env",
        help="生成上下文环境，默认 prod；通常不需要指定",
        hidden=True,
    ),
    headed: bool = typer.Option(
        True,
        "--headed/--headless",
        help="生成验证是否以有头模式运行浏览器",
    ),
    slow_mo: int = typer.Option(
        0,
        "--slow-mo",
        min=0,
        help="生成验证浏览器慢速执行毫秒数",
    ),
):
    configure_file_logger()
    if dry_run:
        console.print("[red]--dry-run 已移除：生成必须执行真实页面验证。[/red]")
        raise typer.Exit(2)
    if no_verify:
        console.print("[red]--no-verify 已移除：生成必须执行真实页面验证。[/red]")
        raise typer.Exit(2)

    tracker = get_token_usage_tracker()
    tracker.start_run(
        run_kind="generate_case",
        metadata={
            "project": project,
            "context_env": context_env.value,
            "spec": spec,
            "overwrite": not no_overwrite,
            "verify": True,
            "headed": headed,
            "slow_mo": slow_mo,
        },
    )
    try:
        console.print(
            f"[cyan][gen][/cyan] project={project} spec={spec} "
            f"overwrite={not no_overwrite} verify=True headed={headed}"
        )

        with console.status("[cyan][gen] 准备生成...[/cyan]", spinner="dots") as status:

            def report(message: str) -> None:
                status.update(f"[cyan][gen] {message}[/cyan]")
                console.print(f"[dim][gen] {message}[/dim]")

            with _temporary_browser_env(headed=headed, slow_mo=slow_mo):
                result = generate_case_files(
                    project=project,
                    env=context_env.value,
                    spec_path=spec,
                    output_name=None,
                    dry_run=False,
                    overwrite=not no_overwrite,
                    use_ai=True,
                    verify=True,
                    progress=report,
                )
        display_generation_result(result)
        for warning in result.warnings:
            logger.warning(f"用例生成警告: {warning}")
        display_token_usage_summary(
            tracker.finish_run(
                status="passed",
                metadata={
                    "case_file": str(result.case_file),
                    "data_file": str(result.data_file),
                },
            )
        )
    except LLMConfigurationError as exc:
        tracker.finish_run(status="failed", metadata={"error": str(exc)})
        logger.error(str(exc))
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        tracker.finish_run(status="failed", metadata={"error": str(exc)})
        logger.error(f"用例生成失败: {exc}")
        console.print(f"[red]用例生成失败: {exc}[/red]")
        raise typer.Exit(1) from exc


def cli() -> None:
    app()


class _temporary_browser_env:
    def __init__(self, *, headed: bool, slow_mo: int) -> None:
        self.headed = headed
        self.slow_mo = slow_mo
        self.previous: dict[str, str | None] = {}

    def __enter__(self):
        self.previous = {
            "PWHEADED": os.environ.get("PWHEADED"),
            "PWSLOWMO": os.environ.get("PWSLOWMO"),
        }
        os.environ["PWHEADED"] = "1" if self.headed else "0"
        os.environ["PWSLOWMO"] = str(self.slow_mo)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for name, value in self.previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    cli()
