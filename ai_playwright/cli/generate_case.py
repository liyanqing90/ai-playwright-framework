from __future__ import annotations

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
    dry_run: bool = typer.Option(False, "--dry-run", help="只预览生成结果，不写文件"),
    no_overwrite: bool = typer.Option(
        False, "--no-overwrite", help="不覆盖已存在的生成文件"
    ),
    no_verify: bool = typer.Option(False, "--no-verify", help="生成后不自动执行验证"),
    context_env: Environment = typer.Option(
        Environment.PROD,
        "--context-env",
        help="生成上下文环境，默认 prod；通常不需要指定",
        hidden=True,
    ),
):
    configure_file_logger()
    tracker = get_token_usage_tracker()
    tracker.start_run(
        run_kind="generate_case",
        metadata={
            "project": project,
            "context_env": context_env.value,
            "spec": spec,
            "dry_run": dry_run,
            "overwrite": not no_overwrite,
            "verify": not no_verify,
        },
    )
    try:
        console.print(
            f"[cyan][gen][/cyan] project={project} spec={spec} "
            f"dry_run={dry_run} overwrite={not no_overwrite}"
        )

        with console.status("[cyan][gen] 准备生成...[/cyan]", spinner="dots") as status:

            def report(message: str) -> None:
                status.update(f"[cyan][gen] {message}[/cyan]")
                console.print(f"[dim][gen] {message}[/dim]")

            result = generate_case_files(
                project=project,
                env=context_env.value,
                spec_path=spec,
                output_name=None,
                dry_run=dry_run,
                overwrite=not no_overwrite,
                use_ai=True,
                verify=not no_verify,
                progress=report,
            )
        display_generation_result(result, dry_run=dry_run)
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


if __name__ == "__main__":
    cli()
