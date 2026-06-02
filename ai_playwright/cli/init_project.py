from __future__ import annotations

import shutil
from pathlib import Path

import typer

from ai_playwright.cli.common import console
from ai_playwright.project_paths import TEMPLATE_ROOT, demo_template_root


app = typer.Typer(
    add_completion=False,
    help="Initialize a local AI Playwright workspace.",
    pretty_exceptions_show_locals=False,
)


@app.command()
def main(
    target_dir: Path = typer.Argument(
        Path("."),
        help="目标目录，默认当前目录",
    ),
    force: bool = typer.Option(False, "--force", help="覆盖已存在的模板文件"),
) -> None:
    target = target_dir.expanduser().resolve()
    _copy_template_dir(TEMPLATE_ROOT / "config", target / "config", force=force)
    _copy_template_dir(
        demo_template_root(),
        target / "test_data" / "demo",
        force=force,
    )
    console.print(f"[green]Initialized AI Playwright workspace:[/green] {target}")
    console.print("Next: edit `.env`, then run `run_case -p demo --headless`.")


def _copy_template_dir(source: Path, target: Path, *, force: bool) -> None:
    if not source.exists():
        raise typer.BadParameter(f"模板目录不存在: {source}")
    if source.resolve() == target.resolve():
        return
    for source_path in source.rglob("*"):
        if source_path.is_dir():
            continue
        relative = source_path.relative_to(source)
        target_path = target / relative
        if target_path.exists() and not force:
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def cli() -> None:
    app()


if __name__ == "__main__":
    cli()
