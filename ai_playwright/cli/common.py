from __future__ import annotations

from rich.console import Console
from rich.table import Table


console = Console()


def normalize_yaml_file_name(value: str | None) -> str:
    if not value:
        return ""
    return value if value.endswith((".yaml", ".yml")) else f"{value}.yaml"


def display_generation_result(result, dry_run: bool) -> None:
    table = Table(title="AI用例生成预览" if dry_run else "AI用例生成结果")
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
