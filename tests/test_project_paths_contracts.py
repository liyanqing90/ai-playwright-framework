from pathlib import Path

from typer.testing import CliRunner

from ai_playwright import project_paths
from ai_playwright.cli.init_project import app


def test_init_project_copies_demo_from_single_canonical_source(tmp_path: Path):
    result = CliRunner().invoke(app, [str(tmp_path)])

    assert result.exit_code == 0
    source = project_paths.demo_template_root()
    copied_case = tmp_path / "test_data" / "demo" / "cases" / "saucedemo_ai.yaml"
    assert copied_case.read_text(encoding="utf-8") == (
        source / "cases" / "saucedemo_ai.yaml"
    ).read_text(encoding="utf-8")
    assert (tmp_path / "test_data" / "demo" / "generation").is_dir()


def test_resolve_project_test_dir_prefers_workspace_demo(
    monkeypatch,
    tmp_path: Path,
):
    workspace_demo = tmp_path / "test_data" / "demo"
    workspace_demo.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    resolved = project_paths.resolve_project_test_dir(
        {"test_dir": "test_data/demo"},
        "demo",
    )

    assert resolved == workspace_demo
    assert not project_paths.is_packaged_template_path(resolved)


def test_source_demo_fallback_is_read_only_outside_workspace(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.chdir(tmp_path)

    resolved = project_paths.resolve_project_test_dir(
        {"test_dir": "test_data/demo"},
        "demo",
    )

    assert resolved == project_paths.demo_template_root()
    assert project_paths.is_packaged_template_path(resolved)
