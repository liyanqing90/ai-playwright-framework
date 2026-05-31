from pathlib import Path

import pytest

from ai_playwright.step_actions.utils import _resolve_allowed_script_path
from ai_playwright.yaml_schema import (
    SchemaIssue,
    ValidationContext,
    YamlSchemaValidationError,
    validate_case_file,
    validate_pytest_targets,
    validate_project,
)
from ai_playwright.utils.yaml_handler import YamlHandler


def test_yaml_schema_rejects_missing_selector(tmp_path: Path):
    project = _write_schema_project(
        tmp_path,
        case_steps=[{"action": "click"}],
        elements={"submit_button": "#submit"},
    )

    with pytest.raises(YamlSchemaValidationError, match="需要 selector"):
        validate_project(project)


def test_yaml_schema_rejects_unknown_selector_key(tmp_path: Path):
    project = _write_schema_project(
        tmp_path,
        case_steps=[{"action": "click", "selector": "missing_button"}],
        elements={"submit_button": "#submit"},
    )

    with pytest.raises(YamlSchemaValidationError, match="selector 未在 elements"):
        validate_project(project)


def test_pytest_target_schema_validation_rejects_missing_yaml_target(tmp_path: Path):
    missing_target = tmp_path / "test_data" / "demo" / "cases" / "missing.yaml"

    with pytest.raises(YamlSchemaValidationError, match="YAML 目标不存在"):
        validate_pytest_targets([missing_target])


def test_pytest_plugin_logs_schema_error_before_exit(monkeypatch):
    from ai_playwright import pytest_plugin

    messages: list[str] = []

    class FakeLogger:
        def error(self, message: str) -> None:
            messages.append(message)

    class FakeTracker:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def finish_run(self, **kwargs):
            self.calls.append(kwargs)
            return None

    fake_tracker = FakeTracker()

    def fake_exit(message: str, returncode: int | None = None) -> None:
        raise RuntimeError(f"{returncode}:{message}")

    monkeypatch.setattr(pytest_plugin, "logger", FakeLogger())
    monkeypatch.setattr(pytest_plugin, "get_token_usage_tracker", lambda: fake_tracker)
    monkeypatch.setattr(pytest_plugin.pytest, "exit", fake_exit)

    exc = YamlSchemaValidationError(
        [SchemaIssue(path="test_data/demo/cases/missing.yaml", message="bad schema")]
    )

    with pytest.raises(RuntimeError, match="2:YAML schema"):
        pytest_plugin._exit_for_yaml_schema_error(exc)

    assert messages and "bad schema" in messages[0]
    assert fake_tracker.calls[0]["status"] == "failed"
    assert fake_tracker.calls[0]["metadata"]["phase"] == "yaml_schema"


def test_yaml_schema_rejects_unknown_step_field(tmp_path: Path):
    project = _write_schema_project(
        tmp_path,
        case_steps=[
            {"action": "click", "selector": "submit_button", "variable_name": "noop"}
        ],
    )

    with pytest.raises(YamlSchemaValidationError):
        validate_project(project)


def test_yaml_schema_allows_smart_target_without_selector(tmp_path: Path):
    project = _write_schema_project(
        tmp_path,
        case_data={
            "description": "smart fallback",
            "mode": "smart",
            "steps": [{"action": "click", "target": "提交按钮"}],
        },
    )

    validate_project(project)


def test_yaml_schema_allows_native_ai_step_instruction(tmp_path: Path):
    project = _write_schema_project(
        tmp_path,
        case_data={
            "description": "native ai step instruction",
            "mode": "strict",
            "steps": [{"action": "ai_step", "instruction": "open cart"}],
        },
    )

    validate_project(project)


def test_yaml_schema_rejects_unknown_action(tmp_path: Path):
    project = _write_schema_project(
        tmp_path,
        case_data={
            "description": "unknown action is rejected",
            "mode": "smart",
            "steps": [
                {
                    "action": "unknown_click",
                    "target": "login button",
                    "mode": "smart",
                }
            ],
        },
    )

    with pytest.raises(YamlSchemaValidationError, match="不支持的 action"):
        validate_project(project)


def test_yaml_schema_rejects_runtime_ai_case(tmp_path: Path):
    project = _write_schema_project(
        tmp_path,
        case_data={
            "description": "runtime ai case",
            "type": "ai_case",
            "mode": "smart",
            "intent": "finish the cart checkout flow",
            "checkpoints": ["cart badge is 1"],
            "final": ["checkout complete page is visible"],
        },
    )

    with pytest.raises(YamlSchemaValidationError, match="standard/agent_case"):
        validate_project(project)


def test_yaml_schema_allows_agent_case_intent_without_mode(tmp_path: Path):
    project = _write_schema_project(
        tmp_path,
        case_data={
            "description": "runtime agent case",
            "type": "agent_case",
            "intent": "open Saucedemo and complete checkout flow",
            "checkpoints": ["cart contains target product"],
            "final": ["page shows Thank you for your order"],
        },
    )

    validate_project(project)


def test_yaml_schema_allows_agent_case_steps_without_intent(tmp_path: Path):
    project = _write_schema_project(
        tmp_path,
        case_data={
            "description": "runtime agent case steps",
            "type": "agent_case",
            "steps": [
                "open Saucedemo login page",
                "login with standard user",
                "open cart page",
            ],
            "checkpoints": ["product list page opens after login"],
            "final": ["cart page visible"],
        },
    )

    validate_project(project)


def test_yaml_schema_rejects_agent_case_mode_field(tmp_path: Path):
    project = _write_schema_project(
        tmp_path,
        case_data={
            "description": "runtime agent case",
            "type": "agent_case",
            "mode": "agent",
            "intent": "open Saucedemo and complete checkout flow",
            "final": ["order complete"],
        },
    )

    with pytest.raises(YamlSchemaValidationError, match="mode"):
        validate_project(project)


def test_yaml_schema_rejects_agent_case_without_checkpoints_or_final(tmp_path: Path):
    project = _write_schema_project(
        tmp_path,
        case_data={
            "description": "runtime agent case",
            "type": "agent_case",
            "intent": "open Saucedemo and complete checkout flow",
        },
    )

    with pytest.raises(YamlSchemaValidationError, match="checkpoints"):
        validate_project(project)


def test_yaml_schema_rejects_nested_agent_case_contract(tmp_path: Path):
    project = _write_schema_project(
        tmp_path,
        case_data={
            "description": "runtime agent case",
            "type": "agent_case",
            "agent_case": {
                "intent": "legacy nested format",
                "final": ["order complete"],
            },
        },
    )

    with pytest.raises(YamlSchemaValidationError, match="agent_case"):
        validate_project(project)


def test_yaml_schema_rejects_agent_mode_on_standard_case(tmp_path: Path):
    project = _write_schema_project(
        tmp_path,
        case_data={
            "description": "invalid agent mode",
            "mode": "agent",
            "steps": [{"action": "click", "selector": "submit_button"}],
        },
    )

    with pytest.raises(YamlSchemaValidationError, match="strict/smart"):
        validate_project(project)


def test_yaml_schema_rejects_missing_module_reference(tmp_path: Path):
    project = _write_schema_project(
        tmp_path,
        case_steps=[{"use_module": "missing_login"}],
    )

    with pytest.raises(YamlSchemaValidationError, match="找不到 module"):
        validate_project(project)


def test_pytest_target_schema_validation_does_not_read_unselected_generated_assets(
    tmp_path: Path,
):
    project = tmp_path / "demo"
    for directory in ("cases", "data", "elements", "modules"):
        (project / directory).mkdir(parents=True, exist_ok=True)

    (project / "cases" / "agent.yaml").write_text(
        "test_cases:\n  - name: test_agent\n",
        encoding="utf-8",
    )
    (project / "data" / "agent.yaml").write_text(
        "\n".join(
            [
                "test_data:",
                "  test_agent:",
                "    type: agent_case",
                "    intent: fill demo sku and search",
                "    checkpoints:",
                "      - search button executed",
            ]
        ),
        encoding="utf-8",
    )
    (project / "cases" / "generated.yaml").write_text(
        "test_cases:\n  - name: test_generated\n",
        encoding="utf-8",
    )
    (project / "data" / "generated.yaml").write_text(
        "\n".join(
            [
                "test_data:",
                "  test_generated:",
                "    mode: smart",
                "    steps:",
                "      - action: click",
                "        selector: generated_later_button",
            ]
        ),
        encoding="utf-8",
    )

    validate_pytest_targets([project / "cases" / "agent.yaml", "-v"])
    with pytest.raises(YamlSchemaValidationError, match="generated_later_button"):
        validate_project(project)


def test_validate_case_file_reports_case_context(tmp_path: Path):
    context = ValidationContext(
        test_dir=tmp_path,
        project="demo",
        elements={},
        test_datas={},
        modules={},
    )
    case_file = tmp_path / "cases.yaml"
    case_file.write_text(
        "test_cases:\n  - name: test_missing_data\n",
        encoding="utf-8",
    )

    validate_case_file(case_file, context)

    assert context.issues
    assert "test_missing_data" in context.issues[0].path


def test_yaml_schema_allows_extended_playwright_actions(tmp_path: Path):
    project = _write_schema_project(
        tmp_path,
        elements={
            "source": "#source",
            "target": "#target",
            "checkbox": "#checkbox",
            "button": "#button",
            "input": "#input",
            "items": ".item",
        },
        case_steps=[
            {"action": "check", "selector": "checkbox"},
            {
                "action": "click",
                "selector": {"role": "button", "name": "Submit", "exact": True},
            },
            {"action": "set_checked", "selector": "checkbox", "checked": False},
            {
                "action": "drag_and_drop",
                "selector": "source",
                "target_selector": "#target",
            },
            {"action": "tap", "selector": "button"},
            {"action": "select_text", "selector": "input"},
            {"action": "press_sequentially", "selector": "input", "value": "abc"},
            {"action": "dispatch_event", "selector": "button", "event_type": "click"},
            {"action": "assert_checked", "selector": "checkbox"},
            {"action": "go_back"},
            {"action": "go_forward"},
            {"action": "wait_for_url", "url": "**/done"},
            {"action": "execute_js", "script": "() => window.__demo = true"},
            {"action": "wait_for_function", "expression": "() => true"},
            {"action": "add_init_script", "script": "window.__x = 1"},
            {"action": "set_viewport_size", "width": 1280, "height": 720},
            {"action": "emulate_media", "color_scheme": "dark"},
            {"action": "wait_for_request", "url_pattern": "**/api/items"},
            {"action": "wait_for_response", "url_pattern": "**/api/items"},
            {
                "action": "mock_route",
                "url_pattern": "**/api/items",
                "json": {"items": []},
            },
            {
                "action": "abort_route",
                "url_pattern": "**/tracking",
                "error_code": "blockedbyclient",
            },
            {"action": "unroute", "url_pattern": "**/api/items"},
            {"action": "set_offline", "offline": False},
            {"action": "set_extra_http_headers", "headers": {"x-test": "1"}},
            {"action": "grant_permissions", "permissions": ["geolocation"]},
            {"action": "clear_permissions"},
            {
                "action": "save_storage_state",
                "path": "evidence/storage/state.json",
                "variable_name": "storage_state",
            },
            {
                "action": "get_value",
                "selector": "input",
                "variable_name": "input_value",
            },
            {
                "action": "get_text",
                "selector": "button",
                "variable_name": "button_text",
            },
            {
                "action": "get_attribute",
                "selector": "button",
                "attribute": "data-id",
                "variable_name": "button_attr",
            },
            {
                "action": "get_bounding_box",
                "selector": "button",
                "variable_name": "button_box",
            },
            {
                "action": "get_all_elements",
                "selector": "items",
                "variable_name": "items",
            },
            {
                "action": "capture",
                "selector": "button",
                "path": "evidence/screenshots/demo.png",
                "full_page": False,
            },
            {
                "action": "cookies",
                "cookie_action": "get",
                "variable_name": "cookies",
            },
        ],
    )

    validate_project(project)


def _write_schema_project(
    tmp_path: Path,
    *,
    case_steps: list[dict] | None = None,
    case_data: dict | None = None,
    elements: dict | None = None,
    modules: dict | None = None,
) -> Path:
    project = tmp_path / "demo"
    for directory in ("cases", "data", "elements", "modules"):
        (project / directory).mkdir(parents=True, exist_ok=True)

    (project / "cases" / "case.yaml").write_text(
        "test_cases:\n  - name: test_demo\n",
        encoding="utf-8",
    )
    data_payload = case_data or {
        "description": "demo",
        "steps": case_steps or [{"action": "click", "selector": "submit_button"}],
    }
    YamlHandler().save_to_yaml(
        {"test_data": {"test_demo": data_payload}},
        project / "data",
        "data",
    )
    YamlHandler().save_to_yaml(
        {"elements": elements or {"submit_button": "#submit"}},
        project / "elements",
        "elements",
    )
    if modules:
        YamlHandler().save_to_yaml(modules, project / "modules", "modules")
    return project


def test_dynamic_script_path_is_limited_to_files_dir(tmp_path: Path):
    outside_script = tmp_path / "script.py"
    outside_script.write_text("def run(): pass\n", encoding="utf-8")

    with pytest.raises(ValueError, match="files"):
        _resolve_allowed_script_path(outside_script)
