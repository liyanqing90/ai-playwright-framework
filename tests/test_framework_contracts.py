import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from src.ai_generation.case_generator import (
    _GenerationArtifacts,
    _build_payload,
    _has_explicit_steps,
    _resolve_navigation_context,
    _save_generation_cache,
    _validate_spec_project_scope,
    _write_payload,
    resolve_generation_spec_path,
)
from src.ai_generation.harness import GenerationHarness
from src.ai_generation.harness import _safe_case_name
from src.ai_generation.project_context import ProjectContext
from src.ai_runtime.cache_scope import normalize_entry_url
from src.ai_runtime.agent_case_executor import (
    AgentCaseAdvisoryCache,
    AgentCaseExecutor,
    _remaining_step_hints,
    _unmet_final_criteria,
)
from src.ai_runtime.contracts import (
    AgentCaseDecision,
    AiStepDecision,
    GeneratedCasePayload,
    SelectorDecision,
    VisionFindResult,
)
from src.ai_runtime.element_store import ElementDefinitionStore
from src.ai_runtime.payload_compactor import (
    build_dom_context,
    compact_dom_candidates,
    selector_for_element_id,
)
from src.ai_runtime.playwright_selectors import heuristic_selectors, redact_value
from src.ai_runtime.provider import (
    ChatCompletionProvider,
    LLMSettings,
    build_response_format,
    load_llm_settings,
    openai_strict_schema,
    parse_json_object,
    parse_model_response,
)
from src.ai_runtime.selector_registry import SelectorRegistry
from src.ai_runtime.smart_resolver import (
    AiStepOperation,
    ResolvedSelector,
    SmartResolver,
    _parse_fill_instruction,
)
from src.ai_runtime.vision_client import (
    VisionClient,
    VisionServiceUnavailable,
    VisionSettings,
)
from src.ai_runtime.vision_resolver import VisionResolution, VisionResolver
from src.runner import build_test_signature
from src.test_case_executor import CaseExecutor
from src.step_actions.commands import assertion_commands as assertion_commands_module
from src.step_actions.safe_expression import SafeExpressionError, safe_eval_expression
from src.step_actions.step_executor import StepExecutor
from src.step_actions.utils import _resolve_allowed_script_path
from src.yaml_schema import (
    ValidationContext,
    YamlSchemaValidationError,
    validate_case_file,
    validate_project,
)
from utils.variable_manager import VariableManager
from utils.yaml_handler import YamlHandler
from utils.token_usage import TokenUsageTracker, normalize_token_usage


def test_assertion_success_log_contains_expected_and_actual(monkeypatch):
    messages: list[str] = []

    class FakeLogger:
        def info(self, message: str):
            messages.append(message)

    monkeypatch.setattr(assertion_commands_module, "logger", FakeLogger())

    assertion_commands_module._log_assertion_success(
        "assert_text",
        selector="#title",
        expected="Products",
        actual="Products",
    )

    assert messages == [
        "断言通过: action=assert_text | selector=#title | 预期结果=Products | 实际结果=Products"
    ]


def test_assert_text_command_logs_resolved_expected_and_dom_actual(monkeypatch):
    messages: list[str] = []

    class FakeLogger:
        def info(self, message: str):
            messages.append(message)

    class FakeVariableManager:
        def replace_variables_refactored(self, value):
            return "Products" if value == "${expected_title}" else value

    class FakeFirstLocator:
        def inner_text(self):
            return "Products"

    class FakeLocator:
        first = FakeFirstLocator()

    class FakePage:
        def locator(self, selector: str):
            assert selector == "#title"
            return FakeLocator()

    class FakeUiHelper:
        page = FakePage()
        variable_manager = FakeVariableManager()

        def assert_text(self, selector: str, expected: str):
            assert selector == "#title"
            assert expected == "${expected_title}"

    monkeypatch.setattr(assertion_commands_module, "logger", FakeLogger())

    assertion_commands_module.AssertTextCommand().execute(
        FakeUiHelper(),
        "#title",
        None,
        {"expected": "${expected_title}"},
    )

    assert "预期结果=Products" in messages[0]
    assert "实际结果=Products" in messages[0]


def test_native_ai_step_executes_through_command_pipeline(monkeypatch):
    calls: list[tuple[str, str, str | None]] = []

    class FakeResolver:
        def resolve_ai_step(self, *, instruction: str, timeout: int):
            assert instruction == "打开购物车"
            return AiStepOperation(
                action="click",
                selector="#cart",
                prompt_version="ai-step-v1",
                schema_version="schema-v1",
                model="test-model",
                candidate_count=3,
                candidate_hash="abc123",
            )

    class FakePage:
        url = "https://example.test"

    class FakeUiHelper:
        pass

    def fake_execute_action_with_command(ui_helper, action, selector, value, step):
        calls.append((action, selector, value))

    monkeypatch.setattr(
        "src.step_actions.step_executor.execute_action_with_command",
        fake_execute_action_with_command,
    )

    executor = StepExecutor(FakePage(), FakeUiHelper(), elements={})
    executor.smart_resolver = FakeResolver()
    executor.execute_step({"action": "ai_step", "instruction": "打开购物车"})

    assert calls == [("click", "#cart", None)]


def test_native_ai_step_rejects_multi_action_instruction():
    class FakeResolver:
        def resolve_ai_step(self, *, instruction: str, timeout: int):
            assert instruction == "login and open cart"
            return AiStepOperation(
                action="reject",
                reason="instruction contains multiple UI actions",
                prompt_version="ai-step-v1",
                schema_version="schema-v1",
                model="test-model",
            )

    class FakePage:
        url = "https://example.test"

    class FakeUiHelper:
        pass

    executor = StepExecutor(FakePage(), FakeUiHelper(), elements={})
    executor.smart_resolver = FakeResolver()

    with pytest.raises(ValueError, match="ai_step"):
        executor.execute_step(
            {"action": "ai_step", "instruction": "login and open cart"}
        )


def test_case_and_step_ai_modes_are_native_executor_defaults():
    executor = StepExecutor(page=None, ui_helper=None, elements={}, default_mode="smart")

    assert executor._resolve_mode({}) == "smart"
    assert executor._resolve_mode({"mode": "smart"}) == "smart"


def test_case_executor_rejects_runtime_ai_case():
    executor = CaseExecutor(
        {
            "description": "runtime ai case",
            "type": "ai_case",
            "mode": "smart",
            "intent": "finish cart flow",
            "final": ["cart flow done"],
        },
        elements={"title": ".title"},
        case_metadata={"name": "test_ai_case"},
    )

    with pytest.raises(ValueError, match="run_case"):
        executor.execute_test_case(page=object(), ui_helper=object())


def test_case_executor_routes_agent_case_to_agent_runner(monkeypatch):
    calls: list[dict] = []

    class FakeAgentRunner:
        def __init__(self, *, page, ui_helper, elements):
            calls.append({"page": page, "ui_helper": ui_helper, "elements": elements})

        def run(self, *, case_name, case_data):
            calls.append({"case_name": case_name, "case_data": case_data})
            return type(
                "Result",
                (),
                {
                    "steps_executed": 2,
                    "model_calls": 1,
                    "cache_replayed_steps": 0,
                    "final_reason": "done",
                },
            )()

    monkeypatch.setattr("src.test_case_executor.AgentCaseExecutor", FakeAgentRunner)

    CaseExecutor(
        {
            "description": "agent case",
            "type": "agent_case",
            "intent": "完成购物车流程",
            "final": ["订单完成"],
        },
        elements={"title": ".title"},
        case_metadata={"name": "test_agent_case"},
    ).execute_test_case(page="page", ui_helper="ui")

    assert calls[0]["elements"] == {"title": ".title"}
    assert calls[1]["case_name"] == "test_agent_case"
    assert calls[1]["case_data"]["intent"] == "完成购物车流程"


def test_agent_case_spec_accepts_natural_language_steps(tmp_path: Path):
    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://www.saucedemo.com/",
        elements={},
        modules={},
        variables={},
        test_cases=[],
        test_data={},
    )

    executor = AgentCaseExecutor(
        page=object(),
        ui_helper=object(),
        elements={},
        context=context,
    )

    spec = executor._agent_spec(
        case_name="test_agent_steps",
        case_data={
            "type": "agent_case",
            "steps": ["打开登录页", "登录标准用户", "打开购物车"],
            "final": ["购物车页面可见"],
        },
    )

    assert spec["input_type"] == "steps"
    assert spec["steps"] == ["打开登录页", "登录标准用户", "打开购物车"]
    assert spec["intent"] == "按顺序完成自然语言步骤：打开登录页；登录标准用户；打开购物车"


def test_compact_dom_candidates_keeps_semantics_and_drops_geometry():
    candidates = [
        {
            "index": 0,
            "tag": "div",
            "selector": ".layout-wrapper",
            "class_name": "layout-wrapper useless deeply nested classes",
            "text": "Decorative container",
            "bbox": [0, 0, 100, 100],
            "center": [50, 50],
        },
        {
            "index": 1,
            "tag": "a",
            "selector": "a[data-test=\"shopping-cart-link\"]",
            "data_test": "shopping-cart-link",
            "text": "1",
            "ancestor_text": "Shopping cart header with a very long text " * 10,
            "bbox": [1, 2, 3, 4],
            "center_norm": [0.1, 0.2],
            "visible": True,
            "enabled": True,
        },
        {
            "index": 2,
            "tag": "button",
            "selector": "#checkout",
            "text": "Checkout",
            "class_name": "btn btn-primary btn-large",
            "bbox_norm": [0.1, 0.1, 0.3, 0.2],
            "visible": True,
            "enabled": True,
        },
    ]

    compacted = compact_dom_candidates(
        candidates,
        limit=2,
        hints=["open shopping cart"],
    )

    assert len(compacted) == 2
    assert any(item["selector"] == 'a[data-test="shopping-cart-link"]' for item in compacted)
    for item in compacted:
        assert "bbox" not in item
        assert "bbox_norm" not in item
        assert "center" not in item
        assert "center_norm" not in item
        assert "class_name" not in item
    cart = next(item for item in compacted if item["selector"] == 'a[data-test="shopping-cart-link"]')
    assert len(cart["ancestor_text"]) <= 163


def test_dom_context_uses_element_ids_and_selector_candidates():
    context = build_dom_context(
        [
            {
                "index": 12,
                "tag": "button",
                "selector": "#add-to-cart-sauce-labs-backpack",
                "data_test": "add-to-cart-sauce-labs-backpack",
                "text": "Add to cart",
                "ancestor_text": "Sauce Labs Backpack $29.99 Add to cart",
                "visible": True,
                "enabled": True,
            },
            {
                "index": 18,
                "tag": "span",
                "selector": ".shopping_cart_badge",
                "text": "1",
                "ancestor_text": "Shopping cart badge 1",
                "visible": True,
                "enabled": True,
            },
        ],
        url="https://www.saucedemo.com/inventory.html",
        title="Swag Labs",
        hints=["Sauce Labs Backpack Add to cart"],
    )

    button = context["interactive_elements"][0]
    assert button["id"] == "e12"
    assert button["selector_candidates"][0] == 'button[data-test="add-to-cart-sauce-labs-backpack"]'
    assert selector_for_element_id(context, "e12") == 'button[data-test="add-to-cart-sauce-labs-backpack"]'
    assert context["business_objects"]["cards"][0]["name"].startswith("Sauce Labs Backpack")
    assert context["compression"]["kept_element_count"] == 2


def test_agent_case_model_payload_uses_compact_dom_context(monkeypatch, tmp_path: Path):
    captured: dict[str, Any] = {}

    class FakeStepExecutor:
        def __init__(self, page, ui_helper, elements, default_mode=None):
            pass

        def execute_step(self, step):
            raise AssertionError("done decision should not execute a step")

    class FakeProvider:
        settings = type("Settings", (), {"model": "test-model"})()

        def complete_model(self, messages, response_model, **kwargs):
            captured["payload"] = json.loads(messages[1]["content"])
            return response_model.model_validate({"action": "done", "reason": "done"})

    class FakePage:
        url = "https://example.test/inventory"

        def title(self):
            return "Swag Labs"

    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.StepExecutor",
        FakeStepExecutor,
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.ChatCompletionProvider",
        FakeProvider,
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.collect_candidates",
        lambda page, limit: [
            {
                "index": 0,
                "tag": "button",
                "selector": "#react-burger-menu-btn",
                "text": "Open Menu",
                "class_name": "bm-burger-button",
                "bbox": [1, 2, 3, 4],
                "visible": True,
                "enabled": True,
            },
            {
                "index": 1,
                "tag": "a",
                "selector": "a[data-test=\"shopping-cart-link\"]",
                "data_test": "shopping-cart-link",
                "text": "1",
                "center": [10, 10],
                "visible": True,
                "enabled": True,
            },
            {
                "index": 2,
                "tag": "div",
                "selector": ".inventory_list",
                "class_name": "inventory_list",
                "ancestor_text": "large inventory text " * 20,
                "bbox_norm": [0, 0, 1, 1],
                "visible": True,
                "enabled": True,
            },
        ],
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.load_ai_config",
        lambda: {
            "runtime": {
                "agent_case_cache_enabled": False,
                "agent_candidate_scan_limit": 20,
                "agent_candidate_limit": 2,
                "agent_context_items": 2,
                "agent_history_limit": 1,
            },
            "agent_policy": {
                "limits": {
                    "max_steps": 2,
                    "max_model_calls": 1,
                    "max_duration_seconds": 30,
                },
                "guardrails": {"require_checkpoints_or_final": True},
            },
            "prompts": {"agent_case_version": "agent-case-test"},
            "llm": {"schema_version": "schema-test"},
        },
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor._unmet_final_criteria",
        lambda **kwargs: [],
    )
    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://example.test/",
        elements={"login_button": "#login", "cart_link": ".cart", "unused": ".x"},
        modules={
            "login": [
                {"action": "goto", "value": "https://www.saucedemo.com/"},
                {"action": "fill", "selector": "username", "value": "${user}"},
            ]
        },
        variables={"standard_username": "standard_user", "common_password": "secret"},
        test_cases=[],
        test_data={},
    )

    AgentCaseExecutor(
        page=FakePage(),
        ui_helper=object(),
        elements={},
        context=context,
    ).run(
        case_name="test_agent_compact",
        case_data={
            "type": "agent_case",
            "intent": "打开购物车并完成验证",
            "final": ["购物车页面可见"],
        },
    )

    payload = captured["payload"]
    assert len(payload["dom_candidates"]) == 2
    assert payload["dom_context"]["interactive_elements"][0]["id"] == "e0"
    assert payload["dom_context"]["interactive_elements"][0]["selector_candidates"]
    assert len(payload["project_context"]["element_keys"]) == 2
    assert payload["project_context"]["modules"]["login"][0]["action"] == "goto"
    for item in payload["dom_context"]["interactive_elements"]:
        assert "class_name" not in item
        assert "bbox" not in item
        assert "bbox_norm" not in item
        assert "center" not in item


def test_agent_case_does_not_expose_business_fast_path():
    assert not hasattr(AgentCaseExecutor, "_fast_path_decision")


def test_agent_case_step_hints_keep_full_short_plan():
    steps = [f"step {index}" for index in range(1, 12)]

    assert _remaining_step_hints(spec={"steps": steps}, history=[]) == steps


def test_agent_case_finish_detects_unmet_logout_final_criteria():
    unmet = _unmet_final_criteria(
        criteria={"final": ["退出登录后回到登录页，登录按钮可见"]},
        history=[
            {"step": {"action": "click", "selector": 'button[data-test="finish"]'}},
        ],
        current_url="https://www.saucedemo.com/checkout-complete.html",
        dom_context={
            "interactive_elements": [
                {"selector": 'button[data-test="finish"]', "text": "Finish"}
            ],
            "assertion_candidates": [],
        },
    )

    assert unmet == ["退出登录后回到登录页，登录按钮可见"]


def test_agent_case_decision_accepts_standard_step_mode():
    decision = AgentCaseDecision.model_validate(
        {"action": "click", "mode": "smart", "selector": "#continue"}
    )

    assert decision.mode == "smart"
    assert decision.selector == "#continue"


def test_agent_case_maps_selector_field_element_id_to_real_selector(tmp_path: Path):
    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://www.saucedemo.com/",
        elements={},
        modules={},
        variables={},
        test_cases=[],
        test_data={},
    )
    executor = AgentCaseExecutor(
        page=object(),
        ui_helper=object(),
        elements={},
        context=context,
    )
    executor.current_dom_context = build_dom_context(
        [
            {
                "index": 1,
                "tag": "input",
                "selector": "#user-name",
                "id": "user-name",
                "placeholder": "Username",
                "visible": True,
                "enabled": True,
            }
        ],
        url="https://www.saucedemo.com/",
        title="Swag Labs",
    )

    step = executor._decision_to_step(
        AgentCaseDecision.model_validate(
            {
                "action": "fill",
                "selector": "e1",
                "value": "${standard_username}",
                "reason": "填写用户名",
            }
        )
    )

    assert step["selector"] == "#user-name"


def test_agent_case_maps_target_field_element_id_to_real_selector(tmp_path: Path):
    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://example.test/",
        elements={},
        modules={},
        variables={},
        test_cases=[],
        test_data={},
    )
    executor = AgentCaseExecutor(
        page=object(),
        ui_helper=object(),
        elements={},
        context=context,
    )
    executor.current_dom_context = build_dom_context(
        [
            {
                "index": 2,
                "tag": "input",
                "selector": "#password",
                "id": "password",
                "placeholder": "Password",
                "visible": True,
                "enabled": True,
            }
        ],
        url="https://example.test/",
        title="Example",
    )

    step = executor._decision_to_step(
        AgentCaseDecision.model_validate(
            {
                "action": "fill",
                "target": "e2",
                "value": "${common_password}",
                "reason": "填写密码",
            }
        )
    )

    assert step["selector"] == "#password"
    assert "target" not in step


def test_agent_case_assert_text_uses_dom_text_selector_without_business_rule(tmp_path: Path):
    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://example.test/",
        elements={},
        modules={},
        variables={},
        test_cases=[],
        test_data={},
    )
    executor = AgentCaseExecutor(
        page=object(),
        ui_helper=object(),
        elements={},
        context=context,
    )
    executor.current_dom_context = build_dom_context(
        [
            {
                "index": 5,
                "tag": "h1",
                "selector": "[data-test='page-title']",
                "data_test": "page-title",
                "text": "Overview",
                "visible": True,
                "enabled": True,
            }
        ],
        url="https://example.test/dashboard",
        title="Example",
    )

    step = executor._decision_to_step(
        AgentCaseDecision.model_validate(
            {
                "action": "assert_text",
                "target": "page heading",
                "value": "Overview",
                "reason": "assert heading",
            }
        )
    )

    assert step["selector"] == 'h1[data-test="page-title"]'
    assert step["value"] == "Overview"
    assert "target" not in step


def test_agent_case_rejects_unresolved_internal_target_reference(tmp_path: Path):
    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://example.test/",
        elements={},
        modules={},
        variables={},
        test_cases=[],
        test_data={},
    )
    executor = AgentCaseExecutor(
        page=object(),
        ui_helper=object(),
        elements={},
        context=context,
    )
    executor.current_dom_context = build_dom_context(
        [
            {
                "index": 1,
                "tag": "input",
                "selector": "#user-name",
                "id": "user-name",
                "placeholder": "Username",
                "visible": True,
                "enabled": True,
            }
        ],
        url="https://example.test/",
        title="Example",
    )

    with pytest.raises(ValueError, match="未解析的内部element_id target: e2"):
        executor._decision_to_step(
            AgentCaseDecision.model_validate(
                {
                    "action": "fill",
                    "target": "e2",
                    "value": "${common_password}",
                    "reason": "填写密码",
                }
            )
        )


def test_parse_fill_instruction_supports_chinese_input_box_phrasing():
    assert _parse_fill_instruction("在密码输入框中输入secret_sauce") == (
        "密码输入框",
        "secret_sauce",
    )
    assert _parse_fill_instruction('在用户名输入框输入"standard_user"') == (
        "用户名输入框",
        "standard_user",
    )


def test_parse_fill_instruction_preserves_english_field_spacing():
    assert _parse_fill_instruction("fill password field with secret_sauce") == (
        "password field",
        "secret_sauce",
    )


def test_llm_response_format_auto_uses_text_for_gguf(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:4000/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "Qwen_Qwen3.5-9B-Q6_K.gguf")
    monkeypatch.delenv("LLM_RESPONSE_FORMAT", raising=False)
    monkeypatch.setattr(
        "src.ai_runtime.provider.load_ai_config",
        lambda: {"llm": {"response_format": "auto"}},
    )

    settings = load_llm_settings()

    assert settings.response_format == "text"
    assert settings.timeout_seconds == 180
    assert settings.reasoning_effort is None
    assert build_response_format(
        settings=settings,
        response_json=True,
        response_model=AgentCaseDecision,
        schema_name="AgentCaseDecision",
    ) == {"type": "text"}


def test_agent_case_cache_replay_is_advisory(monkeypatch, tmp_path: Path):
    executed: list[dict] = []

    class FakeStepExecutor:
        def __init__(self, page, ui_helper, elements, default_mode=None):
            self.page = page

        def execute_step(self, step):
            executed.append(step)
            if step.get("action") == "goto":
                self.page.url = step["value"]

    class FakeProvider:
        settings = type("Settings", (), {"model": "test-model"})()

        def complete_model(self, messages, response_model, **kwargs):
            assert kwargs["usage_operation"] == "runtime.agent_case"
            return response_model.model_validate(
                {"action": "done", "reason": "cached trace already satisfies criteria"}
            )

    class FakePage:
        url = "about:blank"

    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.StepExecutor",
        FakeStepExecutor,
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.ChatCompletionProvider",
        FakeProvider,
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.load_ai_config",
        lambda: {
            "runtime": {
                "agent_case_cache_enabled": True,
                "ai_cache_sqlite_path": str(tmp_path / "ai_cache.sqlite3"),
                "agent_case_cache_max_replay_steps": 10,
                "candidate_limit": 5,
            },
            "agent_policy": {
                "limits": {
                    "max_steps": 5,
                    "max_model_calls": 2,
                    "max_duration_seconds": 30,
                },
                "guardrails": {"require_checkpoints_or_final": True},
            },
            "generation": {"max_context_items": 5},
            "prompts": {"agent_case_version": "agent-case-test"},
                "llm": {"schema_version": "schema-test"},
            },
        )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor._unmet_final_criteria",
        lambda **kwargs: [],
    )

    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://www.saucedemo.com/",
        elements={},
        modules={},
        variables={},
        test_cases=[],
        test_data={},
    )
    executor = AgentCaseExecutor(
        page=FakePage(),
        ui_helper=object(),
        elements={},
        context=context,
    )
    case_data = {
        "type": "agent_case",
        "intent": "打开 https://www.saucedemo.com/ 并完成流程",
        "final": ["完成"],
    }
    spec = executor._agent_spec(case_name="test_agent", case_data=case_data)
    cache_key = executor._cache_key(case_name="test_agent", spec=spec)
    AgentCaseAdvisoryCache(tmp_path / "ai_cache.sqlite3").save_trace(
        key=cache_key,
        project="demo",
        env="prod",
        case_name="test_agent",
        intent=spec["intent"],
        steps=spec["steps"],
        inputs=spec["inputs"],
        trace=[
            {"step": {"action": "goto", "value": "https://www.saucedemo.com/"}},
            {"step": {"action": "click", "target": "Login button", "mode": "smart"}},
        ],
        final_reason="previous run passed",
    )

    result = executor.run(case_name="test_agent", case_data=case_data)

    assert result.cache_replayed_steps == 2
    assert result.model_calls == 1
    assert executed == [
        {"action": "goto", "value": "https://www.saucedemo.com/"},
        {"action": "click", "target": "Login button", "mode": "smart"},
    ]


def test_test_signature_uses_standard_page_fixtures():
    params = list(build_test_signature([]).parameters)

    assert params[:4] == ["page", "ui_helper", "get_test_name", "value"]


def test_element_definition_store_updates_last_effective_elements_file(
    tmp_path: Path,
):
    project_dir = tmp_path / "demo"
    elements_dir = project_dir / "elements"
    elements_dir.mkdir(parents=True)
    base_file = elements_dir / "base.yaml"
    override_file = elements_dir / "override.yaml"
    base_file.write_text(
        "elements:\n  login_button: '#old-base'\n",
        encoding="utf-8",
    )
    override_file.write_text(
        "elements:\n  login_button: '#old-override'\n",
        encoding="utf-8",
    )

    result = ElementDefinitionStore(project_dir).update_selector(
        "login_button",
        "#new-login",
    )

    assert result.updated is True
    assert result.path == override_file
    assert "login_button: '#old-base'" in base_file.read_text(encoding="utf-8")
    assert "#new-login" in override_file.read_text(encoding="utf-8")


def test_smart_resolver_marks_selector_self_healing_metadata():
    class FakeLocator:
        def __init__(self, selector):
            self.selector = selector

        @property
        def first(self):
            return self

        def wait_for(self, state, timeout):
            if self.selector != 'button[data-test="login"]':
                raise TimeoutError("not found")

        def is_enabled(self):
            return True

        def evaluate(self, script):
            if "cssEscape" in script:
                return self.selector
            if "tagName" in script:
                return {
                    "tag": "button",
                    "data_test": "login",
                    "text": "Login",
                    "type": "button",
                }
            return self.selector

    class FakePage:
        url = "https://example.test/login"

        def locator(self, selector):
            return FakeLocator(selector)

        def evaluate(self, script, limit):
            return [
                {
                    "index": 0,
                    "tag": "button",
                    "selector": 'button[data-test="login"]',
                    "data_test": "login",
                    "text": "Login",
                    "visible": True,
                    "enabled": True,
                }
            ]

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None

    resolved = resolver.resolve(
        action="click",
        selector="#old-login",
        target="登录按钮",
        mode="smart",
        timeout=1000,
    )

    assert resolved.selector == 'button[data-test="login"]'
    assert resolved.source == "heuristic"
    assert resolved.healed is True
    assert resolved.healing_attempted is True
    assert resolved.original_selector == "#old-login"
    assert "not found" in (resolved.original_error or "")


def test_smart_resolver_can_disable_selector_registry_by_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("UI_AI_DISABLE_SELECTOR_REGISTRY", "1")
    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.load_ai_config",
        lambda: {
            "selector_registry": {
                "enabled": True,
                "sqlite_path": str(tmp_path / "selectors.db"),
            }
        },
    )

    resolver = SmartResolver(page=object(), project="demo", env="prod")

    assert resolver.registry_enabled is False
    assert resolver.registry is None


def test_smart_resolver_rejects_unresolved_internal_element_id_target():
    class FakePage:
        url = "https://example.test/"

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None

    with pytest.raises(ValueError, match="未解析的内部element_id不能直接用于语义定位: e2"):
        resolver.resolve(
            action="fill",
            target="e2",
            selector=None,
            mode="smart",
            timeout=1000,
        )


def test_ai_step_fast_path_simplifies_header_position_instruction(monkeypatch):
    class FakePage:
        url = "https://example.test/dashboard"

    def fake_semantic_selectors(page, target, action, limit):
        if target == "primary action link":
            return ['a[data-test="primary-action"]']
        return []

    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.semantic_selectors",
        fake_semantic_selectors,
    )
    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.heuristic_selectors",
        lambda target, action: [],
    )
    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.verify_selector",
        lambda page, selector, action, timeout: True,
    )
    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.selector_matches_target",
        lambda page, selector, target, action: True,
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None
    operation = resolver._resolve_ai_step_fast(
        instruction="Click the primary action link in the top-right header.",
        timeout=1000,
    )

    assert operation is not None
    assert operation.source == "ai_step_fast"
    assert operation.selector == 'a[data-test="primary-action"]'


def test_step_executor_persists_healed_element_selector(
    monkeypatch,
    tmp_path: Path,
):
    project_dir = tmp_path / "demo"
    elements_dir = project_dir / "elements"
    elements_dir.mkdir(parents=True)
    elements_file = elements_dir / "login.yaml"
    elements_file.write_text(
        "elements:\n  login_button: '#old-login'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_DIR", str(project_dir))
    calls: list[tuple[str, str, str | None]] = []

    class FakeResolver:
        def resolve(self, **kwargs):
            assert kwargs["target"] == "login_button"
            return ResolvedSelector(
                selector="#login-button",
                source="heuristic",
                healed=True,
                healing_attempted=True,
                original_selector="#old-login",
                original_error="not found",
                confidence=0.95,
            )

    class FakePage:
        url = "https://example.test/login"

    class FakeUiHelper:
        pass

    def fake_execute_action_with_command(ui_helper, action, selector, value, step):
        calls.append((action, selector, value))

    monkeypatch.setattr(
        "src.step_actions.step_executor.execute_action_with_command",
        fake_execute_action_with_command,
    )

    executor = StepExecutor(
        FakePage(),
        FakeUiHelper(),
        elements={"login_button": "#old-login"},
    )
    executor.smart_resolver = FakeResolver()

    executor.execute_step(
        {
            "action": "click",
            "selector": "login_button",
            "mode": "smart",
        }
    )
    for thread in executor._healing_threads:
        thread.join(timeout=3)

    assert calls == [("click", "#login-button", None)]
    assert executor.elements["login_button"] == "#login-button"
    assert "#login-button" in elements_file.read_text(encoding="utf-8")


def test_step_executor_does_not_block_when_healed_element_persist_fails(
    monkeypatch,
):
    calls: list[tuple[str, str, str | None]] = []

    class FakeResolver:
        def resolve(self, **kwargs):
            assert kwargs["target"] == "login_button"
            return ResolvedSelector(
                selector="#login-button",
                source="heuristic",
                healed=True,
                healing_attempted=True,
                original_selector="#old-login",
                original_error="not found",
                confidence=0.95,
            )

    class FailingStore:
        def update_selector(self, key, new_selector):
            raise RuntimeError("disk denied")

    class FakePage:
        url = "https://example.test/login"

    class FakeUiHelper:
        pass

    def fake_execute_action_with_command(ui_helper, action, selector, value, step):
        calls.append((action, selector, value))

    monkeypatch.setattr(
        "src.step_actions.step_executor.execute_action_with_command",
        fake_execute_action_with_command,
    )

    executor = StepExecutor(
        FakePage(),
        FakeUiHelper(),
        elements={"login_button": "#old-login"},
    )
    executor.smart_resolver = FakeResolver()
    executor.element_store = FailingStore()

    executor.execute_step(
        {
            "action": "click",
            "selector": "login_button",
            "mode": "smart",
        }
    )
    for thread in executor._healing_threads:
        thread.join(timeout=3)

    assert calls == [("click", "#login-button", None)]
    assert executor.elements["login_button"] == "#old-login"


def test_missing_variable_reference_fails_fast():
    manager = VariableManager()
    manager.reset()

    with pytest.raises(KeyError, match="missing"):
        manager.replace_variables_refactored("value=${missing}")


def test_yaml_merge_is_ordered_and_type_checked(tmp_path: Path):
    (tmp_path / "01.yaml").write_text(
        "root:\n  a: {}\nitems:\n  - 1\n",
        encoding="utf-8",
    )
    (tmp_path / "02.yaml").write_text(
        "root:\n  b: 0\nitems:\n  - 2\n",
        encoding="utf-8",
    )

    merged = YamlHandler().merge_yaml_files(tmp_path)

    assert list(merged.keys()) == ["root", "items"]
    assert merged["root"] == {"a": {}, "b": 0}
    assert merged["items"] == [1, 2]


def test_safe_expression_allows_math_and_blocks_unapproved_calls():
    assert safe_eval_expression("1 + 2 * 3") == 7

    with pytest.raises(SafeExpressionError):
        safe_eval_expression("__import__('os').system('echo blocked')")


def test_ai_json_parser_validates_contract():
    payload = parse_json_object(
        'prefix {"selector":"#submit","selector_type":"css","confidence":0.9}',
        required_keys={"selector"},
        allowed_keys={"selector", "selector_type", "confidence"},
    )

    assert payload["selector"] == "#submit"

    with pytest.raises(ValueError, match="未声明字段"):
        parse_json_object(
            '{"selector":"#submit","debug":true}',
            required_keys={"selector"},
            allowed_keys={"selector"},
        )


def test_ai_structured_response_format_uses_json_schema_when_enabled():
    settings = LLMSettings(
        url="http://llm.test/chat/completions",
        api_key="test-key",
        model="test-model",
        response_format="json_schema",
    )

    response_format = build_response_format(
        settings=settings,
        response_json=True,
        response_model=SelectorDecision,
        schema_name="SelectorDecision",
    )

    assert response_format is not None
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["name"] == "SelectorDecision"
    assert "selector" in response_format["json_schema"]["schema"]["properties"]


def test_openai_strict_schema_requires_object_fields_and_removes_defaults():
    schema = openai_strict_schema(AiStepDecision)

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "status",
        "action",
        "element_id",
        "selector",
        "value",
        "key",
        "wait_ms",
        "reason",
        "expected",
        "confidence",
    }
    assert "default" not in str(schema)


def test_token_usage_normalizes_cache_hit_and_miss_tokens():
    usage = normalize_token_usage(
        {
            "prompt_tokens": 120,
            "completion_tokens": 40,
            "total_tokens": 160,
            "prompt_tokens_details": {"cached_tokens": 32},
            "completion_tokens_details": {"reasoning_tokens": 7},
        },
        provider="chat_completions",
        model="gpt-test",
    )

    assert usage["provider"] == "chat_completions"
    assert usage["model"] == "gpt-test"
    assert usage["usage_available"] is True
    assert usage["input_tokens"] == 120
    assert usage["output_tokens"] == 40
    assert usage["total_tokens"] == 160
    assert usage["cached_input_tokens"] == 32
    assert usage["uncached_input_tokens"] == 88
    assert usage["reasoning_tokens"] == 7


def test_chat_completion_provider_records_token_usage(monkeypatch, tmp_path: Path):
    tracker = TokenUsageTracker(tmp_path)
    tracker.start_run(run_kind="pytest", metadata={"project": "demo"})

    class FakeResponse:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"selector":"#submit","selector_type":"css","confidence":0.9}'
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 90,
                    "completion_tokens": 20,
                    "total_tokens": 110,
                    "prompt_tokens_details": {"cached_tokens": 10},
                },
            }

    monkeypatch.setattr(
        "src.ai_runtime.provider.requests.post", lambda *a, **k: FakeResponse()
    )
    monkeypatch.setattr(
        "src.ai_runtime.provider.get_token_usage_tracker", lambda: tracker
    )

    provider = ChatCompletionProvider(
        LLMSettings(
            url="http://llm.test/chat/completions",
            api_key="test-key",
            model="gpt-test",
        )
    )
    content = provider.complete(
        [{"role": "user", "content": "hello"}],
        usage_operation="runtime.resolve_selector",
        usage_metadata={"schema_name": "SelectorDecision"},
    )

    snapshot = tracker.snapshot()

    assert "selector" in content
    assert snapshot is not None
    assert snapshot["totals"]["call_count"] == 1
    assert snapshot["totals"]["input_tokens"] == 90
    assert snapshot["totals"]["output_tokens"] == 20
    assert snapshot["totals"]["total_tokens"] == 110
    assert snapshot["totals"]["cached_input_tokens"] == 10
    assert snapshot["totals"]["uncached_input_tokens"] == 80
    assert snapshot["events"][0]["operation"] == "runtime.resolve_selector"
    assert snapshot["events"][0]["metadata"]["schema_name"] == "SelectorDecision"
    model_io_dir = tmp_path.parent / "model_io" / snapshot["run_id"]
    model_io_files = list(model_io_dir.glob("*.json"))
    assert len(model_io_files) == 1
    model_io = json.loads(model_io_files[0].read_text(encoding="utf-8"))
    assert model_io["operation"] == "runtime.resolve_selector"
    assert model_io["request"]["messages"][0]["content"] == "hello"
    assert (
        model_io["response"]["choices"][0]["message"]["content"]
        == '{"selector":"#submit","selector_type":"css","confidence":0.9}'
    )


def test_token_usage_cleans_model_io_on_success_and_keeps_on_failure(tmp_path: Path):
    tracker = TokenUsageTracker(tmp_path / "token_usage")
    run_id = tracker.start_run(run_kind="generate_case")
    kept_path = Path(
        tracker.record_model_io(
            operation="generation.case_generation",
            request_payload={"messages": ["prompt"]},
            response_payload={"content": "{}"},
        )
    )
    assert kept_path.exists()

    summary = tracker.finish_run(status="failed")
    assert kept_path.exists()
    assert summary["model_io_dir"] == str(tmp_path / "model_io" / run_id)

    run_id = tracker.start_run(run_kind="generate_case")
    cleaned_path = Path(
        tracker.record_model_io(
            operation="generation.case_generation",
            request_payload={"messages": ["prompt"]},
            response_payload={"content": "{}"},
        )
    )
    assert cleaned_path.exists()

    tracker.finish_run(status="passed")
    assert not (tmp_path / "model_io" / run_id).exists()


def test_ai_pydantic_contracts_reject_invalid_payloads():
    assert (
        parse_model_response(
            '{"selector":"#submit","selector_type":"css","confidence":0.9}',
            SelectorDecision,
        ).selector
        == "#submit"
    )

    with pytest.raises(ValueError, match="契约"):
        parse_model_response(
            '{"selector":"#submit","selector_type":"css","confidence":2}',
            SelectorDecision,
        )

    with pytest.raises(ValueError):
        AiStepDecision.model_validate({"action": "click"})

    with pytest.raises(ValueError):
        AiStepDecision.model_validate({"action": "press", "selector": "#submit"})

    assert AiStepDecision.model_validate(
        {"action": "reject", "reason": "multiple actions"}
    ).action == "reject"
    assert AiStepDecision.model_validate(
        {"status": "need_more_context", "reason": "候选不足"}
    ).status == "need_more_context"

    with pytest.raises(ValueError):
        AiStepDecision.model_validate({"action": "reject"})

    assert AgentCaseDecision.model_validate(
        {"action": "click", "target": "Login button"}
    ).action == "click"
    assert AgentCaseDecision.model_validate(
        {
            "status": "ok",
            "action": "click",
            "element_id": "e12",
            "reason": "点击目标按钮",
            "expected": "页面状态更新",
            "confidence": 0.9,
        }
    ).element_id == "e12"
    assert AgentCaseDecision.model_validate(
        {"status": "need_more_context", "reason": "候选中没有目标元素"}
    ).status == "need_more_context"
    assert AgentCaseDecision.model_validate(
        {"action": "done", "reason": "success criteria satisfied"}
    ).action == "done"
    with pytest.raises(ValueError):
        AgentCaseDecision.model_validate({"action": "click"})


def test_generated_case_payload_contract_normalizes_defaults():
    payload = GeneratedCasePayload.model_validate(
        {
            "cases": [{"name": "test_generated"}],
            "data": {
                "test_generated": {
                    "steps": [{"action": "assert_visible", "selector": "submit_button"}]
                }
            },
        }
    )

    assert payload.data["test_generated"].mode == "strict"
    assert payload.elements == {}


def test_generation_spec_scope_matches_project():
    _validate_spec_project_scope(
        project="demo",
        spec_path=Path("test_data/demo/generation/saucedemo_ai.yaml"),
        spec={"project": "demo"},
    )

    with pytest.raises(ValueError, match="test_data/crm/generation"):
        _validate_spec_project_scope(
            project="demo",
            spec_path=Path("test_data/crm/generation/smoke.yaml"),
            spec={},
        )

    with pytest.raises(ValueError, match="project=crm"):
        _validate_spec_project_scope(
            project="demo",
            spec_path=Path("test_data/demo/generation/smoke.yaml"),
            spec={"project": "crm"},
        )


def test_generation_spec_short_name_resolves_to_project_generation_dir(tmp_path: Path):
    spec_dir = tmp_path / "test_data" / "demo" / "generation"
    spec_dir.mkdir(parents=True)
    spec_file = spec_dir / "saucedemo_ai.yaml"
    spec_file.write_text("project: demo\ncases: []\n", encoding="utf-8")

    class Context:
        test_dir = tmp_path / "test_data" / "demo"

    assert resolve_generation_spec_path(Context(), "saucedemo_ai") == spec_file
    assert resolve_generation_spec_path(Context(), "saucedemo_ai.yaml") == spec_file


def test_generation_spec_string_steps_still_use_ai():
    natural_spec = {
        "cases": [
            {
                "name": "baidu_search_keyword",
                "steps": [
                    "打开百度网页",
                    "点击搜索输入框",
                    "输入百度",
                    "点击搜索按钮",
                ],
            }
        ]
    }
    explicit_spec = {
        "cases": [
            {
                "name": "baidu_search_keyword",
                "steps": [{"action": "goto", "value": "https://baidu.com"}],
            }
        ]
    }

    assert _has_explicit_steps(natural_spec) is False
    assert _has_explicit_steps(explicit_spec) is True


def test_generation_navigation_prefers_case_steps_then_description_then_modules():
    context = ProjectContext(
        project="demo",
        test_dir=Path("test_data/demo"),
        base_url="https://project.example/",
        elements={},
        modules={
            "login": [
                {"action": "goto", "value": "https://module.example/"},
            ]
        },
        variables={},
        test_cases=[],
        test_data={},
    )

    from_steps = _resolve_navigation_context(
        context,
        {
            "cases": [
                {
                    "name": "test_flow",
                    "description": "打开 https://description.example/ 后登录",
                    "steps": ["打开 https://steps.example/ 登录页"],
                }
            ]
        },
    )
    assert from_steps["resolved"]["source"] == "generation_spec"
    assert from_steps["resolved"]["url"] == "https://steps.example/"

    from_description = _resolve_navigation_context(
        context,
        {
            "cases": [
                {
                    "name": "test_flow",
                    "description": "打开 https://description.example/ 后登录",
                    "steps": ["使用标准用户登录"],
                }
            ]
        },
    )
    assert from_description["resolved"]["source"] == "generation_spec"
    assert from_description["resolved"]["url"] == "https://description.example/"


def test_generation_navigation_falls_back_to_module_then_project_base_url():
    module_context = ProjectContext(
        project="demo",
        test_dir=Path("test_data/demo"),
        base_url="https://project.example/",
        elements={},
        modules={"login": [{"action": "goto", "value": "https://module.example/"}]},
        variables={},
        test_cases=[],
        test_data={},
    )

    from_module = _resolve_navigation_context(
        module_context,
        {"cases": [{"name": "test_flow", "description": "标准用户登录"}]},
    )
    assert from_module["resolved"]["source"] == "module"
    assert from_module["resolved"]["module"] == "login"
    assert from_module["resolved"]["url"] == "https://module.example/"

    project_context = ProjectContext(
        project="demo",
        test_dir=Path("test_data/demo"),
        base_url="https://project.example/",
        elements={},
        modules={},
        variables={},
        test_cases=[],
        test_data={},
    )
    from_project = _resolve_navigation_context(
        project_context,
        {"cases": [{"name": "test_flow", "description": "标准用户登录"}]},
    )
    assert from_project["resolved"]["source"] == "project_config"
    assert from_project["resolved"]["url"] == "https://project.example/"


def test_entry_url_normalization_strips_query_hash_and_default_port():
    assert (
        normalize_entry_url("HTTPS://Example.TEST:443//login/?utm=1#top")
        == "https://example.test/login"
    )
    assert (
        normalize_entry_url("/login?x=1", base_url="https://example.test/app/")
        == "https://example.test/login"
    )


def test_generation_cache_reuses_valid_payload_by_entry_scope(monkeypatch, tmp_path: Path):
    cache_path = tmp_path / "ai_cache.sqlite3"
    calls = {"count": 0}

    monkeypatch.setattr(
        "src.ai_generation.case_generator.load_ai_config",
        lambda: {
            "runtime": {"ai_cache_sqlite_path": str(cache_path)},
            "generation": {"max_context_items": 5, "generation_cache_enabled": True},
            "prompts": {"generation_version": "generation-test"},
            "llm": {"schema_version": "schema-test"},
        },
    )

    class FakeProvider:
        settings = type("Settings", (), {"model": "test-model"})()

        def complete_model(self, messages, response_model, **kwargs):
            calls["count"] += 1
            return response_model.model_validate(
                {
                    "cases": [{"name": "test_generated"}],
                    "data": {
                        "test_generated": {
                            "description": "generated",
                            "mode": "smart",
                            "steps": [
                                {
                                    "action": "assert_text",
                                    "selector": "page_title",
                                    "value": "Products",
                                }
                            ],
                        }
                    },
                    "elements": {},
                    "modules": {},
                    "vars": {},
                }
            )

    monkeypatch.setattr("src.ai_generation.case_generator.ChatCompletionProvider", FakeProvider)

    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://www.saucedemo.com/?utm=1",
        elements={"page_title": ".title"},
        modules={},
        variables={},
        test_cases=[],
        test_data={},
    )
    spec = {
        "cases": [
            {
                "name": "test_generated",
                "intent": "Login and check products.",
                "final": ["Products is visible"],
            }
        ]
    }
    payload = _build_payload(
        context,
        spec,
        env="prod",
        output_name="generated",
        use_ai=True,
    )
    _save_generation_cache(
        context=context,
        spec=spec,
        env="prod",
        output_name="generated",
        payload=payload,
        use_ai=True,
        progress=None,
    )
    cached = _build_payload(
        ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="https://www.saucedemo.com/",
            elements={"page_title": ".title"},
            modules={},
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec,
        env="prod",
        output_name="generated",
        use_ai=True,
    )

    assert cached == payload
    assert calls["count"] == 1


def test_generation_harness_normalizes_model_field_aliases(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={},
            modules={},
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec={"mode": "smart"},
        output_name="generated",
    )

    payload = harness.normalize(
        {
            "cases": [{"name": "test_generated"}],
            "data": {
                "test_generated": {
                    "steps": [
                        {"action": "goto", "url": "https://example.test"},
                        {
                            "action": "fill",
                            "selector": "search_input",
                            "text": "{{search_keyword}}",
                        },
                        {"action": "assert_visible", "selector": "search_button"},
                    ]
                }
            },
            "elements": {
                "search_input": {"selector": "#kw", "desc": "ignored"},
                "search_button": "#su",
            },
        }
    )

    steps = payload["data"]["test_generated"]["steps"]
    assert steps[0] == {"action": "goto", "value": "https://example.test"}
    assert steps[1]["value"] == "${search_keyword}"
    assert "text" not in steps[1]
    assert payload["elements"]["search_input"] == "#kw"


def test_generation_harness_prefixes_pytest_case_names():
    assert _safe_case_name("saucedemo_backpack_cart") == "test_saucedemo_backpack_cart"
    assert _safe_case_name("test_existing_name") == "test_existing_name"


def test_generation_harness_rejects_strict_target_only_steps(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={},
            modules={},
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec={},
        output_name="generated",
    )
    payload = harness.normalize(
        {
            "cases": [{"name": "test_generated"}],
            "data": {
                "test_generated": {
                    "mode": "strict",
                    "steps": [
                        {"action": "click", "target": "登录按钮"},
                        {"action": "assert_visible", "target": "首页"},
                    ],
                }
            },
        }
    )

    with pytest.raises(ValueError, match="mode: smart"):
        harness.validate(payload)


def test_generation_harness_rejects_empty_assertion_expected(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"message": "#message"},
            modules={},
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec={},
        output_name="generated",
    )
    payload = harness.normalize(
        {
            "cases": [{"name": "test_generated"}],
            "data": {
                "test_generated": {
                    "mode": "strict",
                    "steps": [
                        {"action": "assert_text", "selector": "message", "value": " "}
                    ],
                }
            },
        }
    )

    with pytest.raises(ValueError, match="断言期望值不能为空"):
        harness.validate(payload)


def test_generation_harness_normalizes_module_steps_and_params(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"username": "#user-name", "title": ".title"},
            modules={},
            variables={"standard_username": "standard_user"},
            test_cases=[],
            test_data={},
        ),
        spec={"mode": "smart"},
        output_name="generated",
    )
    payload = harness.normalize(
        {
            "cases": [{"name": "test_generated"}],
            "data": {
                "test_generated": {
                    "mode": "smart",
                    "steps": [
                        {
                            "action": "use_module",
                            "target": "login",
                            "params": {"username": "${standard_username}"},
                        },
                        {"action": "assert_visible", "selector": "title"},
                    ],
                }
            },
            "modules": {
                "login": {
                    "steps": [
                        {"action": "open", "url": "https://example.test"},
                        {
                            "action": "input",
                            "selector": "username",
                            "value": "${username}",
                        },
                    ]
                }
            },
        }
    )

    assert payload["modules"]["login"][0] == {
        "action": "goto",
        "value": "https://example.test",
    }
    assert payload["modules"]["login"][1]["action"] == "fill"
    assert payload["data"]["test_generated"]["steps"][0]["params"] == {
        "username": "${standard_username}"
    }
    assert harness.validate(payload) == []


def test_generation_harness_normalizes_model_dict_scalar_fields(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"title": ".title"},
            modules={"login": [{"action": "click", "selector": "title"}]},
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec={"mode": "smart"},
        output_name="generated",
    )
    payload = harness.normalize(
        {
            "cases": [{"name": "test_generated"}],
            "data": {
                "test_generated": {
                    "mode": "smart",
                    "steps": [
                        {"use_module": {"name": "login"}},
                        {
                            "action": {"action": "assert_visible"},
                            "selector": {"key": "title", "selector": ".title"},
                        },
                    ],
                }
            },
        }
    )

    assert payload["data"]["test_generated"]["steps"][0] == {"use_module": "login"}
    assert payload["data"]["test_generated"]["steps"][1] == {
        "action": "assert_visible",
        "selector": "title",
    }
    assert harness.validate(payload) == []


def test_generation_write_payload_restores_files_when_post_verify_fails(tmp_path: Path):
    case_file = tmp_path / "cases" / "generated.yaml"
    data_file = tmp_path / "data" / "generated.yaml"
    case_file.parent.mkdir(parents=True)
    data_file.parent.mkdir(parents=True)
    case_file.write_text("old cases\n", encoding="utf-8")
    data_file.write_text("old data\n", encoding="utf-8")

    result = {
        "case_file": case_file,
        "data_file": data_file,
        "payload": {
            "cases": [{"name": "test_generated"}],
            "data": {"test_generated": {"steps": []}},
        },
    }

    with pytest.raises(RuntimeError, match="post verify failed"):
        _write_payload(
            result,
            overwrite=True,
            verify=lambda: (_ for _ in ()).throw(RuntimeError("post verify failed")),
        )

    assert case_file.read_text(encoding="utf-8") == "old cases\n"
    assert data_file.read_text(encoding="utf-8") == "old data\n"


def test_generation_artifacts_store_debug_data_and_cleanup(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    artifacts = _GenerationArtifacts(project="demo", spec_name="saucedemo ai")

    artifacts.write_json("payload.json", {"selector": {"key": "login_button"}})
    artifacts.write_text("error.txt", "failed")

    assert (artifacts.path / "payload.json").exists()
    assert (artifacts.path / "error.txt").read_text(encoding="utf-8") == "failed"

    artifacts.cleanup()

    assert not artifacts.path.exists()


def test_generation_harness_rejects_missing_module_params(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"username": "#user-name", "title": ".title"},
            modules={},
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec={"mode": "smart"},
        output_name="generated",
    )
    payload = harness.normalize(
        {
            "cases": [{"name": "test_generated"}],
            "data": {
                "test_generated": {
                    "mode": "smart",
                    "steps": [
                        {"use_module": "login"},
                        {"action": "assert_visible", "selector": "title"},
                    ],
                }
            },
            "modules": {
                "login": [
                    {"action": "fill", "selector": "username", "value": "${username}"}
                ]
            },
        }
    )

    with pytest.raises(ValueError, match="params: username"):
        harness.validate(payload)


def test_generation_harness_rejects_missing_context_module_params(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"username": "#user-name", "title": ".title"},
            modules={
                "login": [
                    {"action": "fill", "selector": "username", "value": "${username}"}
                ]
            },
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec={"mode": "smart"},
        output_name="generated",
    )
    payload = harness.normalize(
        {
            "cases": [{"name": "test_generated"}],
            "data": {
                "test_generated": {
                    "mode": "smart",
                    "steps": [
                        {"use_module": "login"},
                        {"action": "assert_visible", "selector": "title"},
                    ],
                }
            },
        }
    )

    with pytest.raises(ValueError, match="params: username"):
        harness.validate(payload)


def test_generation_harness_rejects_undefined_variables(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"username": "#user-name", "title": ".title"},
            modules={},
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec={"mode": "smart"},
        output_name="generated",
    )
    payload = harness.normalize(
        {
            "cases": [{"name": "test_generated"}],
            "data": {
                "test_generated": {
                    "mode": "smart",
                    "steps": [
                        {
                            "action": "fill",
                            "selector": "username",
                            "value": "${username_var}",
                        },
                        {"action": "assert_visible", "selector": "title"},
                    ],
                }
            },
        }
    )

    with pytest.raises(ValueError, match="username_var"):
        harness.validate(payload)


def test_local_ai_candidate_values_are_not_redacted():
    raw = "phone=13812345678 email=qa@example.com token=abcdefghijklmnopqrstuvwxyzABCDEF123456"

    assert redact_value(raw) == raw


def test_heuristic_selectors_prioritize_password_target():
    selectors = heuristic_selectors("密码输入框", "fill")

    assert selectors[0] == 'input[type="password"]'
    assert 'input[name*="password" i]' in selectors


def test_vision_find_result_contract_accepts_service_payload():
    result = VisionFindResult.model_validate(
        {
            "found": True,
            "target": "登录按钮",
            "selected_candidate_index": 1,
            "box": [10, 20, 110, 80],
            "center": [60, 50],
            "confidence": 0.91,
            "method": "ocr+vision",
            "reason": "matched",
            "extra_debug": "ignored",
        }
    )

    assert result.found is True
    assert result.selected_candidate_index == 1
    assert result.center == [60, 50]


def test_vision_client_posts_find_contract_to_lan_service():
    captured: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            captured["path"] = self.path
            captured["payload"] = json.loads(self.rfile.read(length))
            body = json.dumps(
                {
                    "found": True,
                    "selected_candidate_index": 0,
                    "center": [15, 15],
                    "confidence": 0.9,
                    "method": "mock",
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        settings = VisionSettings(
            enabled=True,
            service_url=f"http://127.0.0.1:{server.server_port}",
            min_confidence=0.8,
        )
        result = VisionClient(settings).find(
            image_bytes=b"fake-image",
            target="submit",
            action="click",
            url="http://example.test",
            candidates=[{"index": 0, "selector": "#submit"}],
        )
    finally:
        server.shutdown()
        thread.join(timeout=3)

    assert result.found is True
    assert captured["path"] == "/v1/ui/find"
    assert captured["payload"]["target"] == "submit"
    assert captured["payload"]["dom_candidates"][0]["selector"] == "#submit"


def test_vision_resolver_maps_visual_center_back_to_dom_selector():
    class FakeLocator:
        @property
        def first(self):
            return self

        def wait_for(self, state, timeout):
            return None

        def is_enabled(self):
            return True

    class FakePage:
        url = "http://example.test"

        def screenshot(self, **kwargs):
            return b"fake-image"

        def locator(self, selector):
            return FakeLocator()

    class FakeClient:
        def find(self, **kwargs):
            return VisionFindResult(
                found=True,
                center=[15, 15],
                confidence=0.92,
                method="mock",
                reason="center matched",
            )

    candidates = [
        {
            "index": 0,
            "selector": "#submit",
            "bbox": [10, 10, 30, 30],
            "center": [20, 20],
        }
    ]
    settings = VisionSettings(enabled=True, service_url="http://vision.test")

    resolved = VisionResolver(
        FakePage(),
        settings=settings,
        client=FakeClient(),
    ).resolve(action="click", target="submit", timeout=1000, candidates=candidates)

    assert resolved.selector == "#submit"
    assert resolved.source == "vision_dom"


def test_smart_resolver_prefers_enabled_vision_before_llm(monkeypatch):
    class FakeLocator:
        @property
        def first(self):
            return self

        def wait_for(self, state, timeout):
            return None

        def is_enabled(self):
            return True

    class FakePage:
        url = "http://example.test/form"

        def locator(self, selector):
            return FakeLocator()

        def evaluate(self, script, limit):
            return []

    class ForbiddenProvider:
        def __init__(self, *args, **kwargs):
            raise AssertionError("LLM should not be called before enabled vision")

    class FakeVisionResolver:
        def __init__(self, page, *, settings):
            self.page = page
            self.settings = settings

        def resolve(self, **kwargs):
            return VisionResolution(
                selector="#submit",
                source="vision_dom",
                confidence=0.9,
                method="mock",
                reason="visual matched",
            )

    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.load_vision_settings",
        lambda config: VisionSettings(enabled=True, service_url="http://vision.test"),
    )
    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.ChatCompletionProvider", ForbiddenProvider
    )
    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.VisionResolver", FakeVisionResolver
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None
    resolved = resolver.resolve(
        action="click",
        target="primary action",
        selector=None,
        mode="smart",
        timeout=1000,
    )

    assert resolved.selector == "#submit"
    assert resolved.source == "vision_dom"


def test_smart_resolver_rejects_registry_semantic_mismatch(tmp_path: Path):
    snapshots = {
        "#user-name": {
            "selector": "#user-name",
            "tag": "input",
            "id": "user-name",
            "name": "user-name",
            "placeholder": "Username",
            "type": "text",
        },
        "#password": {
            "selector": "#password",
            "tag": "input",
            "id": "password",
            "name": "password",
            "placeholder": "Password",
            "type": "password",
        },
    }

    class FakeLocator:
        def __init__(self, selector):
            self.selector = selector

        @property
        def first(self):
            return self

        def wait_for(self, state, timeout):
            if self.selector not in snapshots:
                raise TimeoutError("not found")

        def is_enabled(self):
            return True

        def evaluate(self, script):
            if "return tag === 'input'" in script:
                return self.selector in snapshots
            if "const cssEscape" in script:
                return self.selector
            snapshot = dict(snapshots[self.selector])
            snapshot.setdefault("class_name", "")
            snapshot.setdefault("data_test", "")
            snapshot.setdefault("data_testid", "")
            snapshot.setdefault("text", "")
            snapshot.setdefault("value", "")
            snapshot.setdefault("role", "")
            snapshot.setdefault("aria_label", "")
            snapshot.setdefault("title", "")
            snapshot.setdefault("label", "")
            snapshot.setdefault("ancestor_text", "")
            return snapshot

    class FakePage:
        url = "https://www.saucedemo.com/"

        def locator(self, selector):
            return FakeLocator(selector)

        def evaluate(self, script, limit):
            return [
                {
                    "index": 0,
                    "selector": "#user-name",
                    "tag": "input",
                    "id": "user-name",
                    "name": "user-name",
                    "placeholder": "Username",
                    "type": "text",
                },
                {
                    "index": 1,
                    "selector": "#password",
                    "tag": "input",
                    "id": "password",
                    "name": "password",
                    "placeholder": "Password",
                    "type": "password",
                },
            ]

    registry = SelectorRegistry(tmp_path / "selectors.db")
    registry.save(
        project="demo",
        env="test",
        page_key="https://www.saucedemo.com/",
        action="fill",
        target="密码输入框",
        selector="#user-name",
        source="heuristic",
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = registry

    resolved = resolver.resolve(
        action="fill",
        target="密码输入框",
        selector=None,
        mode="smart",
        timeout=1000,
    )

    assert resolved.selector == "#password"
    assert (
        registry.find(
            project="demo",
            env="test",
            page_key="https://www.saucedemo.com/",
            action="fill",
            target="密码输入框",
        ).selector
        == "#password"
    )


def test_smart_resolver_uses_vision_after_llm_selector_failure(monkeypatch):
    class FakeLocator:
        def __init__(self, selector):
            self.selector = selector

        @property
        def first(self):
            return self

        def wait_for(self, state, timeout):
            if self.selector != "#submit":
                raise TimeoutError("not found")

        def is_enabled(self):
            return True

    class FakePage:
        url = "http://example.test/form"

        def locator(self, selector):
            return FakeLocator(selector)

        def evaluate(self, script, limit):
            return [{"index": 0, "selector": "#submit", "bbox": [1, 1, 10, 10]}]

    class FailingProvider:
        def complete_model(self, *args, **kwargs):
            raise ValueError("llm failed")

    class FakeVisionResolver:
        def __init__(self, page, *, settings):
            self.page = page
            self.settings = settings

        def resolve(self, **kwargs):
            return VisionResolution(
                selector="#submit",
                source="vision_dom",
                confidence=0.9,
                method="mock",
                reason="visual matched",
            )

    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.load_vision_settings",
        lambda config: VisionSettings(enabled=True, service_url="http://vision.test"),
    )
    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.ChatCompletionProvider", FailingProvider
    )
    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.VisionResolver", FakeVisionResolver
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None
    resolved = resolver.resolve(
        action="click",
        target="保存",
        selector=None,
        mode="smart",
        timeout=1000,
    )

    assert resolved.selector == "#submit"
    assert resolved.source == "vision_dom"
    assert resolved.vision_reason == "visual matched"


def test_smart_resolver_skips_unavailable_vision_service(monkeypatch):
    class FakePage:
        url = "http://example.test/form"

        def locator(self, selector):
            class FakeLocator:
                @property
                def first(self):
                    return self

                def wait_for(self, state, timeout):
                    raise TimeoutError("not found")

            return FakeLocator()

        def evaluate(self, script, limit):
            return []

    class FailingProvider:
        def complete_model(self, *args, **kwargs):
            raise ValueError("llm failed")

    class UnavailableVisionResolver:
        def __init__(self, page, *, settings):
            pass

        def resolve(self, **kwargs):
            raise VisionServiceUnavailable("service down")

    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.load_vision_settings",
        lambda config: VisionSettings(
            enabled=True, service_url="http://127.0.0.1:59999"
        ),
    )
    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.ChatCompletionProvider", FailingProvider
    )
    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.VisionResolver", UnavailableVisionResolver
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None

    with pytest.raises(ValueError, match="llm failed"):
        resolver.resolve(
            action="click",
            target="保存",
            selector=None,
            mode="smart",
            timeout=1000,
        )


def test_selector_registry_persists_ai_decision_metadata(tmp_path: Path):
    registry = SelectorRegistry(tmp_path / "selectors.db")
    try:
        record = registry.save(
            project="demo",
            env="test",
            page_key="/login",
            action="click",
            target="submit",
            selector="#submit",
            source="ai_selector",
            confidence=0.73,
            prompt_version="selector-v1",
            schema_version="schema-v1",
            model="test-model",
            candidate_hash="abc123",
            candidate_count=7,
        )

        found = registry.find(
            project="demo",
            env="test",
            page_key="/login",
            action="click",
            target="submit",
        )
        assert found is not None
        assert found.id == record.id
        assert found.prompt_version == "selector-v1"
        assert found.schema_version == "schema-v1"
        assert found.model == "test-model"
        assert found.candidate_hash == "abc123"
        assert found.candidate_count == 7

        registry.mark_failed(
            found.id,
            unstable_threshold=1,
            last_error="verification failed",
        )
        failed = registry.find(
            project="demo",
            env="test",
            page_key="/login",
            action="click",
            target="submit",
        )
        assert failed is not None
        assert failed.status == "unstable"
        assert failed.last_error == "verification failed"
    finally:
        registry.close()


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


def test_yaml_schema_rejects_unknown_step_field(tmp_path: Path):
    project = _write_schema_project(
        tmp_path,
        case_steps=[
            {"action": "click", "selector": "submit_button", "variable_name": "noop"}
        ],
    )

    with pytest.raises(YamlSchemaValidationError, match="未声明字段"):
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
            "steps": [{"action": "ai_step", "instruction": "打开购物车"}],
        },
    )

    validate_project(project)


def test_yaml_schema_rejects_vision_as_action(tmp_path: Path):
    project = _write_schema_project(
        tmp_path,
        case_data={
            "description": "vision is capability, not action",
            "mode": "smart",
            "steps": [
                {
                    "action": "vision_click",
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
            "intent": "打开Saucedemo并完成下单流程",
            "checkpoints": ["购物车中存在指定商品"],
            "final": ["页面展示 Thank you for your order"],
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
                "打开 Saucedemo 登录页",
                "使用标准用户登录",
                "打开购物车页面",
            ],
            "checkpoints": ["登录后进入商品列表页"],
            "final": ["购物车页面可见"],
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
            "intent": "打开Saucedemo并完成下单流程",
            "final": ["订单完成"],
        },
    )

    with pytest.raises(YamlSchemaValidationError, match="不需要声明 mode"):
        validate_project(project)


def test_yaml_schema_rejects_agent_case_without_checkpoints_or_final(tmp_path: Path):
    project = _write_schema_project(
        tmp_path,
        case_data={
            "description": "runtime agent case",
            "type": "agent_case",
            "intent": "打开Saucedemo并完成下单流程",
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
                "intent": "旧嵌套格式",
                "final": ["订单完成"],
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
