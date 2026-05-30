import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
import requests
from ruamel.yaml import YAML

from page_objects.base_page import _url_contains
from src.ai_generation import case_generator as case_generator_module
from src.ai_generation.case_generator import (
    _GenerationArtifacts,
    _assert_effective_verification_payload,
    _build_payload,
    _default_output_name,
    _has_explicit_steps,
    _payload_from_explicit_spec,
    _payload_with_referenced_context_modules,
    _result_paths,
    _resolve_navigation_context,
    _validate_spec_project_scope,
    _write_payload,
    generate_case_files,
    resolve_generation_spec_path,
)
from src.ai_generation.harness import GenerationHarness
from src.ai_generation.harness import _safe_case_name
from src.ai_generation.pipeline import execute_compiled_payload_steps
from src.ai_generation.project_context import ProjectContext
from src.ai_runtime.cache_scope import normalize_entry_url
from src.ai_runtime.agent_case_executor import (
    AgentCasePlanCache,
    AgentCaseCompileContractError,
    AgentDecisionRejected,
    AgentCaseExecutor,
    _agent_spec_to_generation_spec,
    _cacheable_plan_steps,
    _completion_wait_seconds_for_step,
    _first_url,
    _observable_completion_terms,
    _parse_agent_decision_response,
    _remaining_step_hints,
    _runtime_feedback,
    _runtime_harness_state,
    _text_query_from_selector,
    _validate_runtime_compiled_payload,
    _verified_target_candidates,
    _selector_for_equivalent_dom_text,
    _unmet_completion_criteria,
    _unmet_final_criteria,
    _unmet_intent_action_requirements,
    _runtime_unmet_requirements,
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
    build_locator_context,
    compact_model_dom_context,
    compact_dom_candidates,
    compact_history,
    selector_for_element_id,
)
from src.ai_runtime.playwright_selectors import (
    collect_candidates,
    heuristic_selectors,
    is_high_quality_selector,
    redact_value,
    semantic_selectors,
    selector_matches_target,
    validate_selector,
)
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
from src.step_actions import step_executor as step_executor_module
from src.step_actions.commands.wait_commands import WaitForElementTextCommand
from src.step_actions.utils import _resolve_allowed_script_path
from src.yaml_schema import (
    ValidationContext,
    YamlSchemaValidationError,
    validate_case_file,
    validate_pytest_targets,
    validate_project,
)
from utils.variable_manager import VariableManager
from utils.yaml_handler import YamlHandler
from utils.token_usage import TokenUsageTracker, normalize_token_usage
from utils.config import Config


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


def test_assert_title_contains_command_logs_expected_and_actual(monkeypatch):
    messages: list[str] = []

    class FakeLogger:
        def info(self, message: str):
            messages.append(message)

    class FakeVariableManager:
        def replace_variables_refactored(self, value):
            return "bmw x3"

    class FakePage:
        def title(self):
            return "[bmw x3] latest info"

    class FakeUiHelper:
        page = FakePage()
        variable_manager = FakeVariableManager()

        def assert_title_contains(self, expected: str):
            assert expected == "${keyword}"

    monkeypatch.setattr(assertion_commands_module, "logger", FakeLogger())

    assertion_commands_module.AssertTitleContainsCommand().execute(
        FakeUiHelper(),
        None,
        None,
        {"expected": "${keyword}"},
    )

    assert "action=assert_title_contains" in messages[0]
    assert "预期结果=bmw x3" in messages[0]
    assert "实际结果=[bmw x3] latest info" in messages[0]


def test_native_ai_step_executes_through_command_pipeline(monkeypatch):
    calls: list[tuple[str, str, str | None]] = []

    class FakeResolver:
        def resolve_ai_step(self, *, instruction: str, timeout: int):
            assert instruction == "open cart"
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
    executor.execute_step({"action": "ai_step", "instruction": "open cart"})

    assert calls == [("click", "#cart", None)]


def test_step_executor_waits_for_stable_after_state_changing_action(monkeypatch):
    calls: list[Any] = []

    def fake_execute_action_with_command(ui_helper, action, selector, value, step):
        calls.append(("execute", action, selector, value))

    class FakePage:
        url = "https://example.test"

    class FakeUiHelper:
        def wait_for_stable(self, *, timeout: int, idle_ms: int):
            calls.append(("stable", timeout, idle_ms))

    monkeypatch.setattr(
        "src.step_actions.step_executor.execute_action_with_command",
        fake_execute_action_with_command,
    )

    executor = StepExecutor(FakePage(), FakeUiHelper(), elements={})
    executor.ai_config.setdefault("runtime", {})["action_stable_timeout_ms"] = 1234
    executor.ai_config["runtime"]["action_stable_idle_ms"] = 321

    executor.execute_step({"action": "click", "selector": "#go"})

    assert calls == [
        ("execute", "click", "#go", None),
        ("stable", 1234, 321),
    ]


def test_step_executor_stable_wait_can_be_skipped_for_non_mutating_steps(
    monkeypatch,
):
    calls: list[Any] = []

    def fake_execute_action_with_command(ui_helper, action, selector, value, step):
        calls.append(("execute", action))

    class FakePage:
        url = "https://example.test"

    class FakeUiHelper:
        def wait_for_stable(self, *, timeout: int, idle_ms: int):
            calls.append(("stable", timeout, idle_ms))

    monkeypatch.setattr(
        "src.step_actions.step_executor.execute_action_with_command",
        fake_execute_action_with_command,
    )

    executor = StepExecutor(FakePage(), FakeUiHelper(), elements={})
    executor.execute_step({"action": "assert_url_contains", "value": "done"})
    executor.execute_step(
        {"action": "click", "selector": "#go", "wait_for_stable": False}
    )

    assert calls == [
        ("execute", "assert_url_contains"),
        ("execute", "click"),
    ]


def test_wait_for_element_text_command_uses_base_page_expected_argument():
    calls: list[Any] = []

    class FakeUiHelper:
        def wait_for_element_text(self, *, selector: str, expected: str, timeout: int):
            calls.append((selector, expected, timeout))

    WaitForElementTextCommand().execute(
        FakeUiHelper(),
        "#message",
        "ready",
        {"timeout": 1234},
    )

    assert calls == [("#message", "ready", 1234)]


def test_step_executor_records_registry_selector_after_verified_action(monkeypatch):
    events: list[Any] = []

    class FakeResolver:
        def resolve(self, **kwargs):
            events.append(("resolve", kwargs["target"]))
            return ResolvedSelector(
                selector="#login",
                source="heuristic",
                confidence=0.9,
                cache_action="click",
                cache_target=kwargs["target"],
                cache_page_key="https://example.test/login",
            )

        def record_verified_selector(self, **kwargs):
            events.append(("cache", kwargs))

    class FakePage:
        url = "https://example.test/login"

    class FakeUiHelper:
        pass

    def fake_execute_action_with_command(ui_helper, action, selector, value, step):
        events.append(("execute", action, selector, value))

    monkeypatch.setattr(
        "src.step_actions.step_executor.execute_action_with_command",
        fake_execute_action_with_command,
    )
    monkeypatch.setenv("UI_SELECTOR_CACHE_COMMIT_MODE", "immediate")

    executor = StepExecutor(FakePage(), FakeUiHelper(), elements={})
    executor.smart_resolver = FakeResolver()

    executor.execute_step(
        {"action": "click", "target": "login button", "mode": "smart"}
    )

    assert events[0] == ("resolve", "login button")
    assert events[1] == ("execute", "click", "#login", None)
    assert events[2][0] == "cache"
    assert events[2][1]["target"] == "login button"
    assert events[2][1]["selector"] == "#login"


def test_step_executor_uses_resolved_element_selector_as_smart_target_without_explicit_target(
    monkeypatch,
):
    events: list[Any] = []

    class FakeResolver:
        def resolve(self, **kwargs):
            events.append(("resolve", kwargs["selector"], kwargs["target"]))
            return ResolvedSelector(
                selector='button:has-text("Submit")',
                source="heuristic",
                confidence=0.9,
                cache_action="click",
                cache_target=kwargs["target"],
                cache_page_key="https://example.test/list",
            )

        def record_verified_selector(self, **kwargs):
            events.append(("cache", kwargs))

    class FakePage:
        url = "https://example.test/list"

    class FakeUiHelper:
        pass

    def fake_execute_action_with_command(ui_helper, action, selector, value, step):
        events.append(("execute", action, selector, value))

    monkeypatch.setenv("UI_SELECTOR_CACHE_COMMIT_MODE", "immediate")
    monkeypatch.setattr(
        "src.step_actions.step_executor.execute_action_with_command",
        fake_execute_action_with_command,
    )

    executor = StepExecutor(
        FakePage(),
        FakeUiHelper(),
        elements={"action_button": 'button:has-text("Submit")'},
    )
    executor.smart_resolver = FakeResolver()

    executor.execute_step({"action": "click", "selector": "action_button", "mode": "smart"})

    assert events[0] == (
        "resolve",
        'button:has-text("Submit")',
        'button:has-text("Submit")',
    )
    assert events[1] == ("execute", "click", 'button:has-text("Submit")', None)


def test_step_executor_defers_selector_cache_until_test_passes(monkeypatch):
    events: list[Any] = []

    class FakeResolver:
        def resolve(self, **kwargs):
            return ResolvedSelector(
                selector="#login",
                source="heuristic",
                confidence=0.9,
                cache_action="click",
                cache_target=kwargs["target"],
                cache_page_key="https://example.test/login",
            )

        def record_verified_selector(self, **kwargs):
            events.append(("cache", kwargs))

    class FakePage:
        url = "https://example.test/login"

    class FakeUiHelper:
        pass

    def fake_execute_action_with_command(ui_helper, action, selector, value, step):
        events.append(("execute", action, selector, value))

    monkeypatch.setenv("UI_SELECTOR_CACHE_COMMIT_MODE", "deferred")
    monkeypatch.setattr(
        "src.step_actions.step_executor.execute_action_with_command",
        fake_execute_action_with_command,
    )

    step_executor_module.discard_pending_selector_cache("test setup")
    executor = StepExecutor(FakePage(), FakeUiHelper(), elements={})
    executor.smart_resolver = FakeResolver()

    executor.execute_step(
        {"action": "click", "target": "login button", "mode": "smart"}
    )

    assert events == [("execute", "click", "#login", None)]
    step_executor_module.commit_pending_selector_cache()
    assert events[1][0] == "cache"
    assert events[1][1]["target"] == "login button"
    step_executor_module.discard_pending_selector_cache("test cleanup")


def test_step_executor_discards_deferred_selector_cache_when_test_fails(monkeypatch):
    events: list[Any] = []

    class FakeResolver:
        def resolve(self, **kwargs):
            return ResolvedSelector(
                selector="#login",
                source="heuristic",
                confidence=0.9,
                cache_action="click",
                cache_target=kwargs["target"],
                cache_page_key="https://example.test/login",
            )

        def record_verified_selector(self, **kwargs):
            events.append(("cache", kwargs))

    class FakePage:
        url = "https://example.test/login"

    class FakeUiHelper:
        pass

    monkeypatch.setenv("UI_SELECTOR_CACHE_COMMIT_MODE", "deferred")
    monkeypatch.setattr(
        "src.step_actions.step_executor.execute_action_with_command",
        lambda ui_helper, action, selector, value, step: None,
    )

    step_executor_module.discard_pending_selector_cache("test setup")
    executor = StepExecutor(FakePage(), FakeUiHelper(), elements={})
    executor.smart_resolver = FakeResolver()

    executor.execute_step(
        {"action": "click", "target": "login button", "mode": "smart"}
    )
    step_executor_module.discard_pending_selector_cache("failed assertion")
    step_executor_module.commit_pending_selector_cache()

    assert events == []


def test_step_executor_skips_selector_cache_when_action_switches_page(monkeypatch):
    events: list[Any] = []

    class FakePage:
        def __init__(self, url):
            self.url = url

    first_page = FakePage("https://example.test/login")
    second_page = FakePage("https://example.test/download")

    class FakeResolver:
        page = first_page

        def resolve(self, **kwargs):
            events.append(("resolve", kwargs["target"]))
            return ResolvedSelector(
                selector="#download",
                source="heuristic",
                confidence=0.9,
                cache_action="click",
                cache_target=kwargs["target"],
                cache_page_key="https://example.test/login",
            )

        def record_verified_selector(self, **kwargs):
            events.append(("cache", kwargs))

    class FakeUiHelper:
        def __init__(self):
            self.page = first_page

    def fake_execute_action_with_command(ui_helper, action, selector, value, step):
        events.append(("execute", action, selector, value))
        ui_helper.page = second_page

    monkeypatch.setattr(
        "src.step_actions.step_executor.execute_action_with_command",
        fake_execute_action_with_command,
    )

    ui_helper = FakeUiHelper()
    resolver = FakeResolver()
    executor = StepExecutor(first_page, ui_helper, elements={})
    executor.smart_resolver = resolver

    executor.execute_step(
        {"action": "click", "target": "download app", "mode": "smart"}
    )

    assert [event[0] for event in events] == ["resolve", "execute"]
    assert executor.page is second_page
    assert resolver.page is second_page


def test_step_executor_skips_selector_cache_when_action_triggers_page_error(
    monkeypatch,
):
    events: list[Any] = []

    class FakeResolver:
        def resolve(self, **kwargs):
            return ResolvedSelector(
                selector="#login",
                source="heuristic",
                confidence=0.9,
                cache_action="click",
                cache_target=kwargs["target"],
                cache_page_key="https://example.test/login",
            )

        def record_verified_selector(self, **kwargs):
            events.append(("cache", kwargs))

    class FakePage:
        url = "https://example.test/login"

    class FakeUiHelper:
        page_errors: list[str] = []

    def fake_execute_action_with_command(ui_helper, action, selector, value, step):
        ui_helper.page_errors.append("login is not defined")

    monkeypatch.setattr(
        "src.step_actions.step_executor.execute_action_with_command",
        fake_execute_action_with_command,
    )

    executor = StepExecutor(FakePage(), FakeUiHelper(), elements={})
    executor.smart_resolver = FakeResolver()

    executor.execute_step({"action": "click", "target": "login button", "mode": "smart"})

    assert events == []


def test_step_executor_records_page_error_without_masking_business_result(monkeypatch):
    class FakePage:
        url = "https://example.test/login"

    class FakeUiHelper:
        page_errors: list[str] = []

        def wait_for_stable(self, *, timeout: int, idle_ms: int):
            return None

    def fake_execute_action_with_command(ui_helper, action, selector, value, step):
        ui_helper.page_errors.append("login is not defined")

    monkeypatch.setattr(
        "src.step_actions.step_executor.execute_action_with_command",
        fake_execute_action_with_command,
    )

    executor = StepExecutor(FakePage(), FakeUiHelper(), elements={})
    step = {"action": "click", "selector": "#login"}

    executor.execute_step(step)

    assert step["_action_page_errors"] == ["login is not defined"]


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
    executor = StepExecutor(
        page=None, ui_helper=None, elements={}, default_mode="smart"
    )

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
                    "final_reason": "done",
                },
            )()

    monkeypatch.setattr("src.test_case_executor.AgentCaseExecutor", FakeAgentRunner)

    CaseExecutor(
        {
            "description": "agent case",
            "type": "agent_case",
            "intent": "complete cart flow",
            "final": ["order complete"],
        },
        elements={"title": ".title"},
        case_metadata={"name": "test_agent_case"},
    ).execute_test_case(page="page", ui_helper="ui")

    assert calls[0]["elements"] == {"title": ".title"}
    assert calls[1]["case_name"] == "test_agent_case"
    assert calls[1]["case_data"]["intent"] == "complete cart flow"


def test_payload_runner_reuses_fill_selector_for_value_assertion():
    executed: list[dict] = []

    class FakeStepExecutor:
        page = type("Page", (), {"url": "https://example.test/form"})()
        elements: dict[str, Any] = {}
        modules_cache: dict[str, Any] = {}

        def execute_step(self, step):
            if step.get("action") == "fill":
                step["_resolved_selector"] = "#product-id"
            executed.append(dict(step))

    result = execute_compiled_payload_steps(
        step_executor=FakeStepExecutor(),
        payload={},
        case_name="test_reuse_selector",
        steps=[
            {
                "action": "fill",
                "target": "商品ID输入框",
                "value": "SPU17799383833255667",
            },
            {
                "action": "assert_value",
                "target": "商品ID输入框",
                "value": "SPU17799383833255667",
            },
        ],
    )

    assert executed[1]["selector"] == "#product-id"
    assert result.history[1]["step"]["selector"] == "#product-id"


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
            "steps": ["open login page", "login standard user", "open cart"],
            "final": ["cart page visible"],
        },
    )

    assert spec["input_type"] == "steps"
    assert spec["steps"] == ["open login page", "login standard user", "open cart"]
    assert (
        spec["intent"]
        == "按顺序完成自然语言步骤：open login page；login standard user；open cart"
    )


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
            "selector": 'a[data-test="shopping-cart-link"]',
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
    assert any(
        item["selector"] == 'a[data-test="shopping-cart-link"]' for item in compacted
    )
    for item in compacted:
        assert "bbox" not in item
        assert "bbox_norm" not in item
        assert "center" not in item
        assert "center_norm" not in item
        assert "class_name" not in item
    cart = next(
        item
        for item in compacted
        if item["selector"] == 'a[data-test="shopping-cart-link"]'
    )
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
    assert (
        button["selector_candidates"][0]
        == 'button[data-test="add-to-cart-sauce-labs-backpack"]'
    )
    assert (
        selector_for_element_id(context, "e12")
        == 'button[data-test="add-to-cart-sauce-labs-backpack"]'
    )
    assert context["business_objects"]["cards"][0]["name"].startswith(
        "Sauce Labs Backpack"
    )
    assert context["compression"]["kept_element_count"] == 2


def test_model_dom_context_prioritizes_current_goal_over_dom_order():
    candidates = [
        {
            "index": 1,
            "tag": "div",
            "role": "menuitem",
            "text": "闂ㄥ簵绠＄悊",
            "selector": "li:nth-of-type(1) > div",
            "visible": True,
            "enabled": True,
        },
        {
            "index": 2,
            "tag": "div",
            "role": "menuitem",
            "text": "绉熸埛绠＄悊",
            "selector": "li:nth-of-type(2) > div",
            "visible": True,
            "enabled": True,
        },
        {
            "index": 3,
            "tag": "input",
            "selector": "#product-audit-search_brandid",
            "id": "product-audit-search_brandid",
            "label": "鍝佺墝",
            "type": "search",
            "visible": True,
            "enabled": True,
        },
        {
            "index": 4,
            "tag": "input",
            "selector": "#product-audit-search_seriesid",
            "id": "product-audit-search_seriesid",
            "label": "杞︾郴",
            "type": "search",
            "visible": True,
            "enabled": True,
        },
        {
            "index": 5,
            "tag": "input",
            "selector": "#product-audit-search_spuCode",
            "id": "product-audit-search_spuCode",
            "label": "鍟嗗搧ID",
            "type": "text",
            "visible": True,
            "enabled": True,
        },
        {
            "index": 6,
            "tag": "button",
            "selector": 'button:has-text("Search")',
            "text": "Search",
            "visible": True,
            "enabled": True,
        },
        {
            "index": 7,
            "tag": "a",
            "selector": 'a:has-text("Log")',
            "text": "Log",
            "visible": True,
            "enabled": True,
        },
    ]
    hints = ["杈撳叆鍟嗗搧ID SPU17799536972067519锛岀偣鍑绘煡璇㈡寜閽紝鐒跺悗鐐瑰嚮鏌ョ湅鏃ュ織"]
    context = build_dom_context(
        candidates,
        url="https://example.test/productAuditList",
        title="鍟嗗搧瀹℃牳",
        limit=7,
        hints=hints,
    )

    compacted = compact_model_dom_context(
        context,
        candidate_limit=3,
        form_limit=2,
        assertion_limit=2,
        hints=hints,
    )

    assert compacted["forms"][0]["label"] == "鍟嗗搧ID"
    visible_actions = {
        item.get("text") or item.get("name")
        for item in compacted["interactive_elements"]
    }
    assert "Search" in visible_actions
    assert "Log" in visible_actions


def test_selector_validation_reports_non_unique_matches():
    class FakeLocator:
        @property
        def first(self):
            return self

        def count(self):
            return 2

        def nth(self, index):
            return self

        def is_visible(self):
            return True

        def wait_for(self, state, timeout):
            return None

        def is_enabled(self):
            return True

    class FakePage:
        def locator(self, selector):
            assert selector == "button"
            return FakeLocator()

    validation = validate_selector(
        FakePage(),
        "button",
        action="click",
        timeout=1000,
        require_unique=True,
    )

    assert validation.ok is False
    assert validation.match_count == 2
    assert validation.visible_count == 2
    assert "matched 2 elements" in (validation.error or "")


def test_collect_candidates_passes_ignore_and_shadow_options():
    captured: dict[str, Any] = {}

    class FakePage:
        def evaluate(self, script, options):
            captured["script"] = script
            captured["options"] = options
            return []

    candidates = collect_candidates(
        FakePage(),
        limit=17,
        ignore_selectors=[".ads", "footer"],
        include_open_shadow_dom=True,
    )

    assert candidates == []
    assert captured["options"] == {
        "limit": 17,
        "ignore_selectors": [".ads", "footer"],
        "include_open_shadow_dom": True,
    }
    assert "shadowRoot" in captured["script"]
    assert "ignoreSelectors" in captured["script"]


def test_agent_case_model_payload_uses_compact_dom_context(monkeypatch, tmp_path: Path):
    captured: dict[str, Any] = {}

    class FakeStepExecutor:
        def __init__(self, page, ui_helper, elements, default_mode=None):
            pass

        def execute_step(self, step):
            raise AssertionError("done decision should not execute a step")

    class FakeProvider:
        settings = type("Settings", (), {"model": "test-model"})()

        def complete(self, messages, **kwargs):
            captured["messages"] = messages
            captured["usage_metadata"] = kwargs.get("usage_metadata")
            captured["stable_payload"] = json.loads(messages[1]["content"])
            captured["payload"] = json.loads(messages[2]["content"])
            return '{"action":"done","reason":"done"}'

        def complete_model(self, messages, response_model, **kwargs):
            captured["messages"] = messages
            captured["usage_metadata"] = kwargs.get("usage_metadata")
            captured["stable_payload"] = json.loads(messages[1]["content"])
            captured["payload"] = json.loads(messages[2]["content"])
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
        lambda page, **kwargs: [
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
                "selector": 'a[data-test="shopping-cart-link"]',
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
                "agent_case_plan_cache_enabled": False,
                "agent_candidate_scan_limit": 20,
                "agent_candidate_limit": 2,
                "agent_model_candidate_limit": 2,
                "agent_model_form_limit": 2,
                "agent_model_assertion_limit": 2,
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
        "src.ai_runtime.agent_case_executor.compile_case_payload",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("this test exercises realtime prompt payload")
        ),
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor._unmet_final_criteria",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor._has_completion_criteria",
        lambda criteria: False,
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor._unmet_completion_criteria",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor._unmet_intent_action_requirements",
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
            "intent": "open cart and verify",
            "final": ["cart page visible"],
        },
    )

    payload = captured["payload"]
    stable_payload = captured["stable_payload"]
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][1]["role"] == "user"
    assert captured["messages"][2]["role"] == "user"
    assert stable_payload["prompt_section"] == "stable_context"
    assert payload["prompt_section"] == "realtime_state"
    assert "dom_context" not in stable_payload
    assert "history" not in stable_payload
    assert "dom_candidates" not in payload
    assert "plan_cursor" not in payload
    assert "runtime_harness" in payload
    assert payload["current_step"]
    assert isinstance(payload["completed_steps"], list)
    assert isinstance(payload["remaining_steps"], list)
    assert isinstance(payload["execution_evidence"], list)
    assert payload["runtime_harness"]["phase"]
    assert payload["runtime_harness"]["phase_categories"]
    assert "phase_observation" in payload["runtime_harness"]
    assert "pending_slots" not in payload["runtime_harness"]
    assert "allowed_actions" not in payload["runtime_harness"]
    assert "recommended_actions" not in payload["runtime_harness"]
    assert "action_categories" in stable_payload
    assert "allowed_actions" not in stable_payload
    assert "unmet_requirements" in payload
    assert "history" not in payload
    assert "agent_state" not in payload
    assert "plan_status" not in payload["runtime_harness"]
    assert "rule" not in payload["runtime_harness"]
    assert payload["dom_context"]["interactive_elements"][0]["id"] == "e0"
    assert payload["dom_context"]["interactive_elements"][0]["selector_candidates"]
    assert (
        len(payload["dom_context"]["interactive_elements"][0]["selector_candidates"])
        == 1
    )
    assert stable_payload["project_context"]["element_keys"] == []
    assert stable_payload["project_context"]["module_names"] == []
    assert stable_payload["project_context"]["modules"] == {}
    assert "use_module" not in stable_payload["action_categories"]["navigation"]
    assert "use_module" not in stable_payload["action_contract"]
    assert captured["usage_metadata"]["stable_chars"] > 0
    assert captured["usage_metadata"]["realtime_chars"] > 0
    assert captured["usage_metadata"]["dom_items"] >= 1
    for item in payload["dom_context"]["interactive_elements"]:
        assert "class_name" not in item
        assert "bbox" not in item
        assert "bbox_norm" not in item
        assert "center" not in item


def test_agent_case_realtime_unmet_requirements_only_report_current_phase():
    spec = {
        "description": "",
        "intent": (
            "visit url https://example.test/login, use OA login, "
            "expand mall menu, click product audit, fill product id SPU1"
        ),
        "steps": [],
        "inputs": {"username": "${username}", "password": "${password}"},
        "criteria": {
            "checkpoints": [
                "page title contains product audit",
                "product id input filled SPU1",
            ],
            "final": ["still on product audit page"],
        },
    }
    history = [
        {
            "step": {"action": "goto", "value": "https://example.test/login"},
            "result": "passed",
            "url_after": "https://example.test/login",
        },
        {
            "step": {"action": "click", "selector": 'button:has-text("使用OA登录 >>")'},
            "result": "passed",
            "url_after": "https://sso.example.test/login",
        },
    ]
    dom_context = {
        "meta": {"url": "https://sso.example.test/login", "route_hint": "login"},
        "forms": [
            {"label": "OA account", "value_state": "empty"},
            {"label": "OA password", "value_state": "empty"},
        ],
        "interactive_elements": [],
        "assertion_candidates": [],
    }

    unmet = _runtime_unmet_requirements(
        spec=spec,
        history=history,
        current_url="https://sso.example.test/login",
        dom_context=dom_context,
        phase="use OA login",
    )

    assert unmet == ["use OA login"]
    assert all("product" not in item.lower() for item in unmet)
    assert all("mall" not in item.lower() for item in unmet)


def test_agent_case_does_not_expose_business_fast_path():
    assert not hasattr(AgentCaseExecutor, "_fast_path_decision")
    assert not hasattr(AgentCaseExecutor, "_local_phase_decision")


def test_agent_case_step_hints_keep_full_short_plan():
    steps = [f"step {index}" for index in range(1, 12)]

    assert _remaining_step_hints(spec={"steps": steps}, history=[]) == steps


def test_agent_case_step_hints_remove_completed_steps():
    steps = ["click search", "click view log"]
    history = [
        {
            "step": {
                "action": "click",
                "selector": 'button:has-text("Search")',
                "_action_target_text": "Search",
                "_action_dom_changed": False,
            },
            "result": "passed",
            "url_after": "https://example.test/list",
        }
    ]

    assert _remaining_step_hints(spec={"steps": steps}, history=history) == [
        "click view log"
    ]


def test_agent_case_step_hints_treat_matching_click_execution_as_done():
    steps = ["点击查询按钮", "点击查看日志"]
    history = [
        {
            "step": {
                "action": "click",
                "selector": 'button:has-text("查询")',
                "_action_executed_selector": 'button:has-text("查询")',
                "_action_target_text": "查询",
                "_action_dom_changed": False,
            },
            "result": "passed",
            "url_after": "https://example.test/list",
        }
    ]

    assert _remaining_step_hints(spec={"steps": steps}, history=history) == [
        "点击查看日志"
    ]


def test_agent_case_step_hints_treat_xpath_query_click_as_done():
    steps = ["点击查询按钮", "点击查看日志"]
    history = [
        {
            "step": {
                "action": "click",
                "selector": "//button[contains(text(),'查询')]",
                "_action_executed_selector": "//button[contains(text(),'查询')]",
                "_action_dom_changed": False,
            },
            "result": "passed",
            "url_after": "https://example.test/list",
        }
    ]

    assert _text_query_from_selector("//button[contains(text(),'查询')]") == "查询"
    assert _remaining_step_hints(spec={"steps": steps}, history=history) == [
        "点击查看日志"
    ]


def test_agent_case_step_hints_do_not_advance_for_unrelated_download_click():
    steps = ["点击登录按钮", "点击商城展开菜单"]
    history = [
        {
            "step": {
                "action": "click",
                "selector": 'a:has-text("汽车人APP下载")',
                "_action_executed_selector": 'a:has-text("汽车人APP下载")',
                "_action_target_text": "汽车人APP下载",
                "_action_dom_changed": False,
            },
            "result": "passed",
            "url_after": "https://sso.example.test/login",
        }
    ]

    assert _remaining_step_hints(spec={"steps": steps}, history=history) == steps


def test_agent_case_step_hints_split_intent_plan():
    hints = _remaining_step_hints(
        spec={"intent": "使用项目模块 login，输入搜索词 hello，点击搜索按钮，然后点击结果详情"},
        history=[],
    )

    assert hints == [
        "使用项目模块 login",
        "输入搜索词 hello",
        "点击搜索按钮",
        "点击结果详情",
    ]


def test_agent_case_plan_cache_uses_plan_namespace(tmp_path: Path):
    cache = AgentCasePlanCache(tmp_path / "ai_cache.sqlite3")
    payload = {
        "cases": [{"name": "test_agent"}],
        "data": {
            "test_agent": {
                "mode": "smart",
                "steps": [{"action": "assert_title_contains", "value": "Done"}],
            }
        },
        "elements": {},
        "modules": {},
        "vars": {},
    }

    cache.save_plan(
        key="plan-key",
        project="demo",
        env="prod",
        case_name="test_agent",
        entry_scope="https://example.test",
        spec={"intent": "check done", "steps": [], "inputs": {}, "criteria": {}},
        payload=payload,
        case_payload_name="test_agent",
        steps=payload["data"]["test_agent"]["steps"],
        prompt_version="agent-case-test",
        schema_version="schema-test",
        model="model-test",
        asset_hash="asset",
    )

    assert cache.load_plan("plan-key")["steps"] == [
        {"action": "assert_title_contains", "value": "Done"}
    ]


def test_agent_case_has_no_trace_replay_cache_api(tmp_path: Path):
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
    assert not hasattr(executor, "_try_replay_cache")


def test_agent_case_completion_waits_only_after_navigation_like_actions():
    assert (
        _completion_wait_seconds_for_step(
            {"action": "click", "selector": "#submit"}, default_seconds=12
        )
        == 12
    )
    assert (
        _completion_wait_seconds_for_step(
            {"action": "fill", "selector": "#search", "value": "bmw x3"},
            default_seconds=12,
        )
        == 0
    )


def test_agent_case_observable_completion_terms_extract_title_and_url_terms():
    title_terms, url_terms = _observable_completion_terms(
        {
            "checkpoints": [
                'after search succeeds, page title contains "bmw x3"',
                'current URL contains "q=bmw+x3"',
            ],
            "final": [],
        }
    )

    assert title_terms == ["bmw x3"]
    assert url_terms == ["q=bmw+x3"]


def test_agent_case_finish_detects_unmet_logout_final_criteria():
    unmet = _unmet_final_criteria(
        criteria={"final": ["閫€鍑虹櫥褰曞悗鍥炲埌鐧诲綍椤碉紝鐧诲綍鎸夐挳鍙"]},
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

    assert unmet == ["閫€鍑虹櫥褰曞悗鍥炲埌鐧诲綍椤碉紝鐧诲綍鎸夐挳鍙"]


def test_agent_case_completion_uses_checkpoints_when_final_is_absent():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": ['after search succeeds, page title contains "bmw x3"'],
            "final": [],
        },
        history=[],
        current_url="https://sou.example.test/search?q=bmw+x3",
        dom_context={
            "meta": {
                "title": "[bmw x3] latest info",
                "url": "https://sou.example.test/search?q=bmw+x3",
            },
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    )

    assert unmet == []


def test_agent_case_completion_extracts_business_page_term():
    unmet = _unmet_completion_criteria(
        criteria={"final": ["current page remains on product audit page"]},
        history=[],
        current_url="https://example.test/productAuditList",
        dom_context={
            "meta": {"title": "Product Audit - Admin"},
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    )

    assert unmet == []


def test_agent_case_completion_requires_checkpoints_with_final():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": ["search button executed"],
            "final": ["current page remains on product audit page"],
        },
        history=[],
        current_url="https://example.test/productAuditList",
        dom_context={
            "meta": {"title": "Product Audit - Admin"},
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    )

    assert unmet == ["search button executed"]


def test_agent_case_completion_uses_module_runtime_steps_as_evidence():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": ["查询按钮已执行"],
            "final": ["当前页面仍在商品审核业务页"],
        },
        history=[
            {
                "step": {
                    "use_module": "crm_search_and_view_log",
                    "_module_executed_steps": [
                        {
                            "action": "fill",
                            "selector": "crm_product_id_input",
                            "value": "SPU17799383833255667",
                        },
                        {
                            "action": "click",
                            "selector": "crm_search_btn",
                        },
                        {
                            "action": "click",
                            "selector": "crm_view_log_btn",
                        },
                    ],
                },
                "result": "passed",
            },
        ],
        current_url="https://example.test/productAuditList",
        dom_context={"meta": {"title": "商品审核 - 新零售运营后台"}},
    )

    assert unmet == []


def test_agent_case_plan_cache_does_not_store_module_runtime_steps():
    cached_steps = _cacheable_plan_steps(
        [
            {
                "use_module": "crm_search_and_view_log",
                "_module_executed_steps": [
                    {"action": "click", "selector": "crm_search_btn"}
                ],
            }
        ]
    )

    assert cached_steps == [{"use_module": "crm_search_and_view_log"}]


def test_agent_case_history_keeps_module_runtime_steps(tmp_path: Path):
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

    item = executor._history_item(
        step={
            "use_module": "crm_search_and_view_log",
            "_module_executed_steps": [
                {
                    "action": "click",
                    "selector": "crm_search_btn",
                    "_action_executed_selector": 'button:has-text("查 询")',
                }
            ],
        },
        source="compiled_agent_case",
    )

    assert item["step"]["_module_executed_steps"] == [
        {
            "action": "click",
            "selector": "crm_search_btn",
            "_action_executed_selector": 'button:has-text("查 询")',
        }
    ]


def test_agent_case_completion_uses_action_history_for_input_and_click():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": [
                "page title contains Product Audit",
                "product id field filled SPU17799383833255667",
                "search button executed",
            ],
            "final": ["current page remains on product audit page"],
        },
        history=[
            {
                "step": {
                    "action": "fill",
                    "selector": "#product-audit-search_spuCode",
                    "value": "SPU17799383833255667",
                },
                "result": "passed",
            },
            {
                "step": {
                    "action": "click",
                    "selector": 'button:has-text("Search")',
                },
                "result": "passed",
            },
        ],
        current_url="https://example.test/productAuditList",
        dom_context={
            "meta": {"title": "Product Audit - Admin"},
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    )

    assert unmet == []


def test_agent_case_completion_accepts_resolved_fill_value_for_cjk_input():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": ["商品ID输入框已输入 SPU17799383833255667"],
            "final": [],
        },
        history=[
            {
                "step": {
                    "action": "fill",
                    "selector": "crm_product_id_input",
                    "value": "${product_id}",
                    "_resolved_selector": "#product-audit-search_spuCode",
                    "_resolved_value": "SPU17799383833255667",
                },
                "result": "passed",
            },
        ],
        current_url="https://example.test/productAuditList",
        dom_context={"meta": {"title": "Product Audit"}},
    )

    assert unmet == []


def test_agent_case_completion_accepts_value_assertion_as_input_evidence():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": ["商品ID输入框已输入 SPU17799383833255667"],
            "final": [],
        },
        history=[
            {
                "step": {
                    "action": "assert_value",
                    "selector": "crm_product_id_input",
                    "value": "SPU17799383833255667",
                    "_resolved_selector": "#product-audit-search_spuCode",
                },
                "result": "passed",
            },
        ],
        current_url="https://example.test/productAuditList",
        dom_context={"meta": {"title": "Product Audit"}},
    )

    assert unmet == []


def test_agent_case_intent_requirements_prevent_early_finish_before_click_log():
    spec = {
        "intent": "fill product id SPU17799383833255667, click search, then click view log",
        "steps": [],
    }
    history = [
        {
            "step": {
                "action": "fill",
                "selector": "#product-audit-search_spuCode",
                "value": "SPU17799383833255667",
            },
            "result": "passed",
        },
        {
            "step": {
                "action": "click",
                "selector": 'button:has-text("Search")',
                "_action_before_url": "https://example.test/productAuditList",
                "_action_after_url": "https://example.test/productAuditList",
                "_action_dom_changed": True,
            },
            "result": "passed",
        },
    ]

    assert _unmet_intent_action_requirements(spec=spec, history=history) == [
        "intent click: view log/log"
    ]


def test_agent_case_intent_requirements_pass_after_click_log():
    spec = {
        "intent": "fill product id SPU17799383833255667, click search, then click view log",
        "steps": [],
    }
    history = [
        {
            "step": {
                "action": "fill",
                "selector": "#product-audit-search_spuCode",
                "value": "SPU17799383833255667",
            },
            "result": "passed",
        },
        {
            "step": {
                "action": "click",
                "selector": 'button:has-text("Search")',
                "_action_before_url": "https://example.test/productAuditList",
                "_action_after_url": "https://example.test/productAuditList",
                "_action_dom_changed": True,
            },
            "result": "passed",
        },
        {
            "step": {
                "action": "click",
                "selector": 'a:has-text("View Log")',
                "_action_before_url": "https://example.test/productAuditList",
                "_action_after_url": "https://example.test/productAuditLog",
            },
            "result": "passed",
        },
    ]

    assert _unmet_intent_action_requirements(spec=spec, history=history) == []


def test_agent_case_intent_requirements_ignore_navigation_url_as_click():
    spec = {
        "intent": "visit url https://example.test/login, use OA login, click search",
        "steps": [],
    }
    history = [
        {
            "step": {"action": "goto", "value": "https://example.test/login"},
            "result": "passed",
            "url_after": "https://example.test/login",
        }
    ]

    assert _unmet_intent_action_requirements(spec=spec, history=history) == [
        "intent click: oa login/login",
        "intent click: search",
    ]


def test_agent_case_intent_requirements_treat_use_login_as_interaction():
    spec = {
        "intent": "visit url https://example.test/login, use OA login, click search",
        "steps": [],
    }
    history = [
        {
            "step": {"action": "goto", "value": "https://example.test/login"},
            "result": "passed",
            "url_after": "https://example.test/login",
        },
        {
            "step": {"action": "click", "selector": 'button:has-text("使用OA登录 >>")'},
            "result": "passed",
            "url_after": "https://sso.example.test/login",
        },
    ]

    assert _unmet_intent_action_requirements(
        spec=spec,
        history=history,
        current_url="https://sso.example.test/login",
        dom_context={
            "meta": {"url": "https://sso.example.test/login", "route_hint": "login"},
            "forms": [{"label": "Password", "value_state": "empty"}],
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    ) == ["intent click: search"]


def test_agent_case_intent_requirements_complete_use_login_after_auth_transition():
    spec = {
        "intent": "visit url https://example.test/login, use OA login, click search",
        "steps": [],
    }
    history = [
        {
            "step": {"action": "goto", "value": "https://example.test/login"},
            "result": "passed",
            "url_after": "https://example.test/login",
        },
        {
            "step": {"action": "click", "selector": 'button:has-text("使用OA登录 >>")'},
            "result": "passed",
            "url_after": "https://sso.example.test/login",
        },
        {
            "step": {"action": "fill", "selector": "#username", "value": "${username}"},
            "result": "passed",
            "url_after": "https://sso.example.test/login",
        },
        {
            "step": {"action": "fill", "selector": "#password", "value": "${password}"},
            "result": "passed",
            "url_after": "https://sso.example.test/login",
        },
        {
            "step": {"action": "click", "selector": 'input[type="submit"]'},
            "result": "passed",
            "url_after": "https://example.test/home",
        },
    ]

    assert _unmet_intent_action_requirements(
        spec=spec,
        history=history,
        current_url="https://example.test/home",
        dom_context={
            "meta": {"url": "https://example.test/home", "route_hint": ""},
            "page_summary": {"visible_text_summary": ["Search"]},
            "forms": [],
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    ) == ["intent click: search"]


def test_agent_case_intent_requirements_ignore_input_parameter_reference():
    spec = {
        "intent": "",
        "steps": [
            "use project module login_flow with input username and password",
            "fill product id SPU17799383833255667",
            "click search button",
            "click view log",
        ],
    }
    history = [
        {
            "step": {
                "use_module": "login_flow",
                "params": {"username": "${username}", "password": "${password}"},
            },
            "result": "passed",
        },
        {
            "step": {
                "action": "fill",
                "selector": "#product-audit-search_spuCode",
                "value": "SPU17799383833255667",
            },
            "result": "passed",
        },
        {
            "step": {
                "action": "click",
                "selector": 'button:has-text("Search")',
                "_action_before_url": "https://example.test/productAuditList",
                "_action_after_url": "https://example.test/productAuditList",
                "_action_dom_changed": True,
            },
            "result": "passed",
        },
        {
            "step": {
                "action": "click",
                "selector": 'a:has-text("View Log")',
                "_action_before_url": "https://example.test/productAuditList",
                "_action_after_url": "https://example.test/productAuditLog",
            },
            "result": "passed",
        },
    ]

    assert _unmet_intent_action_requirements(spec=spec, history=history) == []


def test_agent_case_reuses_observed_selector_when_text_spacing_differs():
    selector = _selector_for_equivalent_dom_text(
        current_selector="button:has-text('Search')",
        dom_context={
            "forms": [],
            "interactive_elements": [
                {
                    "id": "e1",
                    "tag": "button",
                    "role": "button",
                    "text": "Sea rch",
                    "name": "Sea rch",
                    "selector_candidates": ['button:has-text("Sea rch")'],
                }
            ],
            "assertion_candidates": [],
        },
    )

    assert selector == 'button:has-text("Sea rch")'


def test_agent_case_title_checkpoint_ignores_non_assertion_input_history():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": ['after search succeeds, page title contains "bmw x3"'],
            "final": [],
        },
        history=[
            {
                "step": {"action": "fill", "selector": "#search", "value": "bmw x3"},
                "result": "passed",
            }
        ],
        current_url="https://www.example.test/",
        dom_context={
            "meta": {"title": "Home"},
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    )

    assert unmet == ['after search succeeds, page title contains "bmw x3"']


def test_agent_case_decision_accepts_standard_step_mode():
    decision = AgentCaseDecision.model_validate(
        {"action": "click", "mode": "smart", "selector": "#continue"}
    )

    assert decision.mode == "smart"
    assert decision.selector == "#continue"


def test_agent_case_parser_discards_extra_reasoning_fields():
    decision = _parse_agent_decision_response(
        json.dumps(
            {
                "thought": "do not expose this",
                "action": "click",
                "selector": "#continue",
                "reason": "continue",
            }
        ),
        spec={"inputs": {}},
        dom_context={},
    )

    assert decision.action == "click"
    assert decision.selector == "#continue"


def test_agent_case_parser_normalizes_decision_alias_to_action():
    decision = _parse_agent_decision_response(
        json.dumps(
            {
                "decision": "click",
                "selector": "#finish",
                "reason": "finish current workflow",
            }
        ),
        spec={"inputs": {}},
        dom_context={},
    )

    assert decision.action == "click"
    assert decision.selector == "#finish"


def test_agent_case_parser_normalizes_action_wrapped_object():
    decision = _parse_agent_decision_response(
        json.dumps(
            {
                "fill": {
                    "selector": "#product-audit-search_spuCode",
                    "value": "SPU17799383833255667",
                }
            },
            ensure_ascii=False,
        ),
        spec={"inputs": {}},
        dom_context={},
    )

    assert decision.action == "fill"
    assert decision.selector == "#product-audit-search_spuCode"
    assert decision.value == "SPU17799383833255667"


def test_agent_case_parser_normalizes_selectors_list_to_selector():
    decision = _parse_agent_decision_response(
        json.dumps(
            {
                "action": "fill",
                "selectors": ["", "#ipt_password"],
                "value": "${password}",
            }
        ),
        spec={"inputs": {}},
        dom_context={},
    )

    assert decision.action == "fill"
    assert decision.selector == "#ipt_password"
    assert decision.value == "${password}"


def test_agent_case_decision_accepts_assert_title_contains():
    decision = AgentCaseDecision.model_validate(
        {"action": "assert_title_contains", "value": "鍟嗗搧瀹℃牳"}
    )

    assert decision.action == "assert_title_contains"
    assert decision.value == "鍟嗗搧瀹℃牳"


def test_agent_case_parser_normalizes_assert_text_text_alias():
    decision = _parse_agent_decision_response(
        json.dumps(
            {
                "action": "assert_text",
                "selector": "#result-title",
                "text": "瀹濋┈x3",
            },
            ensure_ascii=False,
        ),
        spec={"inputs": {}},
        dom_context={},
    )

    assert decision.action == "assert_text"
    assert decision.value == "瀹濋┈x3"


def test_agent_case_parser_normalizes_assert_text_target_as_value():
    decision = _parse_agent_decision_response(
        json.dumps(
            {
                "action": "assert_text",
                "target": "Welcome",
            },
            ensure_ascii=False,
        ),
        spec={"inputs": {}},
        dom_context={},
    )

    assert decision.action == "assert_text"
    assert decision.target == "Welcome"
    assert decision.value == "Welcome"


def test_agent_case_parser_uses_assert_text_value_as_visible_text_target():
    decision = _parse_agent_decision_response(
        json.dumps(
            {
                "action": "assert_text",
                "value": "Dashboard",
            },
            ensure_ascii=False,
        ),
        spec={"inputs": {}},
        dom_context={},
    )

    assert decision.action == "assert_text"
    assert decision.target == "Dashboard"
    assert decision.value == "Dashboard"


def test_agent_case_parser_defaults_wait_ms():
    decision = _parse_agent_decision_response(
        json.dumps({"action": "wait", "reason": "wait for page update"}),
        spec={"inputs": {}},
        dom_context={},
    )

    assert decision.action == "wait"
    assert decision.wait_ms == 1000


def test_agent_case_parser_normalizes_wait_value_to_wait_ms():
    decision = _parse_agent_decision_response(
        json.dumps({"action": "wait", "value": 2000}),
        spec={"inputs": {}},
        dom_context={},
    )

    assert decision.action == "wait"
    assert decision.wait_ms == 2000
    assert decision.value is None


def test_agent_case_parser_normalizes_need_more_context_action():
    decision = _parse_agent_decision_response(
        json.dumps({"action": "need_more_context", "reason": "missing values"}),
        spec={"inputs": {}},
        dom_context={},
    )

    assert decision.status == "need_more_context"
    assert decision.action is None


def test_agent_case_parser_truncates_long_reason():
    decision = _parse_agent_decision_response(
        json.dumps({"action": "finish", "reason": "x" * 200}),
        spec={"inputs": {}},
        dom_context={},
    )

    assert decision.action == "finish"
    assert decision.reason is not None
    assert len(decision.reason) == 120


def test_agent_case_first_url_stops_at_natural_language_punctuation():
    assert (
        _first_url("打开 https://www.autohome.com.cn/beijing/?from=m；搜索框输入宝马x3")
        == "https://www.autohome.com.cn/beijing/?from=m"
    )
    assert _first_url("open https://example.test/path?q=1, then click") == (
        "https://example.test/path?q=1"
    )
    assert _first_url("visit https://example.test/path.") == (
        "https://example.test/path"
    )


def test_url_contains_accepts_percent_encoded_actual_url():
    assert _url_contains(
        "https://sou.autohome.com.cn/zonghe?q=%E5%AE%9D%E9%A9%ACx3",
        "q=\u5b9d\u9a6cx3",
    )
    assert not _url_contains(
        "https://sou.autohome.com.cn/zonghe?q=%E5%AE%9D%E9%A9%ACx3",
        "q=\u5965\u8fea",
    )


def test_agent_case_parser_infers_missing_fill_value_from_inputs():
    dom_context = build_dom_context(
        [
            {
                "index": 10,
                "tag": "input",
                "selector": "input[data-test='postalCode']",
                "data_test": "postalCode",
                "placeholder": "Zip/Postal Code",
                "visible": True,
                "enabled": True,
            }
        ],
        url="https://www.saucedemo.com/checkout-step-one.html",
        title="Swag Labs",
    )

    decision = _parse_agent_decision_response(
        json.dumps(
            {
                "action": "fill",
                "element_id": "e10",
                "reason": "fill postal code",
            }
        ),
        spec={"inputs": {"checkout_info": {"postal_code": "100000"}}},
        dom_context=dom_context,
    )

    assert decision.value == "100000"


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
                "reason": "fill username",
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
                "reason": "濉啓瀵嗙爜",
            }
        )
    )

    assert step["selector"] == "#password"
    assert "target" not in step


def test_agent_case_assert_text_uses_dom_text_selector_without_business_rule(
    tmp_path: Path,
):
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


def test_agent_case_assert_title_visible_heading_is_converted_to_text_assertion(
    tmp_path: Path,
):
    class FakePage:
        def title(self):
            return "Swag Labs"

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
        page=FakePage(),
        ui_helper=object(),
        elements={},
        context=context,
    )
    executor.current_dom_context = build_dom_context(
        [
            {
                "index": 5,
                "tag": "span",
                "selector": "[data-test='title']",
                "data_test": "title",
                "text": "Products",
                "visible": True,
                "enabled": True,
            }
        ],
        url="https://www.saucedemo.com/inventory.html",
        title="Swag Labs",
    )

    step = executor._decision_to_step(
        AgentCaseDecision.model_validate(
            {
                "action": "assert_title",
                "value": "Products",
                "reason": "confirm inventory heading",
            }
        )
    )

    assert step == {
        "action": "assert_text",
        "mode": "smart",
        "selector": 'span[data-test="title"]',
        "value": "Products",
    }


def test_agent_case_rejects_browser_title_as_visible_text_assertion(tmp_path: Path):
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
        [],
        url="https://example.test/search",
        title="Example",
    )

    with pytest.raises(AgentDecisionRejected, match="browser title is metadata"):
        executor._decision_to_step(
            AgentCaseDecision.model_validate(
                {
                    "action": "assert_text",
                    "selector": "title",
                    "value": "瀹濋┈x3",
                }
            )
        )


def test_agent_case_selector_conflict_prefers_explicit_selector(tmp_path: Path):
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
                "index": 20,
                "tag": "button",
                "selector": "button[data-test='remove-sauce-labs-backpack']",
                "data_test": "remove-sauce-labs-backpack",
                "text": "Remove",
                "visible": True,
                "enabled": True,
            },
            {
                "index": 21,
                "tag": "a",
                "selector": "#shopping_cart_container a",
                "text": "1",
                "visible": True,
                "enabled": True,
            },
        ],
        url="https://www.saucedemo.com/inventory.html",
        title="Swag Labs",
    )

    step = executor._decision_to_step(
        AgentCaseDecision.model_validate(
            {
                "action": "click",
                "element_id": "e20",
                "selector": "#shopping_cart_container a",
                "reason": "open cart",
            }
        )
    )

    assert step["selector"] == "#shopping_cart_container a"


def test_agent_case_repairs_disallowed_remove_to_semantic_progress(tmp_path: Path):
    class FakePage:
        url = "https://example.test/current"

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
        page=FakePage(),
        ui_helper=object(),
        elements={},
        context=context,
    )
    executor.current_dom_context = {
        "forms": [],
        "interactive_elements": [
            {
                "id": "e1",
                "text": "Remove",
                "selector_candidates": ["#danger-action"],
            },
            {
                "id": "e2",
                "text": "Review order",
                "selector_candidates": ["#primary-progress"],
            },
        ],
        "assertion_candidates": [],
    }

    step = executor._decision_to_step(
        AgentCaseDecision.model_validate(
            {"action": "click", "element_id": "e1", "reason": "continue checkout"}
        ),
        spec={
            "description": "",
            "intent": "review order and complete submission",
            "steps": [],
            "inputs": {},
            "criteria": {"final": ["order review is visible"]},
        },
    )

    assert step["selector"] == "#primary-progress"


def test_agent_case_repairs_disallowed_logout_to_close_open_menu(tmp_path: Path):
    class FakePage:
        url = "https://example.test/current"

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
        page=FakePage(),
        ui_helper=object(),
        elements={},
        context=context,
    )
    executor.current_dom_context = {
        "forms": [],
        "interactive_elements": [
            {
                "id": "e1",
                "text": "Logout",
                "selector_candidates": ["#session-exit"],
            },
            {
                "id": "e2",
                "text": "Close Menu",
                "selector_candidates": ["#close-panel"],
            },
        ],
        "assertion_candidates": [],
    }

    step = executor._decision_to_step(
        AgentCaseDecision.model_validate(
            {"action": "click", "element_id": "e1", "reason": "go to cart"}
        ),
        spec={
            "description": "",
            "intent": "continue current authenticated workflow",
            "steps": [],
            "inputs": {},
            "criteria": {"final": ["workflow complete"]},
        },
    )

    assert step["selector"] == "#close-panel"


def test_agent_case_allowed_logout_is_not_repaired_by_menu_reason(tmp_path: Path):
    class FakePage:
        url = "https://example.test/current"

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
        page=FakePage(),
        ui_helper=object(),
        elements={},
        context=context,
    )
    executor.current_dom_context = {
        "forms": [],
        "interactive_elements": [
            {
                "id": "e1",
                "text": "Open Menu",
                "near_text": "Close Menu Logout",
                "selector_candidates": ["#menu-trigger"],
            },
            {
                "id": "e2",
                "text": "Logout",
                "selector_candidates": ["#logout"],
            },
            {
                "id": "e3",
                "text": "Close Menu",
                "selector_candidates": ["#close-menu"],
            },
        ],
        "assertion_candidates": [],
    }

    step = executor._decision_to_step(
        AgentCaseDecision.model_validate(
            {
                "action": "click",
                "selector": "#logout",
                "reason": "鑿滃崟宸叉墦寮€锛岀偣鍑?Logout",
            }
        ),
        spec={
            "description": "",
            "intent": "log out from the current session",
            "steps": [],
            "inputs": {},
            "criteria": {"final": ["login page is visible after logout"]},
        },
    )

    assert step["selector"] == "#logout"


def test_agent_case_close_repair_ignores_near_text_on_other_elements(tmp_path: Path):
    class FakePage:
        url = "https://example.test/current"

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
        page=FakePage(),
        ui_helper=object(),
        elements={},
        context=context,
    )
    executor.current_dom_context = {
        "forms": [],
        "interactive_elements": [
            {
                "id": "e1",
                "text": "Open Menu",
                "near_text": "Close Menu Logout",
                "selector_candidates": ["#menu-trigger"],
            },
            {
                "id": "e2",
                "text": "Logout",
                "selector_candidates": ["#logout"],
            },
            {
                "id": "e3",
                "text": "Close Menu",
                "selector_candidates": ["#close-menu"],
            },
        ],
        "assertion_candidates": [],
    }

    step = executor._decision_to_step(
        AgentCaseDecision.model_validate(
            {"action": "click", "selector": "#logout", "reason": "go to cart"}
        ),
        spec={
            "description": "",
            "intent": "continue current authenticated workflow",
            "steps": [],
            "inputs": {},
            "criteria": {"final": ["workflow complete"]},
        },
    )

    assert step["selector"] == "#close-menu"


def test_agent_case_repairs_menu_button_to_semantic_goal_candidate(tmp_path: Path):
    class FakePage:
        url = "https://example.test/current"

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
        page=FakePage(),
        ui_helper=object(),
        elements={},
        context=context,
    )
    executor.current_dom_context = {
        "forms": [],
        "interactive_elements": [
            {
                "id": "e1",
                "text": "Open Menu",
                "selector_candidates": ["#menu-trigger"],
            },
            {
                "id": "e2",
                "name": "Order review",
                "selector_candidates": ["#target-destination"],
            },
        ],
        "assertion_candidates": [],
    }

    step = executor._decision_to_step(
        AgentCaseDecision.model_validate(
            {
                "action": "click",
                "selector": "#menu-trigger",
                "reason": "open order review",
            }
        ),
        spec={
            "description": "",
            "intent": "open order review and complete submission",
            "steps": [],
            "inputs": {},
            "criteria": {"final": ["order review is visible"]},
        },
    )

    assert step["selector"] == "#target-destination"


def test_agent_case_assert_text_repairs_selector_by_exact_text(tmp_path: Path):
    class FakeLocator:
        def __init__(self, text: str | None):
            self._text = text

        @property
        def first(self):
            return self

        def count(self):
            return 1 if self._text is not None else 0

        def inner_text(self, timeout=None):
            return self._text or ""

    class FakePage:
        def locator(self, selector):
            values = {
                ".title": "Products",
                'a[data-test="inventory-item-container"]': None,
                'div[data-test="header-container"]': "Open Menu Swag Labs Products",
            }
            return FakeLocator(values.get(selector))

    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://example.test/",
        elements={"sauce_page_title": ".title"},
        modules={},
        variables={},
        test_cases=[],
        test_data={},
    )
    executor = AgentCaseExecutor(
        page=FakePage(),
        ui_helper=object(),
        elements=context.elements,
        context=context,
    )
    executor.current_dom_context = build_dom_context(
        [
            {
                "index": 0,
                "tag": "div",
                "selector": 'div[data-test="header-container"]',
                "data_test": "header-container",
                "text": "Open Menu Swag Labs Products",
                "visible": True,
                "enabled": True,
            }
        ],
        url="https://www.saucedemo.com/inventory.html",
        title="Swag Labs",
    )

    step = executor._decision_to_step(
        AgentCaseDecision.model_validate(
            {
                "action": "assert_text",
                "selector": 'a[data-test="inventory-item-container"]',
                "value": "Products",
                "reason": "confirm inventory heading",
            }
        )
    )

    assert step["selector"] == ".title"


def test_agent_case_assert_text_uses_contains_for_contains_criteria(tmp_path: Path):
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
    executor.current_dom_context = {
        "forms": [],
        "interactive_elements": [],
        "assertion_candidates": [
            {"id": "a1", "selector_candidates": ["#route-title"], "text": "bmw x3"}
        ],
    }

    step = executor._decision_to_step(
        AgentCaseDecision.model_validate(
            {
                "action": "assert_text",
                "selector": "#route-title",
                "value": "bmw x3",
            }
        ),
        spec={
            "criteria": {
                "checkpoints": ['after search succeeds, page title contains "bmw x3"'],
                "final": [],
            }
        },
    )

    assert step["action"] == "assert_text_contains"
    assert step["value"] == "bmw x3"


def test_agent_case_execute_step_treats_soft_assertion_as_failure(tmp_path: Path):
    class FakeStepExecutor:
        step_has_error = False
        smart_resolver = None

        def execute_step(self, step):
            self.step_has_error = True

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
    executor.step_executor = FakeStepExecutor()

    with pytest.raises(AssertionError, match="Agent动作执行失败"):
        executor._execute_step(
            {"action": "assert_title", "value": "Products"},
            spec={"guardrails": {}},
        )


def test_agent_case_execute_step_switches_to_new_page_after_click(tmp_path: Path):
    class FakeLocator:
        @property
        def first(self):
            return self

        def evaluate(self, script):
            return True

    class FakeContext:
        def __init__(self):
            self.pages = []

    class FakePage:
        def __init__(self, url: str, context: FakeContext):
            self.url = url
            self.context = context

        def locator(self, selector):
            assert selector == "#search"
            return FakeLocator()

        def wait_for_load_state(self, state, timeout):
            return None

    class FakeUiHelper:
        def __init__(self, page):
            self.page = page
            self.pages = [page]

    class FakeStepExecutor:
        step_has_error = False
        smart_resolver = None

        def __init__(self, page, ui_helper, new_page):
            self.page = page
            self.ui_helper = ui_helper
            self._new_page = new_page

        def execute_step(self, step):
            self.page.context.pages.append(self._new_page)

    page_context = FakeContext()
    first_page = FakePage("https://example.test/start", page_context)
    new_page = FakePage("https://example.test/result", page_context)
    page_context.pages.append(first_page)
    ui_helper = FakeUiHelper(first_page)
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
        page=first_page,
        ui_helper=ui_helper,
        elements={},
        context=context,
    )
    executor.step_executor = FakeStepExecutor(first_page, ui_helper, new_page)

    executor._execute_step(
        {"action": "click", "selector": "#search"},
        spec={"guardrails": {}},
    )

    assert executor.page is new_page
    assert executor.step_executor.page is new_page
    assert ui_helper.page is new_page


def test_agent_case_execute_step_opens_target_blank_form_fallback(tmp_path: Path):
    class FakeLocator:
        @property
        def first(self):
            return self

        def evaluate(self, script):
            if "new URL" in script:
                return "https://search.example.test/?q=car"
            return True

    class FakeContext:
        def __init__(self):
            self.pages = []

        def new_page(self):
            page = FakePage("about:blank", self)
            self.pages.append(page)
            return page

    class FakePage:
        def __init__(self, url: str, context: FakeContext):
            self.url = url
            self.context = context

        def locator(self, selector):
            assert selector == "#submit"
            return FakeLocator()

        def goto(self, url, wait_until=None):
            self.url = url

        def wait_for_load_state(self, state, timeout):
            return None

    class FakeUiHelper:
        def __init__(self, page):
            self.page = page
            self.pages = [page]

    class FakeStepExecutor:
        step_has_error = False
        smart_resolver = None

        def __init__(self, page, ui_helper):
            self.page = page
            self.ui_helper = ui_helper

        def execute_step(self, step):
            return None

    page_context = FakeContext()
    first_page = FakePage("https://example.test/start", page_context)
    page_context.pages.append(first_page)
    ui_helper = FakeUiHelper(first_page)
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
        page=first_page,
        ui_helper=ui_helper,
        elements={},
        context=context,
    )
    executor.step_executor = FakeStepExecutor(first_page, ui_helper)

    executor._execute_step(
        {"action": "click", "selector": "#submit"},
        spec={"guardrails": {}},
    )

    assert executor.page is not first_page
    assert executor.page.url == "https://search.example.test/?q=car"
    assert ui_helper.page is executor.page


def test_agent_case_external_guard_uses_case_entry_scope(tmp_path: Path):
    class FakePage:
        url = "about:blank"

    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://internal.example.test/app",
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

    executor._guard_decision(
        AgentCaseDecision.model_validate(
            {"action": "goto", "value": "https://www.saucedemo.com/"}
        ),
        spec={
            "intent": "open https://www.saucedemo.com/ login page",
            "steps": [],
            "description": "",
            "guardrails": {"stop_on_external_domain": True},
        },
    )


def test_agent_case_external_guard_rejects_truncated_current_url(tmp_path: Path):
    class FakePage:
        url = "http://edit.ahohcrm.terra.corpautohome.com/#/login"

    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://internal.example.test/app",
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

    with pytest.raises(AgentDecisionRejected, match="truncated"):
        executor._guard_decision(
            AgentCaseDecision.model_validate(
                {"action": "goto", "value": "http://edit"}
            ),
            spec={
                "intent": "璁块棶url http://edit.ahohcrm.terra.corpautohome.com/#/login",
                "steps": [],
                "description": "",
                "guardrails": {"stop_on_external_domain": True},
            },
        )


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


def test_agent_case_use_module_infers_missing_params_from_inputs(tmp_path: Path):
    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://example.test/",
        elements={},
        modules={
            "login": [
                {"action": "fill", "selector": "username", "value": "${username}"},
                {
                    "action": "fill",
                    "selector": "password",
                    "value": "${common_password}",
                },
            ]
        },
        variables={"common_password": "secret"},
        test_cases=[],
        test_data={},
    )
    executor = AgentCaseExecutor(
        page=object(),
        ui_helper=object(),
        elements={},
        context=context,
    )

    step = executor._decision_to_step(
        AgentCaseDecision.model_validate(
            {"action": "use_module", "module": "login", "reason": "login"}
        ),
        spec={"inputs": {"username": "${standard_username}"}},
    )

    assert step == {
        "use_module": "login",
        "params": {"username": "${standard_username}"},
    }


def test_agent_case_use_module_rejects_missing_required_params(tmp_path: Path):
    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://example.test/",
        elements={},
        modules={
            "login": [
                {"action": "fill", "selector": "username", "value": "${username}"}
            ]
        },
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

    with pytest.raises(AgentDecisionRejected, match="username"):
        executor._decision_to_step(
            AgentCaseDecision.model_validate(
                {"action": "use_module", "module": "login", "reason": "login"}
            ),
            spec={"inputs": {}},
        )


def test_agent_case_marks_module_entry_only_for_explicit_module(tmp_path: Path):
    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://example.test/",
        elements={},
        modules={
            "login": [
                {"action": "fill", "selector": "username", "value": "${username}"},
                {"action": "click", "selector": "submit"},
            ]
        },
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

    natural_spec = executor._agent_spec(
        case_name="natural",
        case_data={
            "type": "agent_case",
            "intent": "visit login page and use OA login",
            "final": ["done"],
        },
    )
    explicit_spec = executor._agent_spec(
        case_name="explicit",
        case_data={
            "type": "agent_case",
            "steps": ["use project module login"],
            "final": ["done"],
        },
    )

    assert natural_spec["module_entry_allowed"] is False
    assert explicit_spec["module_entry_allowed"] is True


def test_agent_case_rejects_model_chosen_module_without_explicit_module(tmp_path: Path):
    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://example.test/",
        elements={},
        modules={
            "login": [
                {"action": "goto", "value": "https://example.test/login"},
                {"action": "fill", "selector": "username", "value": "${username}"},
            ],
            "login_and_open_search": [
                {"action": "goto", "value": "https://example.test/login"},
                {"action": "click", "selector": "text=Search"},
            ],
        },
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
        case_name="natural",
        case_data={
            "type": "agent_case",
            "intent": "visit login page and use OA login",
            "final": ["done"],
        },
    )

    with pytest.raises(AgentDecisionRejected, match="use_module仅允许"):
        executor._guard_decision(
            AgentCaseDecision.model_validate(
                {"action": "use_module", "module": "login", "reason": "repair"}
            ),
            spec=spec,
        )


def test_agent_case_runtime_harness_reports_phase_without_action_gate():
    spec = {
        "description": "",
        "intent": "complete natural steps in order",
        "steps": [
            "open login page https://example.test/login",
            "fill username and password",
            "click login button",
        ],
        "inputs": {"username": "${username}", "password": "${password}"},
    }
    dom_context = {
        "meta": {"url": "https://example.test/login", "title": "Login"},
        "forms": [
            {
                "id": "f1",
                "tag": "input",
                "name": "username",
                    "label": "OA account",
                "input_type": "text",
                "selector_candidates": ["#username"],
            },
            {
                "id": "f2",
                "tag": "input",
                    "label": "OA password",
                "input_type": "password",
                "selector_candidates": ["#password"],
            },
        ],
        "interactive_elements": [
            {
                "id": "e1",
                "tag": "button",
                "role": "button",
                    "text": "Login",
                    "selector_candidates": ['button:has-text("Login")'],
            },
        ],
        "assertion_candidates": [],
    }

    harness = _runtime_harness_state(
        spec=spec,
        history=[
            {
                "step": {
                    "action": "goto",
                    "value": "https://example.test/login",
                },
                "result": "passed",
                "url_after": "https://example.test/login",
            }
        ],
        dom_context=dom_context,
        current_url="https://example.test/login",
    )

    assert harness["phase"] == spec["steps"][1]
    assert "input" in harness["phase_categories"]
    assert "pending_slots" not in harness
    assert "allowed_actions" not in harness
    assert "recommended_actions" not in harness


def test_agent_case_runtime_harness_keeps_phase_as_context_only_after_fill():
    spec = {
        "description": "",
        "intent": "complete natural steps in order",
        "steps": ["fill username and password", "click login button"],
        "inputs": {"username": "${username}", "password": "${password}"},
    }
    dom_context = {
        "meta": {"url": "https://example.test/login", "title": "Login"},
        "forms": [
            {
                "id": "f1",
                "tag": "input",
                "name": "username",
                "input_type": "text",
                "selector_candidates": ["#username"],
            },
            {
                "id": "f2",
                "tag": "input",
                "input_type": "password",
                "selector_candidates": ["#password"],
            },
        ],
        "interactive_elements": [
            {
                "id": "e1",
                "tag": "button",
                "role": "button",
                    "text": "Login",
                    "selector_candidates": ['button:has-text("Login")'],
            },
        ],
        "assertion_candidates": [],
    }

    harness = _runtime_harness_state(
        spec=spec,
        history=[
            {
                "step": {
                    "action": "fill",
                    "selector": "#username",
                    "value": "${username}",
                },
                "result": "passed",
            }
        ],
        dom_context=dom_context,
        current_url="https://example.test/login",
    )

    assert "phase" in harness
    assert "phase_categories" in harness
    assert "pending_slots" not in harness
    assert "allowed_actions" not in harness


def test_agent_case_runtime_harness_advances_after_login_route_transition():
    spec = {
        "description": "",
        "intent": "visit url https://example.test/login, use OA login, expand mall menu, click product audit",
        "steps": [],
        "inputs": {"username": "${username}", "password": "${password}"},
    }
    history = [
        {
            "step": {"action": "goto", "value": "https://example.test/login"},
            "result": "passed",
            "url_after": "https://example.test/login",
        },
        {
            "step": {"action": "click", "selector": 'button:has-text("使用OA登录 >>")'},
            "result": "passed",
            "url_after": "https://sso.example.test/login",
        },
        {
            "step": {"action": "fill", "selector": "#username", "value": "${username}"},
            "result": "passed",
            "url_after": "https://sso.example.test/login",
        },
        {
            "step": {"action": "fill", "selector": "#password", "value": "${password}"},
            "result": "passed",
            "url_after": "https://sso.example.test/login",
        },
        {
            "step": {"action": "click", "selector": 'input[type="submit"]'},
            "result": "passed",
            "url_after": "https://example.test/home",
        },
    ]
    dom_context = {
        "meta": {"url": "https://example.test/home", "title": "Admin Home"},
        "page_summary": {"visible_text_summary": ["Mall", "Product Audit"]},
        "forms": [],
        "interactive_elements": [
            {
                "id": "e1",
                "tag": "div",
                "role": "menuitem",
                "text": "Mall",
                "selector_candidates": ['div:has-text("Mall")'],
            }
        ],
        "assertion_candidates": [],
    }

    harness = _runtime_harness_state(
        spec=spec,
        history=history,
        dom_context=dom_context,
        current_url="https://example.test/home",
    )

    assert harness["phase"] == "expand mall menu"
    assert harness["phase_observation"]["target_observable"] is True
    assert harness["phase_observation"]["next_target_observable"] is True


def test_agent_case_runtime_harness_keeps_login_phase_until_auth_completes():
    spec = {
        "description": "",
        "intent": "visit url https://example.test/login, use OA login, expand mall menu, click product audit",
        "steps": [],
        "inputs": {"username": "${username}", "password": "${password}"},
    }
    history = [
        {
            "step": {"action": "goto", "value": "https://example.test/login"},
            "result": "passed",
            "url_after": "https://example.test/login",
        },
        {
            "step": {"action": "click", "selector": 'button:has-text("使用OA登录 >>")'},
            "result": "passed",
            "url_after": "https://sso.example.test/login",
        },
    ]
    dom_context = {
        "meta": {"url": "https://sso.example.test/login", "title": "SSO Login"},
        "page_summary": {"visible_text_summary": ["Login", "OA account", "OA password"]},
        "forms": [
            {
                "id": "f1",
                "label": "OA account",
                "input_type": "text",
                "value_state": "empty",
                "selector_candidates": ["#username"],
            },
            {
                "id": "f2",
                "label": "OA password",
                "input_type": "password",
                "value_state": "empty",
                "selector_candidates": ["#password"],
            },
        ],
        "interactive_elements": [],
        "assertion_candidates": [],
    }

    harness = _runtime_harness_state(
        spec=spec,
        history=history,
        dom_context=dom_context,
        current_url="https://sso.example.test/login",
    )

    assert harness["phase"] == "use OA login"


def test_agent_case_runtime_harness_advances_after_completed_click_without_next_target():
    spec = {
        "description": "",
        "intent": "fill product id SPU17799383833255667, click search, click view log",
        "steps": [],
        "inputs": {"product_id": "SPU17799383833255667"},
    }
    history = [
        {
            "step": {
                "action": "fill",
                "selector": "#product-audit-search_spuCode",
                "value": "SPU17799383833255667",
                "_resolved_value_after": "SPU17799383833255667",
            },
            "result": "passed",
            "url_after": "https://example.test/productAuditList",
        },
        {
            "step": {
                "action": "click",
                "selector": 'button:has-text("Search")',
                "_action_before_url": "https://example.test/productAuditList",
                "_action_after_url": "https://example.test/productAuditList",
                "_action_dom_changed": True,
            },
            "result": "passed",
            "url_after": "https://example.test/productAuditList",
        },
    ]
    dom_context = {
        "meta": {
            "url": "https://example.test/productAuditList",
            "title": "Product Audit",
        },
        "page_summary": {"visible_text_summary": ["Product Audit"]},
        "forms": [],
        "interactive_elements": [],
        "assertion_candidates": [],
    }

    harness = _runtime_harness_state(
        spec=spec,
        history=history,
        dom_context=dom_context,
        current_url="https://example.test/productAuditList",
    )

    assert harness["phase"] == "click view log"
    assert harness["phase_observation"]["target_observable"] is False
    assert harness["plan_status"]["completed_count"] == 2
    assert harness["plan_status"]["remaining"] == ["click view log"]


def test_agent_case_runtime_harness_reports_visible_error_feedback():
    history = [
        {
            "step": {
                "action": "click",
                "selector": "input[type=submit]",
                "_action_before_url": "https://sso.example.test/login",
                "_action_after_url": "https://sso.example.test/login",
                "_action_page_errors": ["login is not defined"],
            },
            "result": "passed",
            "url_after": "https://sso.example.test/login",
        }
    ]
    dom_context = {
        "meta": {"url": "https://sso.example.test/login", "route_hint": "login"},
        "forms": [],
        "interactive_elements": [],
        "assertion_candidates": [
            {
                "id": "a1",
                "text": "账号或密码错误",
                "selector_candidates": ["#msg"],
            }
        ],
    }

    feedback = _runtime_feedback(
        history=history,
        dom_context=dom_context,
        current_url="https://sso.example.test/login",
    )

    assert feedback["last_action"]["page_errors"] == ["login is not defined"]
    assert feedback["last_action"]["url_changed"] is False
    assert feedback["filled_fields"] == []
    assert feedback["visible_errors"] == ["账号或密码错误"]
    assert feedback["stalled_on_url"] == 1


def test_agent_case_runtime_feedback_reports_filled_fields_as_facts():
    feedback = _runtime_feedback(
        history=[
            {
                "step": {
                    "action": "fill",
                    "selector": "#username",
                    "value": "${username}",
                },
                "result": "passed",
                "url_after": "https://sso.example.test/login",
            },
            {
                "step": {
                    "action": "fill",
                    "selector": "#password",
                    "value": "${password}",
                },
                "result": "passed",
                "url_after": "https://sso.example.test/login",
            },
        ],
        current_url="https://sso.example.test/login",
        dom_context={
            "meta": {"url": "https://sso.example.test/login"},
            "forms": [],
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    )

    assert feedback["filled_fields"] == [
        {"target": "#username", "value": "${username}"},
        {"target": "#password", "value": "${password}"},
    ]
    assert "allowed_actions" not in feedback
    assert "recommended_actions" not in feedback


def test_parse_agent_decision_normalizes_press_value_to_key():
    decision = _parse_agent_decision_response(
        '{"action":"press","target":"body","value":"PageDown"}',
        spec={"inputs": {}},
        dom_context={},
    )

    assert decision.action == "press"
    assert decision.key == "PageDown"
    assert decision.value is None


def test_agent_case_click_without_observed_progress_does_not_satisfy_intent():
    spec = {
        "description": "",
        "intent": "expand mall menu, click product audit",
        "steps": [],
        "inputs": {},
    }
    history = [
        {
            "step": {
                "action": "click",
                "selector": "text=Mall",
                "_action_before_url": "https://example.test/home",
                "_action_after_url": "https://example.test/home",
                "_action_dom_changed": False,
                "_action_target_text": "Mall",
                "_action_target_visible_after": False,
            },
            "result": "passed",
            "url_after": "https://example.test/home",
        }
    ]

    unmet = _unmet_intent_action_requirements(
        spec=spec,
        history=history,
        current_url="https://example.test/home",
        dom_context={
            "meta": {"url": "https://example.test/home"},
            "forms": [],
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    )

    assert "intent click: mall" not in unmet
    assert any(item.startswith("intent click: product audit") for item in unmet)


def test_agent_case_unmatched_click_without_observed_progress_does_not_satisfy_next_intent():
    spec = {
        "description": "",
        "intent": "expand mall menu, click product audit",
        "steps": [],
        "inputs": {},
    }
    history = [
        {
            "step": {
                "action": "click",
                "selector": "text=Mall",
                "_action_before_url": "https://example.test/home",
                "_action_after_url": "https://example.test/home",
                "_action_dom_changed": False,
                "_action_target_text": "Mall",
                "_action_target_visible_after": False,
            },
            "result": "passed",
            "url_after": "https://example.test/home",
        }
    ]

    unmet = _unmet_intent_action_requirements(
        spec=spec,
        history=history,
        current_url="https://example.test/home",
        dom_context={
            "meta": {"url": "https://example.test/home"},
            "forms": [],
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    )

    assert any(item.startswith("intent click: product audit") for item in unmet)


def test_agent_case_click_with_observed_dom_change_satisfies_intent():
    spec = {
        "description": "",
        "intent": "expand mall menu",
        "steps": [],
        "inputs": {},
    }
    history = [
        {
            "step": {
                "action": "click",
                "selector": "text=Mall",
                "_action_before_url": "https://example.test/home",
                "_action_after_url": "https://example.test/home",
                "_action_dom_changed": True,
                "_action_target_text": "Mall",
            },
            "result": "passed",
            "url_after": "https://example.test/home",
        }
    ]

    assert _unmet_intent_action_requirements(
        spec=spec,
        history=history,
        current_url="https://example.test/home",
        dom_context={
            "meta": {"url": "https://example.test/home"},
            "forms": [],
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    ) == []


def test_agent_case_self_healed_executed_selector_satisfies_intent():
    spec = {
        "description": "",
        "intent": "click view log",
        "steps": [],
        "inputs": {},
    }
    history = [
        {
            "step": {
                "action": "click",
                "selector": "text=View Log",
                "_action_executed_selector": 'a:has-text("Log")',
                "_action_target_text": "View Log",
                "_action_before_url": "https://example.test/productAuditList",
                "_action_after_url": "https://example.test/productAuditList",
                "_action_dom_changed": False,
            },
            "result": "passed",
            "url_after": "https://example.test/productAuditList",
        }
    ]

    assert _unmet_intent_action_requirements(
        spec=spec,
        history=history,
        current_url="https://example.test/productAuditList",
        dom_context={
            "meta": {"url": "https://example.test/productAuditList"},
            "forms": [],
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    ) == []


def test_agent_case_oa_login_click_does_not_satisfy_login_button_step():
    spec = {
        "description": "",
        "intent": "",
        "steps": ["click OA login", "click login button"],
        "inputs": {},
    }
    history = [
        {
            "step": {
                "action": "click",
                "selector": 'button:has-text("使用OA登录 >>")',
                "_action_target_text": "使用OA登录 >>",
                "_action_before_url": "https://example.test/login",
                "_action_after_url": "https://sso.example.test/login",
            },
            "result": "passed",
            "url_after": "https://sso.example.test/login",
        }
    ]

    assert _unmet_intent_action_requirements(spec=spec, history=history) == [
        "intent click: login"
    ]


def test_agent_case_auth_submit_after_credentials_satisfies_login_button_step():
    spec = {
        "description": "",
        "intent": "",
        "steps": ["fill username and password", "click login button"],
        "inputs": {"username": "${username}", "password": "${password}"},
    }
    history = [
        {
            "step": {"action": "fill", "selector": "#username", "value": "${username}"},
            "result": "passed",
            "url_after": "https://sso.example.test/login",
        },
        {
            "step": {"action": "fill", "selector": "#password", "value": "${password}"},
            "result": "passed",
            "url_after": "https://sso.example.test/login",
        },
        {
            "step": {
                "action": "click",
                "selector": 'input[type="submit"]',
                "_action_before_url": "https://sso.example.test/login",
                "_action_after_url": "https://example.test/home",
            },
            "result": "passed",
            "url_after": "https://example.test/home",
        },
    ]

    assert _unmet_intent_action_requirements(spec=spec, history=history) == []


def test_agent_case_runtime_harness_advances_after_auth_submit_in_steps():
    spec = {
        "description": "",
        "intent": "",
        "steps": [
            "fill username and password",
            "click login button",
            "expand mall menu",
        ],
        "inputs": {"username": "${username}", "password": "${password}"},
    }
    history = [
        {
            "step": {"action": "fill", "selector": "#username", "value": "${username}"},
            "result": "passed",
            "url_after": "https://sso.example.test/login",
        },
        {
            "step": {"action": "fill", "selector": "#password", "value": "${password}"},
            "result": "passed",
            "url_after": "https://sso.example.test/login",
        },
        {
            "step": {
                "action": "click",
                "selector": 'input[type="submit"]',
                "_action_before_url": "https://sso.example.test/login",
                "_action_after_url": "https://example.test/home",
            },
            "result": "passed",
            "url_after": "https://example.test/home",
        },
    ]
    dom_context = {
        "meta": {"url": "https://example.test/home", "title": "Admin Home"},
        "page_summary": {"visible_text_summary": ["Mall"]},
        "forms": [],
        "interactive_elements": [
            {
                "id": "e1",
                "tag": "div",
                "role": "menuitem",
                "text": "Mall",
                "selector_candidates": ['div:has-text("Mall")'],
            }
        ],
        "assertion_candidates": [],
    }

    harness = _runtime_harness_state(
        spec=spec,
        history=history,
        dom_context=dom_context,
        current_url="https://example.test/home",
    )

    assert harness["phase"] == "expand mall menu"


def test_agent_case_decision_records_element_text_as_action_target(tmp_path: Path):
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
    executor.current_dom_context = {
        "forms": [],
        "interactive_elements": [
            {
                "id": "e74",
                "tag": "div",
                "role": "menuitem",
                "name": "Mall",
                "text": "Mall",
                "selector_candidates": ["li:nth-of-type(19) > div"],
            }
        ],
        "assertion_candidates": [],
    }

    step = executor._decision_to_step(
        AgentCaseDecision.model_validate({"action": "click", "element_id": "e74"})
    )

    assert step["selector"] == "li:nth-of-type(19) > div"
    assert step["_action_target_text"] == "Mall"


def test_agent_case_element_id_target_text_satisfies_menu_intent():
    spec = {
        "description": "",
        "intent": "expand mall menu, click product audit",
        "steps": [],
        "inputs": {},
    }
    history = [
        {
            "step": {
                "action": "click",
                "selector": "li:nth-of-type(19) > div",
                "_action_target_text": "Mall",
                "_action_before_url": "https://example.test/home",
                "_action_after_url": "https://example.test/home",
                "_action_dom_changed": True,
            },
            "result": "passed",
            "url_after": "https://example.test/home",
        }
    ]

    assert _unmet_intent_action_requirements(spec=spec, history=history) == [
        "intent click: product audit/product/audit"
    ]


def test_agent_case_runtime_harness_advances_after_semantic_menu_click():
    spec = {
        "description": "",
        "intent": "expand mall menu, click product audit, fill product id SPU17799383833255667",
        "steps": [],
        "inputs": {"product_id": "SPU17799383833255667"},
    }
    history = [
        {
            "step": {
                "action": "click",
                "selector": "li:nth-of-type(19) > div",
                "_action_target_text": "Mall",
                "_action_before_url": "https://example.test/home",
                "_action_after_url": "https://example.test/home",
                "_action_dom_changed": True,
            },
            "result": "passed",
            "url_after": "https://example.test/home",
        },
        {
            "step": {
                "action": "click",
                "selector": 'li[title="Product Audit"]',
                "_action_target_text": "Product Audit",
                "_action_before_url": "https://example.test/home",
                "_action_after_url": "https://example.test/productAuditList",
            },
            "result": "passed",
            "url_after": "https://example.test/productAuditList",
        },
    ]
    dom_context = {
        "meta": {
            "url": "https://example.test/productAuditList",
            "title": "Product Audit",
        },
        "forms": [
            {
                "id": "f1",
                "type": "input",
                "input_type": "text",
                "label": "Product ID",
                "selector_candidates": ["#product-audit-search_spuCode"],
            }
        ],
        "interactive_elements": [],
        "assertion_candidates": [],
    }

    harness = _runtime_harness_state(
        spec=spec,
        history=history,
        dom_context=dom_context,
        current_url="https://example.test/productAuditList",
    )

    assert harness["phase"] == "fill product id SPU17799383833255667"


def test_agent_case_runtime_feedback_uses_direct_error_text_only():
    feedback = _runtime_feedback(
        history=[],
        current_url="https://sso.example.test/login",
        dom_context={
            "meta": {"url": "https://sso.example.test/login"},
            "forms": [],
            "interactive_elements": [],
            "assertion_candidates": [
                {
                    "id": "a1",
                    "text": "账号或密码错误",
                    "near_text": "汽车人APP下载 Autohome 登录 显示密码 修改密码 忘记密码",
                    "selector_candidates": ["#msg"],
                }
            ],
        },
    )

    assert feedback["visible_errors"] == ["账号或密码错误"]


def test_compact_history_keeps_action_outcome_feedback():
    history = [
        {
            "step": {
                "action": "click",
                "selector": "#login",
                "_action_before_url": "https://example.test/login",
                "_action_after_url": "https://example.test/login",
                "_action_page_errors": ["login is not defined"],
            },
            "result": "passed",
            "url_after": "https://example.test/login",
        }
    ]

    compacted = compact_history(history)

    assert compacted[0]["step"]["_action_page_errors"] == ["login is not defined"]
    assert compacted[0]["step"]["_action_after_url"] == "https://example.test/login"


def test_compact_history_keeps_click_observation_feedback():
    compacted = compact_history(
        [
            {
                "step": {
                    "action": "click",
                    "selector": "text=Mall",
                    "_action_before_url": "https://example.test/home",
                    "_action_after_url": "https://example.test/home",
                    "_action_dom_changed": False,
                    "_action_target_text": "Mall",
                    "_action_target_visible_after": False,
                },
                "result": "passed",
                "url_after": "https://example.test/home",
            }
        ]
    )

    step = compacted[0]["step"]
    assert step["_action_dom_changed"] is False
    assert step["_action_target_text"] == "Mall"
    assert step["_action_target_visible_after"] is False
    assert step["_action_after_url"] == "https://example.test/home"


def test_compact_history_keeps_executed_selector_feedback():
    compacted = compact_history(
        [
            {
                "step": {
                    "action": "click",
                    "selector": "text=View Log",
                    "_action_executed_selector": 'a:has-text("Log")',
                    "_action_target_text": "View Log",
                },
                "result": "passed",
                "url_after": "https://example.test/list",
            }
        ]
    )

    step = compacted[0]["step"]
    assert step["_action_executed_selector"] == 'a:has-text("Log")'


def test_compact_model_dom_context_adds_menu_outline_and_backfills_menu_items():
    dom_context = {
        "meta": {"url": "https://example.test/home"},
        "page_summary": {
            "visible_text_summary": [
                "Menu 1 Menu 2 Menu 3 Menu 4 Menu 5 Menu 6 Menu 7 Menu 8"
            ]
        },
        "forms": [],
        "interactive_elements": [
            {
                "id": f"e{index}",
                "tag": "div",
                "role": "menuitem",
                "text": f"Menu {index}",
                "selector_candidates": [f"li:nth-of-type({index}) > div"],
            }
            for index in range(1, 8)
        ],
        "assertion_candidates": [],
    }

    compacted = compact_model_dom_context(
        dom_context,
        candidate_limit=3,
        selector_limit=1,
        form_limit=0,
        assertion_limit=0,
        hints=["open missing target"],
        include_business_objects=False,
        include_compression=False,
    )

    assert compacted["page_summary"]["menu_text_outline"][:3] == [
        "Menu 1",
        "Menu 2",
        "Menu 3",
    ]
    assert len(compacted["interactive_elements"]) == 3
    assert compacted["interactive_elements"][0]["text"] == "Menu 1"


def test_compact_model_dom_context_frontloads_current_phase_target():
    dom_context = {
        "meta": {"url": "https://example.test/productAuditList"},
        "page_summary": {"visible_text_summary": ["Product Audit"]},
        "forms": [],
        "interactive_elements": [
            {
                "id": "e1",
                "tag": "div",
                "role": "menuitem",
                "text": "Mall",
                "selector_candidates": ['div:has-text("Mall")'],
            },
            {
                "id": "e2",
                "tag": "button",
                "role": "button",
                "text": "Search",
                "selector_candidates": ['button:has-text("Search")'],
            },
            {
                "id": "e3",
                "tag": "a",
                "role": "link",
                "text": "View Log",
                "selector_candidates": ['a:has-text("View Log")'],
            },
        ],
        "assertion_candidates": [],
    }

    compacted = compact_model_dom_context(
        dom_context,
        candidate_limit=2,
        selector_limit=1,
        form_limit=0,
        assertion_limit=0,
        hints=["根据实时DOM推进当前阶段: click view log", ["intent click: view log"]],
        include_business_objects=False,
        include_compression=False,
    )

    assert compacted["interactive_elements"][0]["text"] == "View Log"


def test_agent_case_has_no_local_harness_action_decision():
    assert not hasattr(AgentCaseExecutor, "_local_harness_decision")


def test_parse_fill_instruction_supports_input_box_phrasing():
    assert _parse_fill_instruction("fill password field with secret_sauce") == (
        "password field",
        "secret_sauce",
    )
    assert _parse_fill_instruction('fill username field with "standard_user"') == (
        "username field",
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


def test_agent_provider_can_use_runtime_model_override(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.load_ai_config",
        lambda: {
            "runtime": {"agent_model": "runtime-fast-model"},
            "agent_policy": {"limits": {}},
        },
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.load_llm_settings",
        lambda: LLMSettings(
            url="http://llm.test/chat/completions",
            api_key="test-key",
            model="global-slow-model",
        ),
    )
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

    assert executor._agent_provider().settings.model == "runtime-fast-model"


def test_agent_case_plan_cache_runs_without_model_compile(
    monkeypatch, tmp_path: Path
):
    executed: list[dict] = []

    class FakeStepExecutor:
        def __init__(self, page, ui_helper, elements, default_mode=None):
            self.page = page

        def execute_step(self, step):
            executed.append(step)
            if step.get("action") == "goto":
                self.page.url = step["value"]

    class FakePage:
        url = "about:blank"

    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.StepExecutor",
        FakeStepExecutor,
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.load_ai_config",
        lambda: {
            "runtime": {
                "agent_case_plan_cache_enabled": True,
                "ai_cache_sqlite_path": str(tmp_path / "ai_cache.sqlite3"),
                "candidate_limit": 5,
                "agent_completion_wait_seconds": 0,
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
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor._unmet_completion_criteria",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor._unmet_intent_action_requirements",
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
        "intent": "open https://www.saucedemo.com/ and finish flow",
        "final": ["done"],
    }
    spec = executor._agent_spec(case_name="test_agent", case_data=case_data)
    cache_key = executor._cache_key(case_name="test_agent", spec=spec)
    payload = {
        "cases": [{"name": "test_agent"}],
        "data": {
            "test_agent": {
                "mode": "smart",
                "steps": [
                    {"action": "goto", "value": "https://www.saucedemo.com/"},
                    {"action": "click", "target": "Login button", "mode": "smart"},
                ],
            }
        },
        "elements": {},
        "modules": {},
        "vars": {},
    }
    AgentCasePlanCache(tmp_path / "ai_cache.sqlite3").save_plan(
        key=cache_key,
        project="demo",
        env="prod",
        case_name="test_agent",
        entry_scope=executor._entry_scope(spec),
        spec=spec,
        payload=payload,
        case_payload_name="test_agent",
        steps=payload["data"]["test_agent"]["steps"],
        prompt_version="agent-case-test",
        schema_version="schema-test",
        model="test-model",
        asset_hash="asset",
    )
    monkeypatch.setattr(
        executor,
        "_decide_next_action",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("runtime planner should not run")
        ),
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor._build_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("compile model should not run")
        ),
    )
    monkeypatch.setattr(executor, "_local_completion_decision", lambda **kwargs: None)
    monkeypatch.setattr(executor, "_maybe_open_start_url", lambda **kwargs: None)

    result = executor.run(case_name="test_agent", case_data=case_data)

    assert result.model_calls == 0
    assert result.final_reason == "compiled steps executed"
    assert executed[0] == {"action": "goto", "value": "https://www.saucedemo.com/"}
    assert {
        key: executed[1].get(key)
        for key in ("action", "target", "mode")
    } == {"action": "click", "target": "Login button", "mode": "smart"}
    assert executed[1]["_action_before_url"] == "https://www.saucedemo.com/"
    assert executed[1]["_action_after_url"] == "https://www.saucedemo.com/"


def test_agent_case_runs_compiled_steps_without_realtime_planning(
    monkeypatch, tmp_path: Path
):
    executed: list[dict] = []

    class FakePage:
        url = "about:blank"

    payload = {
        "cases": [{"name": "test_agent"}],
        "data": {
            "test_agent": {
                "mode": "smart",
                "steps": [
                    {"action": "click", "selector": "compiled_button"},
                    {"action": "assert_title_contains", "value": "Done"},
                ],
            }
        },
        "elements": {"compiled_button": "button:has-text('Run')"},
        "modules": {},
        "vars": {"compiled_value": "ok"},
    }

    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.load_ai_config",
        lambda: {
            "runtime": {
                "agent_case_plan_cache_enabled": False,
                "ai_cache_sqlite_path": str(tmp_path / "ai_cache.sqlite3"),
                "candidate_limit": 5,
                "agent_completion_wait_seconds": 0,
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
        "src.ai_runtime.agent_case_executor._build_payload",
        lambda *args, **kwargs: (
            kwargs.get("cache_info", {}).update(
                {"cache_hit": False, "model_calls": 1}
            )
            or payload
        ),
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor._unmet_completion_criteria",
        lambda **kwargs: (
            []
            if any(
                (item.get("step") or {}).get("action") == "assert_title_contains"
                for item in kwargs.get("history", [])
                if isinstance(item, dict)
            )
            else ["Done"]
        ),
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor._unmet_intent_action_requirements",
        lambda **kwargs: [],
    )

    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="",
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
    monkeypatch.setattr(executor, "_maybe_open_start_url", lambda **kwargs: None)
    monkeypatch.setattr(
        executor,
        "_decide_next_action",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("realtime planner should not run")
        ),
    )

    def fake_execute_step(step, *, spec):
        executed.append(dict(step))

    monkeypatch.setattr(executor, "_execute_step", fake_execute_step)

    result = executor.run(
        case_name="test_agent",
        case_data={
            "type": "agent_case",
            "intent": "click run and verify done",
            "final": ["Done"],
        },
    )

    assert result.model_calls == 1
    assert result.final_reason == "compiled steps executed"
    assert executed[0]["selector"] == "compiled_button"
    assert executor.step_executor.elements["compiled_button"] == "button:has-text('Run')"
    assert not (tmp_path / "cases" / "agent_case.yaml").exists()


def test_agent_case_compiled_plan_cache_skips_generation_cache(
    monkeypatch, tmp_path: Path
):
    cache_path = tmp_path / "ai_cache.sqlite3"
    calls = {"count": 0}
    executed: list[dict] = []

    class FakePage:
        url = "about:blank"

    payload = {
        "cases": [{"name": "test_agent"}],
        "data": {
            "test_agent": {
                "mode": "smart",
                "steps": [{"action": "assert_title_contains", "value": "Done"}],
            }
        },
        "elements": {},
        "modules": {},
        "vars": {},
    }

    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.load_ai_config",
        lambda: {
            "runtime": {
                "agent_case_plan_cache_enabled": True,
                "ai_cache_sqlite_path": str(cache_path),
                "candidate_limit": 5,
                "agent_completion_wait_seconds": 0,
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
        "src.ai_runtime.agent_case_executor._unmet_completion_criteria",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor._unmet_intent_action_requirements",
        lambda **kwargs: [],
    )

    def fake_build_payload(*args, **kwargs):
        calls["count"] += 1
        kwargs.get("cache_info", {}).update({"cache_hit": False, "model_calls": 1})
        return payload

    monkeypatch.setattr("src.ai_runtime.agent_case_executor._build_payload", fake_build_payload)

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
        page=FakePage(),
        ui_helper=object(),
        elements={},
        context=context,
    )
    monkeypatch.setattr(executor, "_maybe_open_start_url", lambda **kwargs: None)
    monkeypatch.setattr(executor, "_execute_step", lambda step, *, spec: executed.append(dict(step)))
    monkeypatch.setattr(
        executor, "_wait_for_compiled_completion_criteria", lambda **kwargs: []
    )
    monkeypatch.setattr(
        executor,
        "_decide_next_action",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("runtime planner should not run")
        ),
    )
    case_data = {
        "type": "agent_case",
        "intent": "click run and verify done",
        "final": ["Done"],
    }

    result = executor.run(
        case_name="test_agent",
        case_data=case_data,
    )

    assert result.model_calls == 1
    assert result.final_reason == "compiled steps executed"
    assert calls["count"] == 1
    assert executed == [{"action": "assert_title_contains", "value": "Done"}]

    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor._build_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("compile model should not run after plan cache save")
        ),
    )
    executed.clear()
    second_executor = AgentCaseExecutor(
        page=FakePage(),
        ui_helper=object(),
        elements={},
        context=context,
    )
    monkeypatch.setattr(second_executor, "_maybe_open_start_url", lambda **kwargs: None)
    monkeypatch.setattr(second_executor, "_execute_step", lambda step, *, spec: executed.append(dict(step)))
    monkeypatch.setattr(
        second_executor,
        "_wait_for_compiled_completion_criteria",
        lambda **kwargs: [],
    )
    result = second_executor.run(case_name="test_agent", case_data=case_data)

    assert result.model_calls == 0
    assert executed == [{"action": "assert_title_contains", "value": "Done"}]


def test_agent_case_compile_rejects_payload_with_generated_assets(
    monkeypatch, tmp_path: Path
):
    cache_path = tmp_path / "ai_cache.sqlite3"
    payload = {
        "cases": [{"name": "test_agent"}],
        "data": {
            "test_agent": {
                "mode": "smart",
                "steps": [{"action": "assert_title_contains", "value": "Done"}],
            }
        },
        "elements": {},
        "modules": {"generated_login": [{"action": "click", "selector": "#login"}]},
        "vars": {},
    }
    config = {
        "runtime": {
            "agent_case_plan_cache_enabled": True,
            "ai_cache_sqlite_path": str(cache_path),
            "candidate_limit": 5,
            "agent_completion_wait_seconds": 0,
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
    }
    calls = {"count": 0}

    class FakePage:
        url = "about:blank"

    def fake_build_payload(*args, **kwargs):
        calls["count"] += 1
        kwargs.get("cache_info", {}).update({"cache_hit": False, "model_calls": 1})
        return payload

    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.load_ai_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor._build_payload",
        fake_build_payload,
    )

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
        page=FakePage(),
        ui_helper=object(),
        elements={},
        context=context,
    )
    monkeypatch.setattr(executor, "_maybe_open_start_url", lambda **kwargs: None)
    monkeypatch.setattr(executor, "_execute_step", lambda step, *, spec: None)
    monkeypatch.setattr(
        executor, "_wait_for_compiled_completion_criteria", lambda **kwargs: []
    )
    monkeypatch.setattr(
        executor,
        "_decide_next_action",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("runtime planner should not run")
        ),
    )

    with pytest.raises(
        AgentCaseCompileContractError,
        match="run_case agent_case 不允许新建module",
    ):
        executor.run(
            case_name="test_agent",
            case_data={
                "type": "agent_case",
                "intent": "click run and verify done",
                "final": ["Done"],
            },
        )

    spec = executor._agent_spec(
        case_name="test_agent",
        case_data={
            "type": "agent_case",
            "intent": "click run and verify done",
            "final": ["Done"],
        },
    )
    assert executor.plan_cache.load_plan(
        executor._cache_key(case_name="test_agent", spec=spec)
    ) is None
    assert calls["count"] == 1


def test_agent_case_generation_spec_forbids_new_runtime_modules():
    generation_spec = _agent_spec_to_generation_spec(
        case_name="test_agent",
        spec={
            "description": "",
            "intent": "login and search",
            "steps": ["使用登录模块", "查询商品"],
            "inputs": {"product_id": "SPU1"},
            "criteria": {"final": ["完成"]},
        },
        allowed_modules=["crm_login_and_navigate"],
    )

    assert generation_spec["runtime_compile"] == {
        "mode": "agent_case",
        "allow_new_modules": False,
        "allowed_modules": ["crm_login_and_navigate"],
    }


def test_agent_case_runtime_compile_rejects_generated_modules():
    payload = {
        "cases": [{"name": "test_agent"}],
        "data": {
            "test_agent": {
                "mode": "smart",
                "steps": [{"use_module": "generated_login", "params": {}}],
            }
        },
        "elements": {},
        "modules": {"generated_login": [{"action": "click", "selector": "#login"}]},
        "vars": {},
    }

    with pytest.raises(
        AgentCaseCompileContractError,
        match="run_case agent_case 不允许新建module",
    ):
        _validate_runtime_compiled_payload(payload, allowed_modules=set())


def test_agent_case_runtime_compile_rejects_unknown_module_reference():
    payload = {
        "cases": [{"name": "test_agent"}],
        "data": {
            "test_agent": {
                "mode": "smart",
                "steps": [{"use_module": "generated_login", "params": {}}],
            }
        },
        "elements": {},
        "modules": {},
        "vars": {},
    }

    with pytest.raises(
        AgentCaseCompileContractError,
        match="只能引用当前项目YAML中已存在的module",
    ):
        _validate_runtime_compiled_payload(
            payload,
            allowed_modules={"crm_login_and_navigate"},
        )


def test_agent_case_runtime_compile_allows_existing_yaml_module_reference():
    payload = {
        "cases": [{"name": "test_agent"}],
        "data": {
            "test_agent": {
                "mode": "smart",
                "steps": [
                    {"use_module": "crm_login_and_navigate", "params": {}},
                    {"action": "assert_title_contains", "value": "商品审核"},
                ],
            }
        },
        "elements": {},
        "modules": {},
        "vars": {},
    }

    _validate_runtime_compiled_payload(
        payload,
        allowed_modules={"crm_login_and_navigate"},
    )


def test_agent_case_compile_contract_error_does_not_fallback_to_realtime(
    monkeypatch, tmp_path: Path
):
    payload = {
        "cases": [{"name": "test_agent"}],
        "data": {
            "test_agent": {
                "mode": "smart",
                "steps": [{"use_module": "crm_search_param", "params": {}}],
            }
        },
        "elements": {},
        "modules": {"crm_search_param": [{"action": "click", "selector": "#search"}]},
        "vars": {},
    }
    config = {
        "runtime": {
            "agent_case_plan_cache_enabled": False,
            "ai_cache_sqlite_path": str(tmp_path / "ai_cache.sqlite3"),
            "candidate_limit": 5,
            "agent_completion_wait_seconds": 0,
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
    }

    class FakePage:
        url = "about:blank"

    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.load_ai_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor._build_payload",
        lambda *args, **kwargs: (
            kwargs.get("cache_info", {}).update(
                {"cache_hit": False, "model_calls": 1}
            )
            or payload
        ),
    )

    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://example.test/",
        elements={},
        modules={"crm_login_and_navigate": [{"action": "goto", "value": "/"}]},
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
    monkeypatch.setattr(executor, "_maybe_open_start_url", lambda **kwargs: None)
    monkeypatch.setattr(
        executor,
        "_decide_next_action",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("runtime planner should not run after compile contract error")
        ),
    )

    with pytest.raises(
        AgentCaseCompileContractError,
        match="run_case agent_case 不允许新建module",
    ):
        executor.run(
            case_name="test_agent",
            case_data={
                "type": "agent_case",
                "intent": "login and search",
                "final": ["Done"],
            },
        )


def test_agent_case_compiled_failure_reports_failed_step_without_realtime_fallback(
    monkeypatch, tmp_path: Path
):
    events: list[Any] = []
    payload = {
        "cases": [{"name": "test_agent"}],
        "data": {
            "test_agent": {
                "mode": "smart",
                "steps": [
                    {"action": "click", "selector": "#missing"},
                    {"action": "assert_title_contains", "value": "Done"},
                ],
            }
        },
        "elements": {},
        "modules": {},
        "vars": {},
    }
    config = {
        "runtime": {
            "agent_case_plan_cache_enabled": False,
            "ai_cache_sqlite_path": str(tmp_path / "ai_cache.sqlite3"),
            "candidate_limit": 5,
            "agent_execution_retry_limit": 1,
        },
        "agent_policy": {
            "limits": {
                "max_steps": 5,
                "max_model_calls": 5,
                "max_duration_seconds": 30,
            },
            "guardrails": {"require_checkpoints_or_final": True},
        },
        "generation": {"max_context_items": 5},
        "prompts": {"agent_case_version": "agent-case-test"},
        "llm": {"schema_version": "schema-test"},
    }

    class FakePage:
        url = "about:blank"

    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.load_ai_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor._build_payload",
        lambda *args, **kwargs: (
            kwargs.get("cache_info", {}).update(
                {"cache_hit": False, "model_calls": 1}
            )
            or payload
        ),
    )

    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://example.test/start",
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

    def fake_open_start_url(**kwargs):
        events.append(
                (
                    "open_start",
                    kwargs.get("force", False),
                    [item["step"]["action"] for item in kwargs["history"]],
                )
            )
        kwargs["history"].append(
            executor._history_item(
                step={"action": "goto", "value": "https://example.test/start"},
                source="bootstrap",
            )
        )

    def fake_execute_step(step, *, spec):
        events.append(("execute", step["action"], step.get("selector")))
        if step.get("selector") == "#missing":
            raise RuntimeError("selector not found")

    monkeypatch.setattr(executor, "_maybe_open_start_url", fake_open_start_url)
    monkeypatch.setattr(executor, "_execute_step", fake_execute_step)
    monkeypatch.setattr(
        executor,
        "_decide_next_action",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("runtime planner should not run after compile failure")
        ),
    )

    with pytest.raises(AssertionError, match="Agent编译步骤执行失败.*#missing"):
        executor.run(
            case_name="test_agent",
            case_data={
                "type": "agent_case",
                "intent": "open start and click missing",
                "final": ["done"],
            },
        )

    assert ("execute", "click", "#missing") in events
    assert ("open_start", False, []) in events
    assert not any(
        item[0] == "open_start" and item[1] is True for item in events
    )


def test_agent_case_compiled_success_allows_verified_selector_cache_commit(
    monkeypatch, tmp_path: Path
):
    events: list[Any] = []
    payload = {
        "cases": [{"name": "test_agent"}],
        "data": {
            "test_agent": {
                "mode": "smart",
                "steps": [
                    {
                        "action": "click",
                        "target": "login button",
                        "mode": "smart",
                    },
                    {"action": "assert_title_contains", "value": "Done"},
                ],
            }
        },
        "elements": {},
        "modules": {},
        "vars": {},
    }

    class FakeResolver:
        def resolve(self, **kwargs):
            return ResolvedSelector(
                selector="#login",
                source="heuristic",
                confidence=0.9,
                cache_action="click",
                cache_target=kwargs["target"],
                cache_page_key="about:blank",
            )

        def record_verified_selector(self, **kwargs):
            events.append(("selector_cache", kwargs))

    class FakePage:
        url = "about:blank"

    class FakeUiHelper:
        pass

    def fake_step_executor(page, ui_helper, elements, default_mode=None):
        executor = StepExecutor(page, ui_helper, elements, default_mode=default_mode)
        executor.smart_resolver = FakeResolver()
        return executor

    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.StepExecutor",
        fake_step_executor,
    )
    monkeypatch.setattr(
        "src.step_actions.step_executor.execute_action_with_command",
        lambda ui_helper, action, selector, value, step: events.append(
            ("execute", action, selector, value)
        ),
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.load_ai_config",
        lambda: {
            "runtime": {
                "agent_case_plan_cache_enabled": False,
                "ai_cache_sqlite_path": str(tmp_path / "ai_cache.sqlite3"),
                "candidate_limit": 5,
                "agent_completion_wait_seconds": 0,
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
        "src.ai_generation.case_generator.load_ai_config",
        lambda: {
            "runtime": {"ai_cache_sqlite_path": str(tmp_path / "ai_cache.sqlite3")},
            "generation": {"max_context_items": 5},
            "llm": {"schema_version": "schema-test"},
        },
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor._build_payload",
        lambda *args, **kwargs: (
            kwargs.get("cache_info", {}).update(
                {"cache_hit": False, "model_calls": 1}
            )
            or payload
        ),
    )
    monkeypatch.setenv("UI_SELECTOR_CACHE_COMMIT_MODE", "deferred")
    step_executor_module.discard_pending_selector_cache("test setup")

    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="",
        elements={},
        modules={},
        variables={},
        test_cases=[],
        test_data={},
    )
    executor = AgentCaseExecutor(
        page=FakePage(),
        ui_helper=FakeUiHelper(),
        elements={},
        context=context,
    )
    monkeypatch.setattr(executor, "_maybe_open_start_url", lambda **kwargs: None)
    monkeypatch.setattr(
        executor, "_wait_for_compiled_completion_criteria", lambda **kwargs: []
    )
    monkeypatch.setattr(
        executor,
        "_decide_next_action",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("runtime planner should not run")
        ),
    )

    result = executor.run(
        case_name="test_agent",
        case_data={
            "type": "agent_case",
            "intent": "click login button",
            "final": ["done"],
        },
    )

    assert result.final_reason == "compiled steps executed"
    assert events == [
        ("execute", "click", "#login", None),
        ("execute", "assert_title_contains", None, "Done"),
    ]
    step_executor_module.commit_pending_selector_cache()
    assert events[2][0] == "selector_cache"
    assert events[2][1]["target"] == "login button"
    assert events[2][1]["selector"] == "#login"
    step_executor_module.discard_pending_selector_cache("test cleanup")


def test_agent_case_compiled_execution_does_not_add_local_completion_assertions(
    monkeypatch, tmp_path: Path
):
    executed: list[dict] = []

    class FakePage:
        url = "about:blank"

    payload = {
        "cases": [{"name": "test_agent"}],
        "data": {
            "test_agent": {
                "mode": "smart",
                "steps": [
                    {"action": "click", "selector": "compiled_button"},
                    {"action": "assert_title_contains", "value": "Done"},
                ],
            }
        },
        "elements": {"compiled_button": "button:has-text('Run')"},
        "modules": {},
        "vars": {},
    }

    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.load_ai_config",
        lambda: {
            "runtime": {
                "agent_case_plan_cache_enabled": False,
                "ai_cache_sqlite_path": str(tmp_path / "ai_cache.sqlite3"),
                "candidate_limit": 5,
                "agent_completion_wait_seconds": 0,
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
        "src.ai_runtime.agent_case_executor._build_payload",
        lambda *args, **kwargs: (
            kwargs.get("cache_info", {}).update(
                {"cache_hit": False, "model_calls": 1}
            )
            or payload
        ),
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor._unmet_completion_criteria",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("local completion inference should not run")
        ),
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor._unmet_intent_action_requirements",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("intent completion inference should not run")
        ),
    )

    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="",
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
    monkeypatch.setattr(executor, "_maybe_open_start_url", lambda **kwargs: None)
    monkeypatch.setattr(
        executor, "_wait_for_compiled_completion_criteria", lambda **kwargs: []
    )
    monkeypatch.setattr(
        executor,
        "_decide_next_action",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("realtime planner should not run")
        ),
    )

    def fake_execute_step(step, *, spec):
        executed.append(dict(step))

    monkeypatch.setattr(executor, "_execute_step", fake_execute_step)

    result = executor.run(
        case_name="test_agent",
        case_data={
            "type": "agent_case",
            "intent": "click run and verify done",
            "final": ["Done"],
        },
    )

    assert result.final_reason == "compiled steps executed"
    assert [step["action"] for step in executed] == ["click", "assert_title_contains"]


def test_agent_case_replans_after_non_assertion_execution_failure(
    monkeypatch, tmp_path: Path
):
    executed: list[dict] = []
    decisions = iter(
        [
            AgentCaseDecision.model_validate(
                {"action": "click", "selector": "#missing", "reason": "first try"}
            ),
            AgentCaseDecision.model_validate(
                {"action": "click", "selector": "#ok", "reason": "retry"}
            ),
            AgentCaseDecision.model_validate({"action": "finish", "reason": "done"}),
        ]
    )

    class FakePage:
        url = "about:blank"

    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.load_ai_config",
        lambda: {
            "runtime": {
                "agent_case_plan_cache_enabled": False,
                "ai_cache_sqlite_path": str(tmp_path / "ai_cache.sqlite3"),
                "candidate_limit": 5,
                "agent_execution_retry_limit": 1,
            },
            "agent_policy": {
                "limits": {
                    "max_steps": 5,
                    "max_model_calls": 5,
                    "max_duration_seconds": 30,
                },
                "guardrails": {"require_checkpoints_or_final": True},
            },
            "generation": {"max_context_items": 5},
            "prompts": {"agent_case_version": "agent-case-test"},
            "llm": {"schema_version": "schema-test"},
        },
    )

    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="",
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
    monkeypatch.setattr(executor, "_maybe_open_start_url", lambda **kwargs: None)
    monkeypatch.setattr(executor, "_try_run_compiled_agent_case", lambda **kwargs: None)
    monkeypatch.setattr(executor, "_runtime_harness", lambda **kwargs: {})
    monkeypatch.setattr(executor, "_post_action_failure_boundary", lambda **kwargs: "")
    monkeypatch.setattr(
        executor,
        "_decide_next_action",
        lambda **kwargs: setattr(executor, "last_decision_used_model", True)
        or next(decisions),
    )

    def fake_execute_step(step, *, spec):
        executed.append(dict(step))
        if step.get("selector") == "#missing":
            raise AssertionError("selector not found")

    monkeypatch.setattr(executor, "_execute_step", fake_execute_step)

    result = executor.run(
        case_name="test_agent",
        case_data={
            "type": "agent_case",
            "intent": "click button",
            "final": ["done"],
        },
    )

    assert result.final_reason == "done"
    assert [step["selector"] for step in executed] == ["#missing", "#ok"]


def test_agent_case_assertion_execution_failure_is_not_retried(
    monkeypatch, tmp_path: Path
):
    decisions = iter(
        [
            AgentCaseDecision.model_validate(
                {
                    "action": "assert_title_contains",
                    "value": "Done",
                    "reason": "verify",
                }
            ),
            AgentCaseDecision.model_validate({"action": "finish", "reason": "done"}),
        ]
    )

    class FakePage:
        url = "about:blank"

    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.load_ai_config",
        lambda: {
            "runtime": {
                "agent_case_plan_cache_enabled": False,
                "ai_cache_sqlite_path": str(tmp_path / "ai_cache.sqlite3"),
                "candidate_limit": 5,
                "agent_execution_retry_limit": 3,
            },
            "agent_policy": {
                "limits": {
                    "max_steps": 5,
                    "max_model_calls": 5,
                    "max_duration_seconds": 30,
                },
                "guardrails": {"require_checkpoints_or_final": True},
            },
            "generation": {"max_context_items": 5},
            "prompts": {"agent_case_version": "agent-case-test"},
            "llm": {"schema_version": "schema-test"},
        },
    )

    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="",
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
    monkeypatch.setattr(executor, "_maybe_open_start_url", lambda **kwargs: None)
    monkeypatch.setattr(executor, "_try_run_compiled_agent_case", lambda **kwargs: None)
    monkeypatch.setattr(executor, "_runtime_harness", lambda **kwargs: {})
    monkeypatch.setattr(
        executor,
        "_decide_next_action",
        lambda **kwargs: setattr(executor, "last_decision_used_model", True)
        or next(decisions),
    )
    monkeypatch.setattr(
        executor,
        "_execute_step",
        lambda step, *, spec: (_ for _ in ()).throw(
            AssertionError("assert failed")
        ),
    )

    with pytest.raises(AssertionError, match="Agent断言执行失败"):
        executor.run(
            case_name="test_agent",
            case_data={
                "type": "agent_case",
                "intent": "verify title",
                "final": ["Done"],
            },
        )


def test_agent_spec_to_generation_spec_omits_empty_inputs():
    generation_spec = _agent_spec_to_generation_spec(
        case_name="test_agent",
        spec={
            "description": "",
            "intent": "输入商品ID SPU17799383833255667 后查询",
            "steps": ["输入商品ID SPU17799383833255667", "点击查询"],
            "inputs": {},
            "criteria": {
                "checkpoints": ["商品ID输入框已输入 SPU17799383833255667"],
                "final": ["查询结果可见"],
            },
        },
    )

    assert "inputs" not in generation_spec
    assert "inputs" not in generation_spec["cases"][0]
    assert generation_spec["cases"][0]["steps"][0] == (
        "输入商品ID SPU17799383833255667"
    )


def test_agent_case_skips_model_when_realtime_completion_is_already_satisfied(
    monkeypatch, tmp_path: Path
):
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
            raise AssertionError("satisfied realtime state should not call the model")

    class FakePage:
        url = "about:blank"

        def title(self):
            return "Done"

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
        lambda page, **kwargs: [
            {
                "index": 0,
                "tag": "h1",
                "selector": "h1",
                "text": "Order Complete",
                "visible": True,
                "enabled": True,
            }
        ],
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.load_ai_config",
        lambda: {
            "runtime": {
                "agent_case_plan_cache_enabled": False,
                "candidate_limit": 5,
                "agent_candidate_scan_limit": 5,
                "agent_candidate_limit": 5,
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

    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://example.test/done",
        elements={},
        modules={},
        variables={},
        test_cases=[],
        test_data={},
    )

    result = AgentCaseExecutor(
        page=FakePage(),
        ui_helper=object(),
        elements={},
        context=context,
    ).run(
        case_name="test_agent_done",
        case_data={
            "type": "agent_case",
            "intent": "verify completed page state",
            "final": ["page shows order complete"],
        },
    )

    assert result.model_calls == 0
    assert result.final_reason == "local completion criteria satisfied before model call"


def test_agent_case_guard_does_not_block_download_link_for_login_step():
    context = ProjectContext(
        project="demo",
        test_dir=Path("."),
        base_url="https://example.test",
        elements={},
        modules={},
        variables={},
        test_cases=[],
        test_data={},
    )

    class FakePage:
        url = "https://example.test/list"

    executor = AgentCaseExecutor(
        page=FakePage(),
        ui_helper=object(),
        elements={},
        context=context,
    )
    executor.current_dom_context = {
        "meta": {"url": "https://sso.example.test/login"},
        "forms": [],
        "interactive_elements": [
            {
                "id": "e1",
                "tag": "a",
                "role": "link",
                "text": "汽车人APP下载",
                "selector_candidates": ['a:has-text("汽车人APP下载")'],
            }
        ],
        "assertion_candidates": [],
    }

    executor._guard_decision(
        AgentCaseDecision(action="click", element_id="e1"),
        spec={
            "steps": ["点击登录按钮"],
            "intent": "点击登录按钮",
            "inputs": {},
            "criteria": {},
            "guardrails": {},
        },
        history=[],
    )


def test_agent_case_guard_does_not_block_download_link_for_mall_step():
    context = ProjectContext(
        project="demo",
        test_dir=Path("."),
        base_url="https://example.test",
        elements={},
        modules={},
        variables={},
        test_cases=[],
        test_data={},
    )

    class FakePage:
        url = "https://sso.example.test/login"

    executor = AgentCaseExecutor(
        page=FakePage(),
        ui_helper=object(),
        elements={},
        context=context,
    )
    executor.current_dom_context = {
        "meta": {"url": "https://sso.example.test/login"},
        "forms": [],
        "interactive_elements": [
            {
                "id": "e1",
                "tag": "a",
                "role": "link",
                "text": "汽车人APP下载",
                "selector_candidates": ['a:has-text("汽车人APP下载")'],
            }
        ],
        "assertion_candidates": [],
    }

    executor._guard_decision(
        AgentCaseDecision(action="click", element_id="e1"),
        spec={
            "steps": ["点击商城展开菜单"],
            "intent": "点击商城展开菜单",
            "inputs": {},
            "criteria": {},
            "guardrails": {},
        },
        history=[],
    )


def test_test_signature_uses_standard_page_fixtures():
    params = list(build_test_signature([]).parameters)

    assert params[:4] == ["page", "ui_helper", "get_test_name", "value"]


def test_config_honors_explicit_test_dir_override(tmp_path: Path):
    previous = os.environ.get("TEST_DIR")
    try:
        Config(
            project="demo",
            env="prod",
            base_url="https://example.test/",
            test_dir=str(tmp_path),
        )

        assert os.environ["TEST_DIR"] == str(tmp_path)
    finally:
        if previous is None:
            os.environ.pop("TEST_DIR", None)
        else:
            os.environ["TEST_DIR"] = previous
        Config(project="demo", env="prod", test_dir=previous or "test_data/demo")


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


def test_smart_resolver_healing_prefers_input_value_over_structural_selector():
    class FakeLocator:
        def __init__(self, selector):
            self.selector = selector

        @property
        def first(self):
            return self

        def count(self):
            return 1

        def nth(self, index):
            return self

        def is_visible(self):
            return True

        def wait_for(self, state, timeout):
            if self.selector not in {
                'input[type="submit"][value="登录"]',
                "input[type='submit']",
                'input[type="submit"]',
            }:
                raise TimeoutError("not found")

        def is_enabled(self):
            return True

        def evaluate(self, script):
            if "cssEscape" in script:
                return 'input[type="submit"][value="登录"]'
            if "tagName" in script:
                return {
                    "tag": "input",
                    "type": "submit",
                    "value": "登录",
                }
            return {}

    class FakePage:
        url = "https://example.test/login"

        def locator(self, selector):
            return FakeLocator(selector)

        def evaluate(self, script, limit):
            return [
                {
                    "index": 0,
                    "tag": "input",
                    "selector": "form > div:nth-of-type(2) > input",
                    "type": "submit",
                    "value": "登录",
                    "visible": True,
                    "enabled": True,
                }
            ]

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None

    resolved = resolver.resolve(
        action="click",
        selector="button[type='submit']",
        target="登录按钮",
        mode="smart",
        timeout=1000,
    )

    assert resolved.selector == 'input[type="submit"][value="登录"]'
    assert "nth-of-type" not in resolved.selector


def test_smart_resolver_can_disable_selector_registry_by_env(
    monkeypatch, tmp_path: Path
):
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


def test_smart_resolver_verifies_explicit_selector_before_registry_fallback(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.load_ai_config",
        lambda: {
            "selector_registry": {"enabled": False},
            "runtime": {"smart_selector_probe_timeout_ms": 1000},
            "native_observe": {"enabled": False},
            "vision": {"enabled": False},
        },
    )
    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.selector_matches_target",
        lambda page, selector, target, action: True,
    )
    verify_calls = []

    def fake_verify_selector(page, selector, *, action, timeout):
        verify_calls.append((selector, timeout))
        if selector == "input[name='password']":
            raise TimeoutError("yaml selector is stale")

    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.verify_selector",
        fake_verify_selector,
    )

    class FakePage:
        url = "https://example.test/login"

    registry = SelectorRegistry(tmp_path / "selectors.db")
    registry.save(
        project="demo",
        env="test",
        page_key="https://example.test/login",
        action="fill",
        target="input[name='password']",
        selector="#ipt_password",
        source="heuristic",
        confidence=0.9,
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = registry
    resolved = resolver.resolve(
        action="fill",
        target=None,
        selector="input[name='password']",
        mode="smart",
        timeout=10000,
    )

    assert resolved.selector == "#ipt_password"
    assert resolved.source == "registry"
    assert resolved.healed is True
    assert resolved.original_selector == "input[name='password']"
    assert verify_calls == [("input[name='password']", 1000), ("#ipt_password", 1000)]


def test_smart_resolver_target_only_can_use_verified_registry_first(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.load_ai_config",
        lambda: {
            "selector_registry": {"enabled": False},
            "runtime": {"smart_selector_probe_timeout_ms": 1000},
            "native_observe": {"enabled": False},
            "vision": {"enabled": False},
        },
    )
    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.selector_matches_target",
        lambda page, selector, target, action: True,
    )
    verify_calls = []

    def fake_verify_selector(page, selector, *, action, timeout):
        verify_calls.append((selector, timeout))

    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.verify_selector",
        fake_verify_selector,
    )

    class FakePage:
        url = "https://example.test/login"

    registry = SelectorRegistry(tmp_path / "selectors.db")
    registry.save(
        project="demo",
        env="test",
        page_key="https://example.test/login",
        action="click",
        target="login button",
        selector="#login",
        source="heuristic",
        confidence=0.9,
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = registry

    resolved = resolver.resolve(
        action="click",
        target="login button",
        selector=None,
        mode="smart",
        timeout=10000,
    )

    assert resolved.selector == "#login"
    assert resolved.source == "registry"
    assert resolved.healing_attempted is False
    assert verify_calls == [("#login", 1000)]


def test_smart_resolver_caps_generated_selector_probe_timeout(monkeypatch):
    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.load_ai_config",
        lambda: {
            "selector_registry": {"enabled": False},
            "runtime": {
                "ai_enabled": False,
                "smart_selector_probe_timeout_ms": 750,
            },
            "native_observe": {"enabled": False},
            "vision": {"enabled": False},
        },
    )
    verify_calls = []

    def fake_verify_selector(page, selector, *, action, timeout):
        verify_calls.append((selector, timeout))
        raise TimeoutError("not visible")

    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.verify_selector",
        fake_verify_selector,
    )

    class FakePage:
        url = "https://example.test/login"

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None

    with pytest.raises(ValueError):
        resolver.resolve(
            action="fill",
            target="password field",
            selector="#slow-password",
            mode="smart",
            timeout=10000,
        )

    assert verify_calls[0] == ("#slow-password", 750)


def test_smart_resolver_uses_ai_when_concrete_text_is_not_visible(monkeypatch):
    ai_called = False

    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.load_ai_config",
        lambda: {
            "selector_registry": {"enabled": False},
            "runtime": {
                "ai_enabled": True,
                "allow_ai_in_smart": True,
                "smart_selector_probe_timeout_ms": 10,
            },
            "native_observe": {"enabled": False},
            "vision": {"enabled": False},
        },
    )

    class EmptyLocator:
        def count(self):
            return 0

        def nth(self, index):
            return self

        def is_visible(self):
            return False

    class FakePage:
        url = "https://example.test/login"

        def locator(self, selector):
            return EmptyLocator()

        def get_by_text(self, text, exact=False):
            return EmptyLocator()

        def title(self):
            return "Login"

    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.verify_selector",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("not visible")),
    )
    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.heuristic_selectors",
        lambda target, action: [],
    )

    def fake_resolve_with_ai(self, *, action, target, timeout):
        nonlocal ai_called
        ai_called = True
        return ResolvedSelector(
            selector="#query",
            source="ai_selector",
            healed=True,
            ai_called=True,
            confidence=0.9,
        )

    monkeypatch.setattr(SmartResolver, "_resolve_with_ai", fake_resolve_with_ai)
    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None

    resolved = resolver.resolve(
        action="click",
        target="text=鍟嗗搧瀹℃牳",
        selector="text=鍟嗗搧瀹℃牳",
        mode="smart",
        timeout=1000,
    )

    assert resolved.selector == "#query"
    assert ai_called is True


def test_smart_resolver_fails_when_concrete_text_is_not_visible_without_ai(
    monkeypatch,
):
    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.load_ai_config",
        lambda: {
            "selector_registry": {"enabled": False},
            "runtime": {
                "ai_enabled": False,
                "allow_ai_in_smart": False,
                "smart_selector_probe_timeout_ms": 10,
            },
            "native_observe": {"enabled": False},
            "vision": {"enabled": False},
        },
    )

    class EmptyLocator:
        def count(self):
            return 0

        def nth(self, index):
            return self

        def is_visible(self):
            return False

    class FakePage:
        url = "https://example.test/login"

        def locator(self, selector):
            return EmptyLocator()

        def get_by_text(self, text, exact=False):
            return EmptyLocator()

        def title(self):
            return "Login"

    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.verify_selector",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("not visible")),
    )
    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.heuristic_selectors",
        lambda target, action: [],
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None

    with pytest.raises(ValueError, match="target text is not visible"):
        resolver.resolve(
            action="click",
            target="text=鍟嗗搧瀹℃牳",
            selector="text=鍟嗗搧瀹℃牳",
            mode="smart",
            timeout=1000,
        )


def test_selector_matches_target_rejects_login_target_on_unrelated_link():
    class FakeLocator:
        @property
        def first(self):
            return self

        def evaluate(self, script):
            return {
                "tag": "a",
                "text": "Download app",
                "href": "https://example.test/download",
            }

    class FakePage:
        def locator(self, selector):
            return FakeLocator()

    assert (
        selector_matches_target(FakePage(), "a.download-link", "login button", "click")
        is False
    )


def test_selector_matches_target_rejects_unrelated_text_fallback():
    class FakeLocator:
        @property
        def first(self):
            return self

        def evaluate(self, script):
            return {
                "tag": "a",
                "text": "姹借溅浜篈PP涓嬭浇",
                "title": "姹借溅浜篈PP涓嬭浇",
                "role": "link",
            }

    class FakePage:
        def locator(self, selector):
            return FakeLocator()

    assert (
        selector_matches_target(
            FakePage(), 'a:has-text("姹借溅浜篈PP涓嬭浇")', "text=鍟嗗煄", "click"
        )
        is False
    )
    assert (
        selector_matches_target(
            FakePage(), "#username", "li.menu-item:has-text('鍟嗗煄')", "click"
        )
        is False
    )


def test_selector_matches_target_rejects_unrelated_fill_field():
    class FakeLocator:
        @property
        def first(self):
            return self

        def evaluate(self, script):
            return {
                "tag": "input",
                "id": "product-audit-search_brandid",
                "name": "brandId",
                "placeholder": "",
                "label": "鍝佺墝",
            }

    class FakePage:
        def locator(self, selector):
            return FakeLocator()

    assert (
        selector_matches_target(
            FakePage(),
            "#product-audit-search_brandid",
            "text=杈撳叆鍟嗗搧id/鍟嗗搧鍚嶇О",
            "fill",
        )
        is False
    )


def test_selector_matches_target_accepts_exact_fill_placeholder():
    class FakeLocator:
        @property
        def first(self):
            return self

        def evaluate(self, script):
            return {
                "tag": "input",
                "id": "product-audit-search_productid",
                "placeholder": "杈撳叆鍟嗗搧id/鍟嗗搧鍚嶇О",
            }

    class FakePage:
        def locator(self, selector):
            return FakeLocator()

    assert (
        selector_matches_target(
            FakePage(),
            "#product-audit-search_productid",
            "text=杈撳叆鍟嗗搧id/鍟嗗搧鍚嶇О",
            "fill",
        )
        is True
    )


def test_selector_matches_target_accepts_cjk_display_spacing():
    class FakeLocator:
        @property
        def first(self):
            return self

        def evaluate(self, script):
            return {"tag": "button", "text": "Sea rch"}

    class FakePage:
        def locator(self, selector):
            return FakeLocator()

    assert (
        selector_matches_target(
            FakePage(), 'button:has-text("Sea rch")', "text=Search", "click"
        )
        is True
    )


def test_selector_matches_target_accepts_precise_action_synonym():
    class FakeLocator:
        def __init__(self, selector):
            self.selector = selector

        @property
        def first(self):
            return self

        def evaluate(self, script):
            text = "\u67e5\u8be2" if "\u67e5\u8be2" in self.selector else "\u53d6\u6d88"
            return {"tag": "button", "text": text, "role": "button"}

    class FakePage:
        def locator(self, selector):
            return FakeLocator(selector)

    assert (
        selector_matches_target(
            FakePage(),
            'button:has-text("\u67e5\u8be2")',
            "text=\u641c\u7d22\u6309\u94ae",
            "click",
        )
        is True
    )
    assert (
        selector_matches_target(
            FakePage(),
            'button:has-text("\u53d6\u6d88")',
            "text=\u641c\u7d22\u6309\u94ae",
            "click",
        )
        is False
    )


def test_selector_matches_target_rejects_logout_for_query_even_in_ai_mode():
    class FakeLocator:
        @property
        def first(self):
            return self

        def evaluate(self, script):
            return {"tag": "a", "text": "Logout", "role": "link"}

    class FakePage:
        def locator(self, selector):
            return FakeLocator()

    assert (
        selector_matches_target(
            FakePage(),
            'a:has-text("Logout")',
            "text=Search",
            "click",
            strict_text_match=False,
        )
        is False
    )


def test_selector_matches_target_relaxed_ai_accepts_action_object_text():
    class FakeLocator:
        def __init__(self, selector):
            self.selector = selector

        @property
        def first(self):
            return self

        def evaluate(self, script):
            text = "\u65e5\u5fd7" if "\u65e5\u5fd7" in self.selector else "\u5546\u54c1\u5ba1\u6838"
            return {"tag": "a", "text": text, "role": "link"}

    class FakePage:
        def locator(self, selector):
            return FakeLocator(selector)

    assert (
        selector_matches_target(
            FakePage(),
            'a:has-text("\u65e5\u5fd7")',
            "text=\u67e5\u770b\u65e5\u5fd7",
            "click",
            strict_text_match=False,
        )
        is True
    )
    assert (
        selector_matches_target(
            FakePage(),
            'a:has-text("\u5546\u54c1\u5ba1\u6838")',
            "text=\u67e5\u770b\u65e5\u5fd7",
            "click",
            strict_text_match=False,
        )
        is False
    )


def test_smart_resolver_expands_ai_selector_scan_when_target_term_missing():
    class FakePage:
        url = "https://example.test/list"

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None
    resolver.llm_selector_candidate_scan_limit = 2
    resolver.llm_selector_candidate_expand_limit = 4
    calls = []

    def fake_collect_candidates(*, limit, respect_config_max=True):
        calls.append((limit, respect_config_max))
        base = [
            {
                "index": 1,
                "tag": "a",
                "text": "\u5546\u54c1\u5ba1\u6838",
                "selector": 'a:has-text("\u5546\u54c1\u5ba1\u6838")',
            },
            {
                "index": 2,
                "tag": "button",
                "text": "\u67e5 \u8be2",
                "selector": 'button:has-text("\u67e5 \u8be2")',
            },
        ]
        if limit <= 2:
            return base
        return base + [
            {
                "index": 3,
                "tag": "a",
                "text": "\u65e5\u5fd7",
                "selector": 'a:has-text("\u65e5\u5fd7")',
            }
        ]

    resolver._collect_candidates = fake_collect_candidates

    candidates = resolver._collect_ai_locator_candidates(
        action="click",
        target="text=\u67e5\u770b\u65e5\u5fd7",
    )

    assert calls == [(2, True), (4, False)]
    assert any(candidate.get("text") == "\u65e5\u5fd7" for candidate in candidates)


def test_smart_resolver_rejects_unresolved_internal_element_id_target():
    class FakePage:
        url = "https://example.test/"

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None

    with pytest.raises(
        ValueError, match="未解析的内部element_id不能直接用于语义定位: e2"
    ):
        resolver.resolve(
            action="fill",
            target="e2",
            selector=None,
            mode="smart",
            timeout=1000,
        )


def test_smart_resolver_ai_path_rejects_direct_selector(monkeypatch):
    class FakePage:
        url = "https://example.test/form"

        def evaluate(self, script, limit):
            return [
                {
                    "index": 0,
                    "tag": "button",
                    "selector": "#submit",
                    "text": "Submit",
                    "visible": True,
                    "enabled": True,
                }
            ]

    class DirectSelectorProvider:
        settings = type("Settings", (), {"model": "test-model"})()

        def complete_model(self, messages, response_model, **kwargs):
            return response_model.model_validate(
                {"selector": "#submit", "selector_type": "css", "confidence": 0.9}
            )

    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.ChatCompletionProvider",
        DirectSelectorProvider,
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None

    with pytest.raises(ValueError, match="必须返回候选element_id"):
        resolver._resolve_with_ai(action="click", target="鎻愪氦鎸夐挳", timeout=1000)


def test_smart_resolver_ai_path_uses_candidate_element_id(monkeypatch):
    class FakeLocator:
        @property
        def first(self):
            return self

        def count(self):
            return 1

        def nth(self, index):
            return self

        def is_visible(self):
            return True

        def wait_for(self, state, timeout):
            return None

        def is_enabled(self):
            return True

    class FakePage:
        url = "https://example.test/form"

        def locator(self, selector):
            assert selector == 'button[data-test="submit"]'
            return FakeLocator()

        def evaluate(self, script, limit):
            return [
                {
                    "index": 0,
                    "tag": "button",
                    "selector": "#submit",
                    "data_test": "submit",
                    "text": "Submit",
                    "visible": True,
                    "enabled": True,
                }
            ]

    class ElementIdProvider:
        settings = type("Settings", (), {"model": "test-model"})()

        def complete_model(self, messages, response_model, **kwargs):
            return response_model.model_validate(
                {"element_id": "e0", "confidence": 0.91}
            )

    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.ChatCompletionProvider",
        ElementIdProvider,
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None

    resolved = resolver._resolve_with_ai(
        action="click",
        target="鎻愪氦鎸夐挳",
        timeout=1000,
    )

    assert resolved.selector == 'button[data-test="submit"]'
    assert resolved.source == "ai_selector"
    assert resolved.ai_called is True


def test_smart_resolver_uses_role_text_candidate_from_structural_menu_source(
    monkeypatch,
):
    captured: dict[str, Any] = {}

    class FakeLocator:
        @property
        def first(self):
            return self

        def count(self):
            return 1

        def nth(self, index):
            return self

        def is_visible(self):
            return True

        def wait_for(self, state, timeout):
            return None

        def is_enabled(self):
            return True

    class FakePage:
        url = "https://example.test/home"

        def locator(self, selector):
            assert selector == 'div[role="menuitem"]:has-text("Mall")'
            return FakeLocator()

        def evaluate(self, script, limit):
            return [
                {
                    "index": 19,
                    "tag": "div",
                    "role": "menuitem",
                    "selector": "li:nth-of-type(19) > div",
                    "text": "Mall",
                    "visible": True,
                    "enabled": True,
                }
            ]

        def title(self):
            return "Home"

    class ElementIdProvider:
        settings = type("Settings", (), {"model": "test-model"})()

        def complete_model(self, messages, response_model, **kwargs):
            captured["system"] = messages[0]["content"]
            return response_model.model_validate(
                {"element_id": "e19", "confidence": 0.91}
            )

    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.ChatCompletionProvider",
        ElementIdProvider,
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None

    resolved = resolver._resolve_with_ai(action="click", target="text=Mall", timeout=1000)

    assert resolved.selector == 'div[role="menuitem"]:has-text("Mall")'
    assert "高质量selector候选" in captured["system"]
    assert "nth-of-type" in captured["system"]


def test_smart_resolver_rejects_low_quality_ai_selector_and_prompts_for_quality(
    monkeypatch,
):
    captured: dict[str, Any] = {}

    class FakePage:
        url = "https://example.test/home"

        def locator(self, selector):
            raise AssertionError(f"low-quality selector should not be verified: {selector}")

        def evaluate(self, script, limit):
            return [
                    {
                        "index": 19,
                        "tag": "div",
                        "role": "menuitem",
                        "selector": "li:nth-of-type(19) > div",
                        "visible": True,
                        "enabled": True,
                    }
            ]

        def title(self):
            return "Home"

    class ElementIdProvider:
        settings = type("Settings", (), {"model": "test-model"})()

        def complete_model(self, messages, response_model, **kwargs):
            captured["system"] = messages[0]["content"]
            return response_model.model_validate(
                {"element_id": "e19", "confidence": 0.91}
            )

    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.ChatCompletionProvider",
        ElementIdProvider,
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None

    with pytest.raises(ValueError, match="no selector candidates"):
        resolver._resolve_with_ai(action="click", target="text=Mall", timeout=1000)

    assert "高质量selector候选" in captured["system"]
    assert "nth-of-type" in captured["system"]


def test_smart_resolver_does_not_cache_low_quality_selector(tmp_path: Path):
    class FakePage:
        url = "https://example.test/home"

    registry = SelectorRegistry(tmp_path / "selectors.db")
    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = registry

    resolver.record_verified_selector(
        action="click",
        target="text=Mall",
        selector="li:nth-of-type(19) > div",
        source="ai_selector",
        confidence=0.9,
        page_key="https://example.test/home",
    )

    assert (
        registry.find(
            project="demo",
            env="test",
            page_key="https://example.test/home",
            action="click",
            target="text=Mall",
        )
        is None
    )


def test_selector_quality_rejects_structural_position_chain():
    assert is_high_quality_selector("li:nth-of-type(19) > div") is False
    assert is_high_quality_selector('div[role="menuitem"]:has-text("Mall")') is True
    assert is_high_quality_selector('button:has-text("Mall")') is True
    assert is_high_quality_selector('[data-testid="mall-menu"]') is True


def test_dom_context_prefers_role_text_selector_over_structural_path():
    context = build_dom_context(
        [
            {
                "index": 19,
                "tag": "div",
                "role": "menuitem",
                "selector": "li:nth-of-type(19) > div",
                "text": "Mall",
                "visible": True,
                "enabled": True,
            }
        ],
        url="https://example.test/home",
        title="Home",
        hints=["open Mall"],
    )

    element = context["interactive_elements"][0]
    assert element["selector_candidates"][0] == 'div[role="menuitem"]:has-text("Mall")'
    assert "li:nth-of-type(19) > div" not in element["selector_candidates"]


def test_smart_resolver_collect_candidates_uses_native_observe_config(monkeypatch):
    captured: dict[str, Any] = {}

    class FakePage:
        url = "https://example.test"

    def fake_collect_candidates(page, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.load_ai_config",
        lambda: {
            "native_observe": {
                "max_candidates": 7,
                "include_open_shadow_dom": False,
                "ignore_selectors": [".noise"],
            }
        },
    )
    monkeypatch.setattr(
        "src.ai_runtime.smart_resolver.collect_candidates",
        fake_collect_candidates,
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver._collect_candidates(limit=20)

    assert captured == {
        "limit": 7,
        "ignore_selectors": (".noise",),
        "include_open_shadow_dom": False,
    }


def test_ai_step_fast_path_simplifies_header_position_instruction(monkeypatch):
    class FakePage:
        url = "https://example.test/dashboard"

    def fake_semantic_selectors(page, target, action, limit, **kwargs):
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


def test_step_executor_records_healed_selector_without_writing_elements_yaml(
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
    cache_events: list[dict[str, Any]] = []

    class FakeResolver:
        def resolve(self, **kwargs):
            assert kwargs["target"] == "#old-login"
            return ResolvedSelector(
                selector="#login-button",
                source="heuristic",
                healed=True,
                healing_attempted=True,
                original_selector="#old-login",
                original_error="not found",
                confidence=0.95,
                cache_action="click",
                cache_target=kwargs["target"],
                cache_page_key="https://example.test/login",
            )

        def record_verified_selector(self, **kwargs):
            cache_events.append(kwargs)

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
    assert calls == [("click", "#login-button", None)]
    assert executor.elements["login_button"] == "#old-login"

    step_executor_module.commit_pending_selector_cache()
    for thread in executor._healing_threads:
        thread.join(timeout=3)

    assert executor.elements["login_button"] == "#old-login"
    assert "#login-button" not in elements_file.read_text(encoding="utf-8")
    assert cache_events[0]["selector"] == "#login-button"
    assert cache_events[0]["target"] == "#old-login"


def test_step_executor_does_not_call_element_store_when_healed_selector_verified(
    monkeypatch,
):
    calls: list[tuple[str, str, str | None]] = []

    class FakeResolver:
        def resolve(self, **kwargs):
            assert kwargs["target"] == "#old-login"
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
            raise AssertionError("runtime should not auto-write elements yaml")

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
    step_executor_module.commit_pending_selector_cache()
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

    with pytest.raises(ValueError):
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


def test_chat_completion_provider_extracts_valid_json_from_reasoning_content(
    monkeypatch, tmp_path: Path
):
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
                            "content": "",
                            "reasoning_content": (
                                'ignore {"foo": 1} final '
                                '{"cases":[{"name":"generated"}],'
                                '"data":{"generated":{"mode":"smart","steps":[]}},'
                                '"elements":{},"modules":{},"vars":{}}'
                            ),
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
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
            model="deepseek-test",
        )
    )

    payload = provider.complete_model(
        [{"role": "user", "content": "generate"}],
        GeneratedCasePayload,
    )

    assert payload.cases[0].name == "generated"
    assert payload.data["generated"].mode == "smart"


def test_chat_completion_provider_retries_transient_http_error(
    monkeypatch, tmp_path: Path
):
    tracker = TokenUsageTracker(tmp_path)
    tracker.start_run(run_kind="pytest", metadata={"project": "demo"})
    calls = {"count": 0}

    class FakeResponse:
        def __init__(self, status_code: int):
            self.status_code = status_code
            self.text = '{"error":"upstream"}' if status_code >= 400 else ""

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(response=self)

        def json(self):
            return {
                "choices": [{"message": {"content": '{"ok":true}'}}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        return FakeResponse(500 if calls["count"] == 1 else 200)

    monkeypatch.setattr("src.ai_runtime.provider.requests.post", fake_post)
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

    assert provider.complete([{"role": "user", "content": "hello"}]) == '{"ok":true}'
    assert calls["count"] == 2


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


def test_token_usage_exposes_active_run_kind(tmp_path: Path):
    tracker = TokenUsageTracker(tmp_path / "token_usage")

    assert tracker.active_run_kind is None
    tracker.start_run(run_kind="generate_case")
    assert tracker.active_run_kind == "generate_case"
    tracker.finish_run(status="passed")
    assert tracker.active_run_kind is None


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

    assert (
        AiStepDecision.model_validate(
            {"action": "reject", "reason": "multiple actions"}
        ).action
        == "reject"
    )
    assert (
        AiStepDecision.model_validate(
            {"status": "need_more_context", "reason": "not enough candidates"}
        ).status
        == "need_more_context"
    )

    with pytest.raises(ValueError):
        AiStepDecision.model_validate({"action": "reject"})

    assert (
        AgentCaseDecision.model_validate(
            {"action": "click", "target": "Login button"}
        ).action
        == "click"
    )
    assert (
        AgentCaseDecision.model_validate(
            {
                "status": "ok",
                "action": "click",
                "element_id": "e12",
                "reason": "click target button",
                "expected": "page state changes",
                "confidence": 0.9,
            }
        ).element_id
        == "e12"
    )
    assert (
        AgentCaseDecision.model_validate(
            {"status": "need_more_context", "reason": "鍊欓€変腑娌℃湁鐩爣鍏冪礌"}
        ).status
        == "need_more_context"
    )
    assert (
        AgentCaseDecision.model_validate(
            {"action": "done", "reason": "success criteria satisfied"}
        ).action
        == "done"
    )
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

    assert payload.data["test_generated"].mode == "smart"
    assert payload.elements == {}


def test_explicit_generation_spec_defaults_to_smart_without_description():
    payload = _payload_from_explicit_spec(
        {
            "cases": [
                {
                    "name": "generated",
                    "steps": [
                        {"action": "click", "target": "鎻愪氦鎸夐挳"},
                        {
                            "action": "assert_url_contains",
                            "value": "/done",
                        },
                    ],
                }
            ]
        }
    )

    assert payload["cases"] == [{"name": "generated"}]
    assert "description" not in payload["data"]["generated"]
    assert payload["data"]["generated"]["mode"] == "smart"


def test_generation_spec_scope_matches_project():
    _validate_spec_project_scope(
        project="demo",
        spec_path=Path("test_data/demo/generation/saucedemo_ai.yaml"),
        spec={},
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
    spec_file.write_text("cases: []\n", encoding="utf-8")

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
                    "open search page",
                    "click search input",
                    "fill search keyword",
                    "click search button",
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
                    "description": "open https://description.example/ then login",
                    "steps": ["open https://steps.example/ login page"],
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
                    "description": "open https://description.example/ then login",
                    "steps": ["login with standard user"],
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
        {"cases": [{"name": "test_flow", "description": "鏍囧噯鐢ㄦ埛鐧诲綍"}]},
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
        {"cases": [{"name": "test_flow", "description": "鏍囧噯鐢ㄦ埛鐧诲綍"}]},
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


def test_generation_build_payload_does_not_reuse_hidden_generation_cache(
    monkeypatch, tmp_path: Path
):
    calls = {"count": 0}

    monkeypatch.setattr(
        "src.ai_generation.case_generator.load_ai_config",
        lambda: {
            "generation": {"max_context_items": 5},
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

    monkeypatch.setattr(
        "src.ai_generation.case_generator.ChatCompletionProvider", FakeProvider
    )

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
    _build_payload(
        context,
        spec,
        env="prod",
        output_name="generated",
        use_ai=True,
    )
    _build_payload(
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

    assert calls["count"] == 2


def test_generate_case_verifies_candidate_before_formal_persist(
    monkeypatch, tmp_path: Path
):
    events: list[str] = []
    test_dir = tmp_path / "test_data" / "demo"
    spec_file = test_dir / "generation" / "spec.yaml"
    spec_file.parent.mkdir(parents=True)
    spec_file.write_text(
        "cases:\n  - name: generated\n    intent: check page\n",
        encoding="utf-8",
    )
    for dirname in ("cases", "data", "elements", "modules", "vars"):
        (test_dir / dirname).mkdir(parents=True)
    context = ProjectContext(
        project="demo",
        test_dir=test_dir,
        base_url="https://example.test/",
        elements={},
        modules={},
        variables={},
        test_cases=[],
        test_data={},
    )
    payload = {
        "cases": [{"name": "test_generated"}],
        "data": {
            "test_generated": {
                "description": "generated",
                "mode": "strict",
                "steps": [
                    {"action": "goto", "value": "https://example.test/"},
                    {"action": "assert_url_contains", "value": "example.test"},
                ],
            }
        },
        "elements": {},
        "modules": {},
        "vars": {},
    }

    monkeypatch.setattr(
        "src.ai_generation.case_generator.load_project_context",
        lambda project, env="prod": context,
    )
    monkeypatch.setattr(
        "src.ai_generation.case_generator.load_ai_config",
        lambda: {
            "generation": {
                "verify_after_generate": True,
                "runtime_repair_attempts": 1,
            }
        },
    )
    monkeypatch.setattr(
        "src.ai_generation.case_generator._build_payload",
        lambda *args, **kwargs: payload,
    )
    monkeypatch.setattr(
        "src.ai_generation.case_generator._normalize_validate_payload",
        lambda **kwargs: (payload, []),
    )
    monkeypatch.setattr(
        "src.ai_generation.case_generator._write_and_verify_candidate",
        lambda **kwargs: events.append("candidate_verify"),
    )

    def fake_verify_generated_case(*, stage="生成", **kwargs):
        events.append(f"verify:{stage}")

    def fake_write_payload(result, *, overwrite, verify=None, post_verify=None):
        events.append("formal_write")
        if post_verify:
            post_verify()

    monkeypatch.setattr(
        "src.ai_generation.case_generator._verify_generated_case",
        fake_verify_generated_case,
    )
    monkeypatch.setattr(
        "src.ai_generation.case_generator._write_payload",
        fake_write_payload,
    )
    generate_case_files(
        project="demo",
        spec_path=spec_file,
        overwrite=True,
    )

    assert events == [
        "candidate_verify",
        "formal_write",
        "verify:正式存储后",
    ]


def test_generate_case_does_not_persist_when_candidate_verify_fails(
    monkeypatch, tmp_path: Path
):
    events: list[str] = []
    test_dir = tmp_path / "test_data" / "demo"
    spec_file = test_dir / "generation" / "spec.yaml"
    spec_file.parent.mkdir(parents=True)
    spec_file.write_text(
        "cases:\n  - name: generated\n    intent: check page\n",
        encoding="utf-8",
    )
    for dirname in ("cases", "data", "elements", "modules", "vars"):
        (test_dir / dirname).mkdir(parents=True)
    context = ProjectContext(
        project="demo",
        test_dir=test_dir,
        base_url="https://example.test/",
        elements={},
        modules={},
        variables={},
        test_cases=[],
        test_data={},
    )
    payload = {
        "cases": [{"name": "test_generated"}],
        "data": {
            "test_generated": {
                "steps": [{"action": "assert_url_contains", "value": "example.test"}]
            }
        },
        "elements": {},
        "modules": {},
        "vars": {},
    }

    monkeypatch.setattr(
        "src.ai_generation.case_generator.load_project_context",
        lambda project, env="prod": context,
    )
    monkeypatch.setattr(
        "src.ai_generation.case_generator.load_ai_config",
        lambda: {"generation": {"verify_after_generate": True}},
    )
    monkeypatch.setattr(
        "src.ai_generation.case_generator._build_payload",
        lambda *args, **kwargs: payload,
    )
    monkeypatch.setattr(
        "src.ai_generation.case_generator._normalize_validate_payload",
        lambda **kwargs: (payload, []),
    )

    def fail_candidate(**kwargs):
        events.append("candidate_verify")
        raise AssertionError("candidate failed")

    def fail_if_written(*args, **kwargs):
        raise AssertionError("formal write must not happen")

    monkeypatch.setattr(
        "src.ai_generation.case_generator._write_and_verify_candidate",
        fail_candidate,
    )
    monkeypatch.setattr(
        "src.ai_generation.case_generator._write_payload",
        fail_if_written,
    )
    monkeypatch.setattr(
        "src.ai_generation.case_generator._repair_payload_with_ai",
        lambda **kwargs: events.append("repair") or payload,
    )

    with pytest.raises(AssertionError, match="运行验证修复失败"):
        generate_case_files(
            project="demo",
            spec_path=spec_file,
            overwrite=True,
        )

    assert events == ["candidate_verify", "repair", "candidate_verify"]


def test_write_payload_rolls_back_when_post_verify_fails(tmp_path: Path):
    result = {
        "payload": {
            "cases": [{"name": "test_generated"}],
            "data": {"test_generated": {"steps": []}},
        },
        "case_file": tmp_path / "cases" / "generated.yaml",
        "data_file": tmp_path / "data" / "generated.yaml",
    }

    def fail_post_verify():
        raise RuntimeError("browser verification failed")

    with pytest.raises(RuntimeError, match="browser verification failed"):
        _write_payload(
            result,
            overwrite=True,
            post_verify=fail_post_verify,
        )

    assert not result["case_file"].exists()
    assert not result["data_file"].exists()


def test_generation_requires_effective_information_assertion(tmp_path: Path):
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
    payload = {
        "cases": [{"name": "test_generated"}],
        "data": {
        "test_generated": {"steps": [{"action": "click", "target": "提交按钮"}]}
        },
    }

    with pytest.raises(ValueError, match="缺少有效信息断言"):
        _assert_effective_verification_payload(context, payload)


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


def test_generation_harness_scopes_generated_element_key_collisions_by_page_context(
    tmp_path: Path,
):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"search_input": "#existing-search"},
            modules={},
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec={"mode": "smart"},
        output_name="crm/order/create",
    )

    payload = harness.normalize(
        {
            "cases": [{"name": "test_generated"}],
            "data": {
                "test_generated": {
                    "mode": "smart",
                    "steps": [
                        {"action": "assert_title_contains", "value": "Product Audit"},
                        {"action": "fill", "selector": "search_input", "value": "x"},
                        {"action": "click", "target": "search_input"},
                    ],
                }
            },
            "elements": {"search_input": "#generated-search"},
        }
    )

    assert payload["elements"] == {"search_input_Product_Audit": "#generated-search"}
    steps = payload["data"]["test_generated"]["steps"]
    assert steps[1]["selector"] == "search_input_Product_Audit"
    assert steps[2] == {
        "action": "click",
        "selector": "search_input_Product_Audit",
    }


def test_generation_result_paths_preserve_generation_relative_directory(
    tmp_path: Path,
):
    context = ProjectContext(
        project="demo",
        test_dir=tmp_path / "test_data" / "demo",
        base_url="",
        elements={},
        modules={},
        variables={},
        test_cases=[],
        test_data={},
    )
    spec_file = context.test_dir / "generation" / "crm" / "order" / "create.yaml"
    payload = {
        "cases": [{"name": "test_generated"}],
        "data": {"test_generated": {"steps": []}},
        "elements": {"confirm_button": "#confirm"},
        "modules": {"confirm_flow": [{"action": "click", "selector": "confirm_button"}]},
        "vars": {"username": "qa"},
    }

    output_name = _default_output_name(spec_file, context=context)
    result = _result_paths(context, payload, output_name=output_name)

    assert output_name == "crm/order/create"
    assert result["case_file"] == context.test_dir / "cases" / "crm/order/create.yaml"
    assert result["data_file"] == context.test_dir / "data" / "crm/order/create.yaml"
    assert (
        result["elements_file"]
        == context.test_dir / "elements" / "crm/order/create.yaml"
    )
    assert (
        result["modules_file"]
        == context.test_dir / "modules" / "crm/order/create.yaml"
    )
    assert result["vars_file"] == context.test_dir / "vars" / "crm/order/create.yaml"


def test_generation_write_payload_merges_existing_vars_file(tmp_path: Path):
    vars_file = tmp_path / "vars" / "generated.yaml"
    vars_file.parent.mkdir(parents=True)
    vars_file.write_text("username: qa\npassword: secret\n", encoding="utf-8")
    result = {
        "payload": {
            "cases": [{"name": "test_generated"}],
            "data": {"test_generated": {"steps": []}},
            "vars": {"product_id": "SPU123"},
        },
        "case_file": tmp_path / "cases" / "generated.yaml",
        "data_file": tmp_path / "data" / "generated.yaml",
        "vars_file": vars_file,
    }

    _write_payload(result, overwrite=True)

    written = YAML(typ="safe").load(vars_file.read_text(encoding="utf-8"))
    assert written == {
        "username": "qa",
        "password": "secret",
        "product_id": "SPU123",
    }


def test_runtime_repair_payload_includes_referenced_context_modules(tmp_path: Path):
    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="",
        elements={},
        modules={
            "login": [
                {"action": "click", "selector": "old_login"},
                {"action": "click", "selector": "old_menu"},
            ]
        },
        variables={},
        test_cases=[],
        test_data={},
    )
    payload = {
        "cases": [{"name": "test_generated"}],
        "data": {
            "test_generated": {
                "steps": [{"use_module": "login"}],
            }
        },
        "elements": {},
        "modules": {},
        "vars": {},
    }

    enriched = _payload_with_referenced_context_modules(payload, context=context)

    assert enriched["modules"] == {
        "login": [
            {"action": "click", "selector": "old_login"},
            {"action": "click", "selector": "old_menu"},
        ]
    }
    assert payload["modules"] == {}


def test_runtime_repair_keeps_context_module_when_model_truncates_steps(
    tmp_path: Path,
):
    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="",
        elements={},
        modules={
            "login": [
                {"action": "click", "selector": "old_login"},
                {"action": "click", "selector": "old_menu"},
            ]
        },
        variables={},
        test_cases=[],
        test_data={},
    )
    previous_payload = {
        "cases": [{"name": "test_generated"}],
        "data": {
            "test_generated": {
                "steps": [{"use_module": "login"}],
            }
        },
        "elements": {},
        "modules": {},
        "vars": {},
    }
    repaired_payload = {
        "cases": [{"name": "test_generated"}],
        "data": {
            "test_generated": {
                "steps": [{"use_module": "login"}],
            }
        },
        "elements": {},
        "modules": {"login": [{"action": "click", "selector": "old_login"}]},
        "vars": {},
    }

    preserved = case_generator_module._payload_preserving_referenced_context_modules(
        repaired_payload,
        previous_payload=previous_payload,
        context=context,
    )

    assert preserved["modules"]["login"] == context.modules["login"]


def test_generation_harness_separates_same_named_elements_on_different_pages(
    tmp_path: Path,
):
    page_a_harness = GenerationHarness(
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
        output_name="crm/order/create",
    )
    page_a_payload = page_a_harness.normalize(
        {
            "cases": [{"name": "test_create"}],
            "data": {
                "test_create": {
                    "mode": "smart",
                    "steps": [
                        {"action": "click", "selector": "confirm_button"},
                        {"action": "assert_title_contains", "value": "created"},
                    ],
                }
            },
            "elements": {
                "confirm_button": {
                    "selector": "#create-confirm",
                    "page": "Create Order",
                }
            },
        }
    )

    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements=page_a_payload["elements"],
            modules={},
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec={"mode": "smart"},
        output_name="crm/order/cancel",
    )

    payload = harness.normalize(
        {
            "cases": [{"name": "test_cancel"}],
            "data": {
                "test_cancel": {
                    "mode": "smart",
                    "steps": [
                        {"action": "click", "selector": "confirm_button"},
                        {"action": "assert_title_contains", "value": "cancelled"},
                    ],
                }
            },
            "elements": {
                "confirm_button": {
                    "selector": "#cancel-confirm",
                    "page": "Cancel Order",
                }
            },
        }
    )

    assert page_a_payload["elements"] == {"confirm_button": "#create-confirm"}
    assert payload["elements"] == {"confirm_button_Cancel_Order": "#cancel-confirm"}
    steps = payload["data"]["test_cancel"]["steps"]
    assert steps[0]["selector"] == "confirm_button_Cancel_Order"


def test_generation_harness_accepts_title_contains_without_selector(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="https://example.test/",
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
                    "mode": "smart",
                    "steps": [
                        {"action": "goto", "value": "https://example.test"},
                        {"action": "assert_title_contains", "value": "Products"},
                    ],
                }
            },
        }
    )

    assert harness.validate(payload) == []


def test_generation_harness_rewrites_element_key_targets(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="https://example.test/",
            elements={"known_search_input": "#search"},
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
                            "target": "known_search_input",
                            "value": "x",
                        },
                        {"action": "click", "target": "generated_button"},
                        {"action": "assert_title_contains", "value": "x"},
                    ],
                }
            },
            "elements": {
                "generated_button": {"target": "椤甸潰椤堕儴鎼滅储鎸夐挳"},
                "ignored_without_selector": {"description": "not an element asset"},
            },
        }
    )

    steps = payload["data"]["test_generated"]["steps"]
    assert steps[0] == {
        "action": "fill",
        "value": "x",
        "selector": "known_search_input",
    }
    assert steps[1] == {"action": "click", "target": "椤甸潰椤堕儴鎼滅储鎸夐挳"}
    assert payload["elements"] == {}
    assert harness.validate(payload) == []


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
                        {"action": "click", "target": "鐧诲綍鎸夐挳"},
                        {"action": "assert_visible", "target": "棣栭〉"},
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

    with pytest.raises(ValueError):
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


def test_generation_harness_moves_case_level_params_to_module_step(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"title": ".title"},
            modules={
                "login": [
                    {"action": "fill", "selector": "title", "value": "${username}"}
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
                    "params": {"username": "qa"},
                    "steps": [
                        {"use_module": "login"},
                        {"action": "assert_visible", "selector": "title"},
                    ],
                }
            },
        }
    )

    case_data = payload["data"]["test_generated"]
    assert "params" not in case_data
    assert case_data["steps"][0] == {
        "use_module": "login",
        "params": {"username": "qa"},
    }


def test_generation_harness_prefers_spec_inputs_for_module_params(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"title": ".title"},
            modules={
                "login": [
                    {"action": "fill", "selector": "title", "value": "${username}"},
                    {"action": "fill", "selector": "title", "value": "${password}"},
                ]
            },
            variables={"common_password": "configured_pass"},
            test_cases=[],
            test_data={},
        ),
        spec={
            "mode": "smart",
            "inputs": {"username": "${username}", "password": "${password}"},
        },
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
                            "use_module": "login",
                            "params": {
                                "username": "${username}",
                                "password": "${common_password}",
                            },
                        },
                        {"action": "assert_visible", "selector": "title"},
                    ],
                }
            },
        }
    )

    assert payload["data"]["test_generated"]["steps"][0]["params"] == {
        "username": "${username}",
        "password": "${password}",
    }


def test_generation_harness_inlines_non_global_spec_inputs(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"product_id": "#product-id", "title": ".title"},
            modules={},
            variables={"username": "configured_user"},
            test_cases=[],
            test_data={},
        ),
        spec={"mode": "smart", "inputs": {"product_id": "SPU17799383833255667"}},
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
                            "selector": "product_id",
                            "value": "${product_id}",
                        },
                        {"action": "assert_visible", "selector": "title"},
                    ],
                }
            },
        }
    )

    assert payload["data"]["test_generated"]["steps"][0]["value"] == (
        "SPU17799383833255667"
    )
    assert harness.validate(payload) == []


def test_generation_harness_accepts_literal_values_without_inputs(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"product_id": "#product-id", "title": ".title"},
            modules={},
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec={
            "mode": "smart",
            "steps": ["输入商品ID SPU17799383833255667", "点击查询"],
            "final": ["商品ID输入框已输入 SPU17799383833255667"],
        },
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
                            "selector": "product_id",
                            "value": "SPU17799383833255667",
                        },
                        {
                            "action": "assert_text",
                            "selector": "title",
                            "value": "SPU17799383833255667",
                        },
                    ],
                }
            },
        }
    )

    assert payload["data"]["test_generated"]["steps"][0]["value"] == (
        "SPU17799383833255667"
    )
    assert harness.validate(payload) == []


def test_generation_harness_normalizes_input_value_attribute_assertion(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"product_id": "#product-id"},
            modules={},
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec={"mode": "smart", "final": ["商品ID输入框已输入 SPU17799383833255667"]},
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
                            "action": "assert_attribute",
                            "selector": "product_id",
                            "attribute": "value",
                            "value": "SPU17799383833255667",
                        },
                    ],
                }
            },
        }
    )

    step = payload["data"]["test_generated"]["steps"][0]
    assert step == {
        "action": "assert_value",
        "selector": "product_id",
        "value": "SPU17799383833255667",
    }
    assert harness.validate(payload) == []


def test_generation_harness_normalizes_action_use_module_module_name(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"title": ".title"},
            modules={
                "login": [
                    {"action": "fill", "selector": "title", "value": "${username}"}
                ]
            },
            variables={"username": "qa"},
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
                            "module_name": "login",
                            "params": {"username": "${username}"},
                        },
                        {"action": "assert_visible", "selector": "title"},
                    ],
                }
            },
        }
    )

    assert payload["data"]["test_generated"]["steps"][0] == {
        "use_module": "login",
        "params": {"username": "${username}"},
    }
    assert harness.validate(payload) == []


def test_generation_harness_keeps_supported_planning_assertions(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"product_id": "#product-id", "view_log": "a.log"},
            modules={},
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec={
            "mode": "smart",
            "checkpoints": [
                "页面标题包含商品审核",
                "商品ID输入框已输入 SPU17799383833255667",
            ],
            "final": ["当前页面仍在商品审核业务页"],
        },
        output_name="generated",
    )
    payload = harness.normalize(
        {
            "cases": [{"name": "test_generated"}],
            "data": {
                "test_generated": {
                    "mode": "smart",
                    "steps": [
                        {"action": "click", "selector": "view_log"},
                        {"action": "assert_visible", "selector": "view_log"},
                        {"action": "assert_title_contains", "value": "商品审核"},
                        {
                            "action": "assert_value",
                            "selector": "product_id",
                            "value": "SPU17799383833255667",
                        },
                    ],
                }
            },
        }
    )

    steps = payload["data"]["test_generated"]["steps"]
    assert [step["action"] for step in steps] == [
        "click",
        "assert_visible",
        "assert_title_contains",
        "assert_value",
    ]
    assert harness.validate(payload) == []


def test_generation_harness_normalizes_fill_value_text_assertion_to_supported_value_assertion(
    tmp_path: Path,
):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={
                "product_id": "#product-id",
                "audit_link": 'a:has-text("商品审核")',
            },
            modules={},
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec={
            "mode": "smart",
            "steps": ["输入商品ID SPU17799383833255667", "点击查询"],
            "final": ["商品ID输入框已输入 SPU17799383833255667"],
        },
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
                            "selector": "product_id",
                            "value": "SPU17799383833255667",
                        },
                        {
                            "action": "assert_text_contains",
                            "selector": "audit_link",
                            "value": "SPU17799383833255667",
                        },
                    ],
                }
            },
        }
    )

    steps = payload["data"]["test_generated"]["steps"]
    assert steps[1] == {
        "action": "assert_value",
        "selector": "product_id",
        "value": "SPU17799383833255667",
    }
    assert harness.validate(payload) == []


def test_generation_harness_rejects_unbacked_variable_without_inputs(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"product_id": "#product-id", "title": ".title"},
            modules={},
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec={
            "mode": "smart",
            "steps": ["输入商品ID SPU17799383833255667", "点击查询"],
            "final": ["商品ID输入框已输入 SPU17799383833255667"],
        },
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
                            "selector": "product_id",
                            "value": "${product_id}",
                        },
                        {"action": "assert_visible", "selector": "title"},
                    ],
                }
            },
        }
    )

    with pytest.raises(ValueError, match="product_id"):
        harness.validate(payload)


def test_generation_harness_drops_unbacked_generated_url_assertions(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="https://example.test/",
            elements={"title": ".title"},
            modules={},
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec={
            "mode": "smart",
            "final": ["current page remains on product audit page"],
        },
        output_name="generated",
    )
    payload = harness.normalize(
        {
            "cases": [{"name": "test_generated"}],
            "data": {
                "test_generated": {
                    "mode": "smart",
                    "steps": [
                        {"action": "assert_title_contains", "value": "Product Audit"},
                        {"action": "assert_url_contains", "value": "product-audit"},
                    ],
                }
            },
        }
    )

    steps = payload["data"]["test_generated"]["steps"]
    assert [step["action"] for step in steps] == ["assert_title_contains"]


def test_generation_harness_converts_cjk_display_url_assertion_to_title(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="https://example.test/",
            elements={},
            modules={},
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec={
            "mode": "smart",
            "final": ["当前页面仍在商品审核业务页"],
        },
        output_name="generated",
    )
    payload = harness.normalize(
        {
            "cases": [{"name": "test_generated"}],
            "data": {
                "test_generated": {
                    "mode": "smart",
                    "steps": [
                        {"action": "assert_url_contains", "value": "商品审核"},
                    ],
                }
            },
        }
    )

    assert payload["data"]["test_generated"]["steps"] == [
        {"action": "assert_title_contains", "value": "商品审核"}
    ]


def test_generation_harness_keeps_spec_backed_url_assertions(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="https://example.test/",
            elements={"title": ".title"},
            modules={},
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec={"mode": "smart", "final": ["url contains checkout-complete"]},
        output_name="generated",
    )
    payload = harness.normalize(
        {
            "cases": [{"name": "test_generated"}],
            "data": {
                "test_generated": {
                    "mode": "smart",
                    "steps": [
                        {"action": "assert_url_contains", "value": "checkout-complete"},
                    ],
                }
            },
        }
    )

    assert payload["data"]["test_generated"]["steps"] == [
        {"action": "assert_url_contains", "value": "checkout-complete"}
    ]


def test_generation_harness_does_not_override_existing_context_vars(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={},
            modules={},
            variables={"username": "configured_user", "password": "configured_pass"},
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
                    "steps": [{"action": "assert_title_contains", "value": "ok"}],
                }
            },
            "vars": {
                "username": "your_username",
                "password": "your_password",
                "new_token": "abc",
            },
        }
    )

    assert payload["vars"] == {"new_token": "abc"}


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
    selectors = heuristic_selectors("password input", "fill")

    assert selectors[0] == 'input[type="password"]'
    assert 'input[name*="password" i]' in selectors


def test_heuristic_selectors_strip_text_prefix_and_add_contextual_text_selectors():
    selectors = heuristic_selectors("text=商城", "click")

    assert "text=商城" not in selectors
    assert "text=text=商城" not in selectors
    assert 'button:has-text("text=商城")' not in selectors
    assert any("ant-pro-menu-item-title" in selector for selector in selectors)
    assert any(
        selector.startswith("//span")
        and "normalize-space()='商城'" in selector
        for selector in selectors
    )
    assert selectors.index('button:has-text("商城")') > min(
        index
        for index, selector in enumerate(selectors)
        if "ant-pro-menu-item-title" in selector
    )


def test_heuristic_selectors_do_not_use_ambiguous_short_id_token_for_fill():
    selectors = heuristic_selectors("text=输入商品id/商品名称", "fill")

    assert "input" not in selectors
    assert "textarea" not in selectors
    assert not any('[id*="id" i]' in selector for selector in selectors)
    assert any("商品名称" in selector for selector in selectors)


def test_heuristic_selectors_match_cjk_display_spacing_precisely():
    selectors = heuristic_selectors("text=查询", "click")

    assert any(
        "translate(normalize-space(), ' ', '')='查询'" in selector
        for selector in selectors
    )
    assert 'button:has-text("查询")' in selectors


def test_heuristic_selectors_expand_precise_action_synonyms():
    selectors = heuristic_selectors("text=搜索按钮", "click")

    assert any('button:has-text("查询")' == selector for selector in selectors)
    assert any(
        "translate(normalize-space(), ' ', '')='检索'" in selector
        for selector in selectors
    )


def test_locator_context_prioritizes_cjk_display_spaced_target():
    context = build_locator_context(
        action="click",
        target="text=Search",
        limit=3,
        url="https://example.test/list",
        title="List",
        candidates=[
            {
                "index": 1,
                "tag": "a",
                "text": "Logout",
                "role": "link",
                "selector": 'a:has-text("Logout")',
                "visible": True,
                "enabled": True,
            },
            {
                "index": 2,
                "tag": "button",
                "text": "Sea rch",
                "selector": 'button:has-text("Sea rch")',
                "visible": True,
                "enabled": True,
            },
            {
                "index": 3,
                "tag": "input",
                "id": "brandId",
                "label": "Brand",
                "selector": "#brandId",
                "visible": True,
                "enabled": True,
            },
        ],
    )

    assert any(item.get("text") == "Sea rch" for item in context["candidates"])


def test_locator_context_prioritizes_precise_action_synonym():
    context = build_locator_context(
        action="click",
        target="text=search button",
        limit=2,
        url="https://example.test/list",
        title="List",
        candidates=[
            {
                "index": 1,
                "tag": "a",
                "text": "Settings",
                "role": "link",
                "selector": 'a:has-text("Settings")',
                "visible": True,
                "enabled": True,
            },
            {
                "index": 2,
                "tag": "button",
                "text": "Search",
                "selector": 'button:has-text("Search")',
                "visible": True,
                "enabled": True,
            },
        ],
    )

    assert context["candidates"][0].get("text") == "Search"


def test_locator_context_keeps_semantic_suffix_action_candidate():
    candidates = [
        {
            "index": index,
            "tag": "div",
            "role": "menuitem",
            "text": f"Menu {index}",
            "selector": f"li:nth-of-type({index}) > div",
            "visible": True,
            "enabled": True,
        }
        for index in range(30)
    ]
    candidates.append(
        {
            "index": 100,
            "tag": "a",
            "role": "link",
            "text": "Log",
            "selector": 'a:has-text("Log")',
            "visible": True,
            "enabled": True,
        }
    )

    context = build_locator_context(
        action="click",
        target="text=view log",
        limit=8,
        url="https://example.test/list",
        title="List",
        candidates=candidates,
    )

    assert any(item.get("text") == "Log" for item in context["candidates"])


def test_semantic_selectors_rank_semantics_then_stable_concise_selector(monkeypatch):
    class FakePage:
        pass

    candidates = [
        {
            "index": 0,
            "tag": "button",
            "selector": "main > section > div:nth-of-type(3) > button",
            "text": "Login",
            "visible": True,
            "enabled": True,
        },
        {
            "index": 1,
            "tag": "button",
            "selector": 'button[data-test="login"]',
            "data_test": "login",
            "text": "Login",
            "visible": True,
            "enabled": True,
        },
        {
            "index": 2,
            "tag": "a",
            "selector": 'a[title="Download app"]',
            "title": "Download app",
            "text": "Download app",
            "visible": True,
            "enabled": True,
        },
    ]

    monkeypatch.setattr(
        "src.ai_runtime.playwright_selectors.collect_candidates",
        lambda *args, **kwargs: candidates,
    )

    selectors = semantic_selectors(FakePage(), "login button", "click")

    assert selectors[0] == 'button[data-test="login"]'
    assert 'a[title="Download app"]' not in selectors


def test_verified_target_candidates_reuse_semantic_selector_pipeline(monkeypatch):
    class FakeLocator:
        @property
        def first(self):
            return self

        def count(self):
            return 1

        def is_visible(self, *args, **kwargs):
            return True

        def is_enabled(self, *args, **kwargs):
            return True

    class FakePage:
        def locator(self, selector):
            return FakeLocator()

    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.semantic_selectors",
        lambda *args, **kwargs: ['button:has-text("Search")'],
    )
    monkeypatch.setattr(
        "src.ai_runtime.agent_case_executor.heuristic_selectors",
        lambda *args, **kwargs: [],
    )

    candidates = _verified_target_candidates(
        page=FakePage(),
        dom_context={
            "meta": {"url": "https://example.test/list"},
            "forms": [],
            "interactive_elements": [],
            "assertion_candidates": [],
        },
        target="search",
        action="click",
        limit=3,
    )

    assert candidates == [
        {
            "selector": 'button:has-text("Search")',
            "source": "verified_selector",
            "text": "Search",
            "match_score": 86,
        }
    ]


def test_verified_target_candidates_rank_full_semantic_match_before_generic_term():
    class FakeLocator:
        @property
        def first(self):
            return self

        def count(self):
            return 1

        def is_visible(self, *args, **kwargs):
            return True

        def is_enabled(self, *args, **kwargs):
            return True

    class FakePage:
        def locator(self, selector):
            return FakeLocator()

    candidates = _verified_target_candidates(
        page=FakePage(),
        dom_context={
            "meta": {"url": "https://example.test/login"},
            "forms": [],
            "interactive_elements": [
                {
                    "id": "e1",
                    "tag": "div",
                    "role": "tab",
                    "text": "账号密码登录",
                    "selector_candidates": ["#account-tab"],
                },
                {
                    "id": "e2",
                    "tag": "button",
                    "role": "button",
                    "text": "使用OA登录 >>",
                    "selector_candidates": ['button:has-text("使用OA登录 >>")'],
                },
            ],
            "assertion_candidates": [],
        },
        target="OA登录",
        action="click",
        limit=2,
    )

    assert candidates[0]["element_id"] == "e2"


def test_vision_find_result_contract_accepts_service_payload():
    result = VisionFindResult.model_validate(
        {
            "found": True,
            "target": "鐧诲綍鎸夐挳",
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


def test_selector_decision_normalizes_common_status_aliases():
    decision = SelectorDecision.model_validate(
        {"status": "success", "element_id": "e1", "confidence": 0.9}
    )

    assert decision.status == "ok"


def test_vision_find_result_normalizes_negative_not_found_index():
    result = VisionFindResult.model_validate(
        {"found": False, "selected_candidate_index": -1, "reason": "not found"}
    )

    assert result.selected_candidate_index is None


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
        target="password input",
        selector="#user-name",
        source="heuristic",
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = registry

    resolved = resolver.resolve(
        action="fill",
        target="password input",
        selector=None,
        mode="smart",
        timeout=1000,
    )

    assert resolved.selector == "#password"
    assert resolved.cache_action == "fill"
    assert resolved.cache_target == "password input"
    assert resolved.cache_page_key == "https://www.saucedemo.com/"
    assert (
        registry.find(
            project="demo",
            env="test",
            page_key="https://www.saucedemo.com/",
            action="fill",
            target="password input",
        )
        is None
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

        def count(self):
            return 1 if self.selector == "text=淇濆瓨" else 0

        def nth(self, index):
            return self

        def is_visible(self):
            return self.selector == "text=淇濆瓨"

    class FakePage:
        url = "http://example.test/form"

        def locator(self, selector):
            return FakeLocator(selector)

        def get_by_text(self, text, exact=False):
            return FakeLocator(f"text={text}")

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
        target="淇濆瓨",
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

        def get_by_text(self, text, exact=False):
            class FakeTextLocator:
                def count(self):
                    return 1

                def nth(self, index):
                    return self

                def is_visible(self):
                    return True

            return FakeTextLocator()

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
            target="淇濆瓨",
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


def test_selector_registry_filters_low_score_and_deprecates_after_failures(
    tmp_path: Path,
):
    registry = SelectorRegistry(tmp_path / "selectors.db")
    try:
        record = registry.save(
            project="demo",
            env="test",
            page_key="/login",
            action="click",
            target="submit",
            selector="#weak",
            source="ai_selector",
            confidence=0.2,
        )

        assert (
            registry.find(
                project="demo",
                env="test",
                page_key="/login",
                action="click",
                target="submit",
                min_score=0.75,
            )
            is None
        )

        registry.mark_failed(
            record.id,
            unstable_threshold=1,
            deprecated_after_failures=2,
            last_error="first failure",
        )
        unstable = registry.find(
            project="demo",
            env="test",
            page_key="/login",
            action="click",
            target="submit",
        )
        assert unstable is not None
        assert unstable.status == "unstable"

        registry.mark_failed(
            record.id,
            unstable_threshold=1,
            deprecated_after_failures=2,
            last_error="second failure",
        )
        assert (
            registry.find(
                project="demo",
                env="test",
                page_key="/login",
                action="click",
                target="submit",
            )
            is None
        )
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
                "    intent: fill product id and search",
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
