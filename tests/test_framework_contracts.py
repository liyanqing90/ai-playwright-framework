import json
import os
from pathlib import Path
from typing import Any

import pytest
import requests
from ruamel.yaml import YAML

from ai_playwright.page_objects.base_page import _url_contains
from ai_playwright.ai_generation import case_generator as case_generator_module
from ai_playwright.ai_generation.case_generator import (
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
from ai_playwright.ai_generation.harness import GenerationHarness
from ai_playwright.ai_generation.harness import _safe_case_name
from ai_playwright.ai_generation.pipeline import execute_compiled_payload_steps
from ai_playwright.ai_generation.project_context import ProjectContext
from ai_playwright.ai_runtime.cache_scope import normalize_entry_url
from ai_playwright.ai_runtime.agent_case_executor import (
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
from ai_playwright.ai_runtime.contracts import (
    AgentCaseDecision,
    AiStepDecision,
    GeneratedCasePayload,
    SelectorDecision,
)
from ai_playwright.ai_runtime.element_store import ElementDefinitionStore
from ai_playwright.ai_runtime.payload_compactor import (
    build_dom_context,
    build_locator_context,
    compact_model_dom_context,
    compact_dom_candidates,
    compact_history,
    selector_for_element_id,
)
from ai_playwright.ai_runtime.playwright_selectors import (
    collect_candidates,
    heuristic_selectors,
    is_high_quality_selector,
    semantic_selectors,
    selector_matches_target,
    validate_selector,
)
from ai_playwright.ai_runtime.provider import (
    ChatCompletionProvider,
    LLMSettings,
    build_response_format,
    load_llm_settings,
    openai_strict_schema,
    parse_json_object,
    parse_model_response,
)
from ai_playwright.ai_runtime.selector_registry import SelectorRegistry
from ai_playwright.ai_runtime.smart_resolver import (
    AiStepOperation,
    ResolvedSelector,
    SmartResolver,
    _parse_fill_instruction,
)
from ai_playwright.runner import build_test_signature
from ai_playwright.test_case_executor import CaseExecutor
from ai_playwright.step_actions.commands import (
    assertion_commands as assertion_commands_module,
)
from ai_playwright.step_actions.safe_expression import (
    SafeExpressionError,
    safe_eval_expression,
)
from ai_playwright.step_actions.step_executor import StepExecutor
from ai_playwright.step_actions import step_executor as step_executor_module
from ai_playwright.step_actions.commands.wait_commands import WaitForElementTextCommand
from ai_playwright.utils.variable_manager import VariableManager
from ai_playwright.utils.yaml_handler import YamlHandler
from ai_playwright.utils.token_usage import TokenUsageTracker, normalize_token_usage
from ai_playwright.utils.config import Config


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
        "ai_playwright.step_actions.step_executor.execute_action_with_command",
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
        "ai_playwright.step_actions.step_executor.execute_action_with_command",
        fake_execute_action_with_command,
    )

    executor = StepExecutor(FakePage(), FakeUiHelper(), elements={})
    executor.ai_config.setdefault("runtime", {})["action_stable_timeout_ms"] = 1234
    executor.ai_config["runtime"]["action_stable_idle_ms"] = 321
    executor.ai_config["runtime"]["wait_for_stable_after_action"] = True

    executor.execute_step({"action": "click", "selector": "#go"})

    assert calls == [
        ("execute", "click", "#go", None),
        ("stable", 1234, 321),
    ]


def test_step_executor_does_not_wait_for_stable_by_default(monkeypatch):
    calls: list[Any] = []

    def fake_execute_action_with_command(ui_helper, action, selector, value, step):
        calls.append(("execute", action, selector, value))

    class FakePage:
        url = "https://example.test"

    class FakeUiHelper:
        def wait_for_stable(self, *, timeout: int, idle_ms: int):
            calls.append(("stable", timeout, idle_ms))

    monkeypatch.setattr(
        "ai_playwright.step_actions.step_executor.execute_action_with_command",
        fake_execute_action_with_command,
    )

    executor = StepExecutor(FakePage(), FakeUiHelper(), elements={})
    executor.ai_config.setdefault("runtime", {}).pop(
        "wait_for_stable_after_action", None
    )
    executor.ai_config["runtime"].pop("auto_wait_for_stable", None)

    executor.execute_step({"action": "click", "selector": "#go"})

    assert calls == [("execute", "click", "#go", None)]


def test_declared_non_ai_actions_are_registered():
    from ai_playwright.step_actions.action_types import StepAction
    from ai_playwright.step_actions.commands.base_command import CommandFactory

    all_actions = {
        action.lower()
        for attr in dir(StepAction)
        if isinstance((group := getattr(StepAction, attr)), list)
        for action in group
    }
    special_actions = {
        action.lower()
        for group in (
            StepAction.USE_MODULE,
            StepAction.IF_CONDITION,
            StepAction.FOR_EACH,
            StepAction.AI_STEP,
        )
        for action in group
    }

    assert sorted(all_actions - set(CommandFactory._commands) - special_actions) == []


def test_action_registry_is_schema_source():
    from ai_playwright.step_actions.action_registry import (
        ACTION_ALLOWED_FIELDS as REGISTRY_ALLOWED_FIELDS,
        ACTION_SPEC_BY_NAME,
        NO_SELECTOR_ACTIONS as REGISTRY_NO_SELECTOR_ACTIONS,
        VALID_ACTIONS as REGISTRY_VALID_ACTIONS,
    )
    from ai_playwright.yaml_schema import (
        ACTION_ALLOWED_FIELDS as SCHEMA_ALLOWED_FIELDS,
        NO_SELECTOR_ACTIONS as SCHEMA_NO_SELECTOR_ACTIONS,
        VALID_ACTIONS as SCHEMA_VALID_ACTIONS,
    )

    assert SCHEMA_VALID_ACTIONS == REGISTRY_VALID_ACTIONS
    assert SCHEMA_NO_SELECTOR_ACTIONS == REGISTRY_NO_SELECTOR_ACTIONS
    assert SCHEMA_ALLOWED_FIELDS == REGISTRY_ALLOWED_FIELDS
    assert ACTION_SPEC_BY_NAME["execute_js"].unsafe is True


def test_javascript_action_aliases_resolve_to_script_command():
    from ai_playwright.step_actions.commands.base_command import CommandFactory

    for action in ("execute_script", "execute_js", "eval_js", "evaluate", "javascript"):
        command = CommandFactory.get_command(action)

        assert command is not None
        assert command.__class__.__name__ == "ExecuteScriptCommand"


def test_execute_js_is_denied_by_default(monkeypatch):
    from ai_playwright.step_actions.commands.base_command import CommandFactory

    monkeypatch.delenv("AI_PLAYWRIGHT_ALLOW_UNSAFE_ACTIONS", raising=False)
    command = CommandFactory.get_command("execute_js")
    assert command is not None

    with pytest.raises(PermissionError, match="默认禁用"):
        command.execute(
            None, "", None, {"action": "execute_js", "script": "() => true"}
        )


def test_execute_js_can_call_helper_when_explicitly_enabled(monkeypatch):
    from ai_playwright.step_actions.commands.base_command import CommandFactory

    calls: list[str] = []

    class FakeUiHelper:
        def execute_script(self, script: str) -> None:
            calls.append(script)

    monkeypatch.setenv("AI_PLAYWRIGHT_ALLOW_UNSAFE_ACTIONS", "1")
    command = CommandFactory.get_command("execute_js")
    assert command is not None
    command.execute(
        FakeUiHelper(),
        "",
        None,
        {"action": "execute_js", "script": "() => window.__demo = true"},
    )

    assert calls == ["() => window.__demo = true"]


def test_structured_locator_builds_role_locator():
    from ai_playwright.page_objects.base_page import _semantic_locator

    calls: list[Any] = []

    class FakeLocator:
        def __init__(self):
            self.first = self

        def filter(self, **kwargs):
            calls.append(("filter", kwargs))
            return self

        def nth(self, index):
            calls.append(("nth", index))
            return self

    class FakePage:
        def get_by_role(self, role, **kwargs):
            calls.append(("role", role, kwargs))
            return FakeLocator()

    _semantic_locator(
        FakePage(),
        {
            "role": "button",
            "name": "Submit",
            "exact": True,
            "has_text": "Submit",
            "nth": 1,
        },
    )

    assert calls == [
        ("role", "button", {"name": "Submit", "exact": True}),
        ("filter", {"has_text": "Submit"}),
        ("nth", 1),
    ]


def test_network_context_commands_delegate_to_ui_helper():
    from ai_playwright.step_actions.commands.network_commands import (
        AbortRouteCommand,
        GrantPermissionsCommand,
        MockRouteCommand,
        SaveStorageStateCommand,
        SetOfflineCommand,
        WaitForResponseCommand,
    )

    calls: list[Any] = []

    class FakeVariableManager:
        def set_variable(self, name, value, scope):
            calls.append(("store", name, value, scope))

    class FakeUiHelper:
        variable_manager = FakeVariableManager()

        def wait_for_response(self, url_pattern, timeout):
            calls.append(("wait_response", url_pattern, timeout))
            return {"status": 200}

        def mock_route(self, **kwargs):
            calls.append(("mock", kwargs))

        def abort_route(self, **kwargs):
            calls.append(("abort", kwargs))

        def set_offline(self, offline):
            calls.append(("offline", offline))

        def grant_permissions(self, permissions, origin=None):
            calls.append(("permissions", permissions, origin))

        def save_storage_state(self, path=None):
            calls.append(("storage", path))
            return {"cookies": []}

    ui = FakeUiHelper()
    WaitForResponseCommand().execute(
        ui,
        "",
        None,
        {"url_pattern": "/api/items", "variable_name": "response"},
    )
    MockRouteCommand().execute(ui, "", None, {"url_pattern": "**/api", "status": 201})
    AbortRouteCommand().execute(ui, "", None, {"url_pattern": "**/ads"})
    SetOfflineCommand().execute(ui, "", None, {"offline": "true"})
    GrantPermissionsCommand().execute(
        ui, "", None, {"permissions": "geolocation", "origin": "https://example.test"}
    )
    SaveStorageStateCommand().execute(ui, "", None, {"path": "state.json"})

    assert calls == [
        ("wait_response", "**/api/items**", 10000),
        ("store", "response", {"status": 200}, "global"),
        (
            "mock",
            {
                "url_pattern": "**/api",
                "status": 201,
                "body": "",
                "json_data": None,
                "headers": None,
                "content_type": None,
            },
        ),
        ("abort", {"url_pattern": "**/ads", "error_code": "failed"}),
        ("offline", True),
        ("permissions", ["geolocation"], "https://example.test"),
        ("storage", "state.json"),
    ]


def test_step_executor_classifies_failures_and_records_action_result(monkeypatch):
    class FakePage:
        url = "https://example.test"

    class FakeUiHelper:
        page_errors = []

    def fake_execute_action_with_command(ui_helper, action, selector, value, step):
        return {"ok": True}

    monkeypatch.setattr(
        "ai_playwright.step_actions.step_executor.execute_action_with_command",
        fake_execute_action_with_command,
    )

    step = {"action": "get_text", "selector": "#message"}
    executor = StepExecutor(FakePage(), FakeUiHelper(), elements={})
    executor.execute_step(step)

    assert step["_action_result"] == {
        "success": True,
        "failure_kind": None,
        "artifacts": {},
        "data": {"ok": True},
    }
    assert (
        StepExecutor._classify_step_exception(TimeoutError("locator timeout"))
        == "locator_timeout"
    )


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
        "ai_playwright.step_actions.step_executor.execute_action_with_command",
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
        "ai_playwright.step_actions.step_executor.execute_action_with_command",
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
        "ai_playwright.step_actions.step_executor.execute_action_with_command",
        fake_execute_action_with_command,
    )

    executor = StepExecutor(
        FakePage(),
        FakeUiHelper(),
        elements={"action_button": 'button:has-text("Submit")'},
    )
    executor.smart_resolver = FakeResolver()

    executor.execute_step(
        {"action": "click", "selector": "action_button", "mode": "smart"}
    )

    assert events[0] == (
        "resolve",
        'button:has-text("Submit")',
        'button:has-text("Submit")',
    )
    assert events[1] == ("execute", "click", 'button:has-text("Submit")', None)


def test_step_executor_does_not_treat_raw_selector_as_smart_target(monkeypatch):
    events: list[Any] = []

    class FakeResolver:
        def resolve(self, **kwargs):
            raise AssertionError("raw selector should not invoke smart resolver")

    class FakePage:
        url = "https://example.test/products"

    class FakeUiHelper:
        pass

    def fake_execute_action_with_command(ui_helper, action, selector, value, step):
        events.append(("execute", action, selector, value))

    monkeypatch.setattr(
        "ai_playwright.step_actions.step_executor.execute_action_with_command",
        fake_execute_action_with_command,
    )

    executor = StepExecutor(FakePage(), FakeUiHelper(), elements={})
    executor.smart_resolver = FakeResolver()

    executor.execute_step(
        {
            "action": "assert_text",
            "selector": ".title",
            "value": "Products",
            "mode": "smart",
        }
    )

    assert events == [("execute", "assert_text", ".title", "Products")]


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
        "ai_playwright.step_actions.step_executor.execute_action_with_command",
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
        "ai_playwright.step_actions.step_executor.execute_action_with_command",
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
        "ai_playwright.step_actions.step_executor.execute_action_with_command",
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
        "ai_playwright.step_actions.step_executor.execute_action_with_command",
        fake_execute_action_with_command,
    )

    executor = StepExecutor(FakePage(), FakeUiHelper(), elements={})
    executor.smart_resolver = FakeResolver()

    executor.execute_step(
        {"action": "click", "target": "login button", "mode": "smart"}
    )

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
        "ai_playwright.step_actions.step_executor.execute_action_with_command",
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

    monkeypatch.setattr(
        "ai_playwright.test_case_executor.AgentCaseExecutor", FakeAgentRunner
    )

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
                "target": "商品编号输入框",
                "value": "DEMO-SKU-001",
            },
            {
                "action": "assert_value",
                "target": "商品编号输入框",
                "value": "DEMO-SKU-001",
            },
        ],
    )

    assert executed[1]["selector"] == "#product-id"
    assert result.history[1]["step"]["selector"] == "#product-id"


def test_test_signature_uses_standard_page_fixtures():
    params = list(build_test_signature([]).parameters)

    assert params[:4] == ["page", "ui_helper", "get_test_name", "value"]


def test_run_case_passes_headed_options_to_pytest_playwright(tmp_path: Path):
    from ai_playwright.cli.run_case import build_pytest_args

    previous = Config()
    previous_state = {
        "project": previous.project,
        "env": previous.env,
        "base_url": previous.base_url,
        "test_dir": previous.test_dir,
        "headed": previous.headed,
        "browser": previous.browser,
        "slow_mo": previous.slow_mo,
        "test_file": previous.test_file,
    }
    try:
        config = Config(
            project="demo",
            env="prod",
            test_dir=str(tmp_path),
            headed=True,
            browser="firefox",
            slow_mo=250,
            test_file="smoke.yaml",
        )

        args = build_pytest_args(config)

        assert "--browser" in args
        assert args[args.index("--browser") + 1] == "firefox"
        assert "--headed" in args
        assert "--slowmo=250" in args
    finally:
        Config(**previous_state)


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


def test_browser_launch_options_follow_config_headed_and_slow_mo():
    from ai_playwright import pytest_plugin

    previous_headed = pytest_plugin.config.headed
    previous_slow_mo = pytest_plugin.config.slow_mo
    try:
        Config(project="demo", env="prod", headed=True, slow_mo=250)

        assert pytest_plugin._browser_launch_options() == {
            "headless": False,
            "slow_mo": 250,
        }

        Config(project="demo", env="prod", headed=False, slow_mo=0)

        assert pytest_plugin._browser_launch_options() == {"headless": True}
    finally:
        Config(
            project="demo",
            env="prod",
            headed=previous_headed,
            slow_mo=previous_slow_mo,
        )


def test_browser_launch_options_follow_runtime_headless_env(monkeypatch):
    from ai_playwright import pytest_plugin

    monkeypatch.setenv("PWHEADED", "0")
    monkeypatch.setenv("PWSLOWMO", "125")

    assert pytest_plugin._browser_launch_options() == {
        "headless": True,
        "slow_mo": 125,
    }


def test_browser_launch_options_default_to_headed_without_runtime_env(monkeypatch):
    from ai_playwright import pytest_plugin

    monkeypatch.delenv("PWHEADED", raising=False)
    monkeypatch.delenv("PWSLOWMO", raising=False)
    Config(project="demo", env="prod", headed=True, slow_mo=0)

    assert pytest_plugin._browser_launch_options() == {"headless": False}


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
