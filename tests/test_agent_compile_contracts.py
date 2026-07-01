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
        "ai_playwright.ai_runtime.provider.load_ai_config",
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
        "ai_playwright.ai_runtime.agent_case_executor.load_ai_config",
        lambda: {
            "runtime": {"agent_model": "runtime-fast-model"},
            "agent_policy": {"limits": {}},
        },
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor.load_llm_settings",
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


def test_agent_case_plan_cache_runs_without_model_compile(monkeypatch, tmp_path: Path):
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
        "ai_playwright.ai_runtime.agent_case_executor.StepExecutor",
        FakeStepExecutor,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor.load_ai_config",
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
        "ai_playwright.ai_runtime.agent_case_executor._unmet_final_criteria",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor._unmet_completion_criteria",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor._unmet_intent_action_requirements",
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
        "ai_playwright.ai_runtime.agent_case_executor._build_payload",
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
    assert {key: executed[1].get(key) for key in ("action", "target", "mode")} == {
        "action": "click",
        "target": "Login button",
        "mode": "smart",
    }
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
        "ai_playwright.ai_runtime.agent_case_executor.load_ai_config",
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
        "ai_playwright.ai_runtime.agent_case_executor._build_payload",
        lambda *args, **kwargs: (
            kwargs.get("cache_info", {}).update({"cache_hit": False, "model_calls": 1})
            or payload
        ),
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor._unmet_completion_criteria",
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
        "ai_playwright.ai_runtime.agent_case_executor._unmet_intent_action_requirements",
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
    assert (
        executor.step_executor.elements["compiled_button"] == "button:has-text('Run')"
    )
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
        "ai_playwright.ai_runtime.agent_case_executor.load_ai_config",
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
        "ai_playwright.ai_runtime.agent_case_executor._unmet_completion_criteria",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor._unmet_intent_action_requirements",
        lambda **kwargs: [],
    )

    def fake_build_payload(*args, **kwargs):
        calls["count"] += 1
        kwargs.get("cache_info", {}).update({"cache_hit": False, "model_calls": 1})
        return payload

    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor._build_payload",
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
    monkeypatch.setattr(
        executor, "_execute_step", lambda step, *, spec: executed.append(dict(step))
    )
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
        "ai_playwright.ai_runtime.agent_case_executor._build_payload",
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
    monkeypatch.setattr(
        second_executor,
        "_execute_step",
        lambda step, *, spec: executed.append(dict(step)),
    )
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
        "ai_playwright.ai_runtime.agent_case_executor.load_ai_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor._build_payload",
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
    assert (
        executor.plan_cache.load_plan(
            executor._cache_key(case_name="test_agent", spec=spec)
        )
        is None
    )
    assert calls["count"] == 1


def test_agent_case_generation_spec_forbids_new_runtime_modules():
    generation_spec = _agent_spec_to_generation_spec(
        case_name="test_agent",
        spec={
            "description": "",
            "intent": "login and search",
            "steps": ["使用登录模块", "查询商品"],
            "inputs": {"product_id": "DEMO-SKU-1"},
            "criteria": {"final": ["完成"]},
        },
        allowed_modules=["admin_login_and_navigate"],
    )

    assert generation_spec["runtime_compile"] == {
        "mode": "agent_case",
        "allow_new_modules": False,
        "allowed_modules": ["admin_login_and_navigate"],
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
            allowed_modules={"admin_login_and_navigate"},
        )


def test_agent_case_runtime_compile_allows_existing_yaml_module_reference():
    payload = {
        "cases": [{"name": "test_agent"}],
        "data": {
            "test_agent": {
                "mode": "smart",
                "steps": [
                    {"use_module": "admin_login_and_navigate", "params": {}},
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
        allowed_modules={"admin_login_and_navigate"},
    )


def test_agent_case_compile_contract_error_does_not_fallback_to_realtime(
    monkeypatch, tmp_path: Path
):
    payload = {
        "cases": [{"name": "test_agent"}],
        "data": {
            "test_agent": {
                "mode": "smart",
                "steps": [{"use_module": "admin_search_param", "params": {}}],
            }
        },
        "elements": {},
        "modules": {"admin_search_param": [{"action": "click", "selector": "#search"}]},
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
        "ai_playwright.ai_runtime.agent_case_executor.load_ai_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor._build_payload",
        lambda *args, **kwargs: (
            kwargs.get("cache_info", {}).update({"cache_hit": False, "model_calls": 1})
            or payload
        ),
    )

    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://example.test/",
        elements={},
        modules={"admin_login_and_navigate": [{"action": "goto", "value": "/"}]},
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
            AssertionError(
                "runtime planner should not run after compile contract error"
            )
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
        "ai_playwright.ai_runtime.agent_case_executor.load_ai_config",
        lambda: config,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor._build_payload",
        lambda *args, **kwargs: (
            kwargs.get("cache_info", {}).update({"cache_hit": False, "model_calls": 1})
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
    assert not any(item[0] == "open_start" and item[1] is True for item in events)


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
        "ai_playwright.ai_runtime.agent_case_executor.StepExecutor",
        fake_step_executor,
    )
    monkeypatch.setattr(
        "ai_playwright.step_actions.step_executor.execute_action_with_command",
        lambda ui_helper, action, selector, value, step: events.append(
            ("execute", action, selector, value)
        ),
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor.load_ai_config",
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
        "ai_playwright.ai_generation.case_generator.load_ai_config",
        lambda: {
            "runtime": {"ai_cache_sqlite_path": str(tmp_path / "ai_cache.sqlite3")},
            "generation": {"max_context_items": 5},
            "llm": {"schema_version": "schema-test"},
        },
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor._build_payload",
        lambda *args, **kwargs: (
            kwargs.get("cache_info", {}).update({"cache_hit": False, "model_calls": 1})
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
        "ai_playwright.ai_runtime.agent_case_executor.load_ai_config",
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
        "ai_playwright.ai_runtime.agent_case_executor._build_payload",
        lambda *args, **kwargs: (
            kwargs.get("cache_info", {}).update({"cache_hit": False, "model_calls": 1})
            or payload
        ),
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor._unmet_completion_criteria",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("local completion inference should not run")
        ),
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor._unmet_intent_action_requirements",
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
        "ai_playwright.ai_runtime.agent_case_executor.load_ai_config",
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
        "ai_playwright.ai_runtime.agent_case_executor.load_ai_config",
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
        lambda step, *, spec: (_ for _ in ()).throw(AssertionError("assert failed")),
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
            "intent": "输入商品编号 DEMO-SKU-001 后查询",
            "steps": ["输入商品编号 DEMO-SKU-001", "点击查询"],
            "inputs": {},
            "criteria": {
                "checkpoints": ["商品编号输入框已输入 DEMO-SKU-001"],
                "final": ["查询结果可见"],
            },
        },
    )

    assert "inputs" not in generation_spec
    assert "inputs" not in generation_spec["cases"][0]
    assert generation_spec["cases"][0]["steps"][0] == ("输入商品编号 DEMO-SKU-001")


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
        "ai_playwright.ai_runtime.agent_case_executor.StepExecutor",
        FakeStepExecutor,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor.ChatCompletionProvider",
        FakeProvider,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor.collect_candidates_diagnostic",
        lambda page, **kwargs: {
            "candidates": [
                {
                    "index": 0,
                    "tag": "h1",
                    "selector": "h1",
                    "text": "Order Complete",
                    "visible": True,
                    "enabled": True,
                }
            ],
        },
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor.load_ai_config",
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
    assert (
        result.final_reason == "local completion criteria satisfied before model call"
    )


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
