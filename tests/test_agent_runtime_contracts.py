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
            "selector": "#catalog-review-search_brandid",
            "id": "catalog-review-search_brandid",
            "label": "品牌",
            "type": "search",
            "visible": True,
            "enabled": True,
        },
        {
            "index": 4,
            "tag": "input",
            "selector": "#catalog-review-search_seriesid",
            "id": "catalog-review-search_seriesid",
            "label": "系列",
            "type": "search",
            "visible": True,
            "enabled": True,
        },
        {
            "index": 5,
            "tag": "input",
            "selector": "#catalog-review-search_sku",
            "id": "catalog-review-search_sku",
            "label": "商品编号",
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
    hints = ["输入商品编号 DEMO-SKU-002，点击查询，然后点击查看日志"]
    context = build_dom_context(
        candidates,
        url="https://example.test/catalogReviewList",
        title="目录审核",
        limit=7,
        hints=hints,
        data_policy="trusted_local",
    )

    compacted = compact_model_dom_context(
        context,
        candidate_limit=3,
        form_limit=2,
        assertion_limit=2,
        hints=hints,
    )

    assert compacted["forms"][0]["label"] == "商品编号"
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
        "time_budget_ms": 0,
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
        "ai_playwright.ai_runtime.agent_case_executor.StepExecutor",
        FakeStepExecutor,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor.ChatCompletionProvider",
        FakeProvider,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor.collect_candidates",
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
        "ai_playwright.ai_runtime.agent_case_executor.load_ai_config",
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
        "ai_playwright.ai_runtime.agent_case_executor.compile_case_payload",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("this test exercises realtime prompt payload")
        ),
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor._unmet_final_criteria",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor._has_completion_criteria",
        lambda criteria: False,
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
            "expand mall menu, click catalog review, fill product id DEMO-SKU-1"
        ),
        "steps": [],
        "inputs": {"username": "${username}", "password": "${password}"},
        "criteria": {
            "checkpoints": [
                "page title contains catalog review",
                "product id input filled DEMO-SKU-1",
            ],
            "final": ["still on catalog review page"],
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
        spec={
            "intent": "使用项目模块 login，输入搜索词 hello，点击搜索按钮，然后点击结果详情"
        },
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


def test_agent_case_title_checkpoint_ignores_visible_text_assertion_history():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": ["登录成功后进入商品列表页，页面标题为 Products"],
            "final": [],
        },
        history=[
            {
                "step": {
                    "action": "assert_text",
                    "selector": ".title",
                    "value": "Products",
                },
                "result": "passed",
            }
        ],
        current_url="https://www.saucedemo.com/checkout-complete.html",
        dom_context={
            "meta": {
                "title": "Swag Labs",
                "url": "https://www.saucedemo.com/checkout-complete.html",
            },
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    )

    assert len(unmet) == 1
    assert "Products" in unmet[0]


def test_agent_case_numeric_checkpoint_accepts_assertion_history():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": ["购物车角标数量为 1"],
            "final": [],
        },
        history=[
            {
                "step": {
                    "action": "assert_text",
                    "selector": "shopping_cart_badge",
                    "value": "1",
                },
                "result": "passed",
            }
        ],
        current_url="https://www.saucedemo.com/cart.html",
        dom_context={
            "meta": {
                "title": "Swag Labs",
                "url": "https://www.saucedemo.com/cart.html",
            },
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    )

    assert unmet == []


def test_agent_case_completion_extracts_business_page_term():
    unmet = _unmet_completion_criteria(
        criteria={"final": ["current page remains on catalog review page"]},
        history=[],
        current_url="https://example.test/catalogReviewList",
        dom_context={
            "meta": {"title": "Catalog Review - Admin"},
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    )

    assert unmet == []


def test_agent_case_completion_final_overrides_checkpoints():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": ["search button executed"],
            "final": ["current page remains on catalog review page"],
        },
        history=[],
        current_url="https://example.test/catalogReviewList",
        dom_context={
            "meta": {"title": "Catalog Review - Admin"},
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    )

    assert unmet == []


def test_agent_case_completion_uses_module_runtime_steps_as_evidence():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": ["查询按钮已执行"],
            "final": ["当前页面仍在商品审核业务页"],
        },
        history=[
            {
                "step": {
                    "use_module": "admin_search_and_view_log",
                    "_module_executed_steps": [
                        {
                            "action": "fill",
                            "selector": "admin_product_id_input",
                            "value": "DEMO-SKU-001",
                        },
                        {
                            "action": "click",
                            "selector": "admin_search_btn",
                        },
                        {
                            "action": "click",
                            "selector": "admin_view_log_btn",
                        },
                    ],
                },
                "result": "passed",
            },
        ],
        current_url="https://example.test/catalogReviewList",
        dom_context={"meta": {"title": "商品审核 - 新零售运营后台"}},
    )

    assert unmet == []


def test_agent_case_plan_cache_does_not_store_module_runtime_steps():
    cached_steps = _cacheable_plan_steps(
        [
            {
                "use_module": "admin_search_and_view_log",
                "_module_executed_steps": [
                    {"action": "click", "selector": "admin_search_btn"}
                ],
            }
        ]
    )

    assert cached_steps == [{"use_module": "admin_search_and_view_log"}]


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
            "use_module": "admin_search_and_view_log",
            "_module_executed_steps": [
                {
                    "action": "click",
                    "selector": "admin_search_btn",
                    "_action_executed_selector": 'button:has-text("查 询")',
                }
            ],
        },
        source="compiled_agent_case",
    )

    assert item["step"]["_module_executed_steps"] == [
        {
            "action": "click",
            "selector": "admin_search_btn",
            "_action_executed_selector": 'button:has-text("查 询")',
        }
    ]


def test_agent_case_completion_uses_action_history_for_input_and_click():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": [
                "page title contains Catalog Review",
                "product id field filled DEMO-SKU-001",
                "search button executed",
            ],
            "final": ["current page remains on catalog review page"],
        },
        history=[
            {
                "step": {
                    "action": "fill",
                    "selector": "#catalog-review-search_sku",
                    "value": "DEMO-SKU-001",
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
        current_url="https://example.test/catalogReviewList",
        dom_context={
            "meta": {"title": "Catalog Review - Admin"},
            "interactive_elements": [],
            "assertion_candidates": [],
        },
    )

    assert unmet == []


def test_agent_case_completion_accepts_resolved_fill_value_for_cjk_input():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": ["商品编号输入框已输入 DEMO-SKU-001"],
            "final": [],
        },
        history=[
            {
                "step": {
                    "action": "fill",
                    "selector": "admin_product_id_input",
                    "value": "${product_id}",
                    "_resolved_selector": "#catalog-review-search_sku",
                    "_resolved_value": "DEMO-SKU-001",
                },
                "result": "passed",
            },
        ],
        current_url="https://example.test/catalogReviewList",
        dom_context={"meta": {"title": "Catalog Review"}},
    )

    assert unmet == []


def test_agent_case_completion_accepts_value_assertion_as_input_evidence():
    unmet = _unmet_completion_criteria(
        criteria={
            "checkpoints": ["商品编号输入框已输入 DEMO-SKU-001"],
            "final": [],
        },
        history=[
            {
                "step": {
                    "action": "assert_value",
                    "selector": "admin_product_id_input",
                    "value": "DEMO-SKU-001",
                    "_resolved_selector": "#catalog-review-search_sku",
                },
                "result": "passed",
            },
        ],
        current_url="https://example.test/catalogReviewList",
        dom_context={"meta": {"title": "Catalog Review"}},
    )

    assert unmet == []


def test_agent_case_intent_requirements_prevent_early_finish_before_click_log():
    spec = {
        "intent": "fill product id DEMO-SKU-001, click search, then click view log",
        "steps": [],
    }
    history = [
        {
            "step": {
                "action": "fill",
                "selector": "#catalog-review-search_sku",
                "value": "DEMO-SKU-001",
            },
            "result": "passed",
        },
        {
            "step": {
                "action": "click",
                "selector": 'button:has-text("Search")',
                "_action_before_url": "https://example.test/catalogReviewList",
                "_action_after_url": "https://example.test/catalogReviewList",
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
        "intent": "fill product id DEMO-SKU-001, click search, then click view log",
        "steps": [],
    }
    history = [
        {
            "step": {
                "action": "fill",
                "selector": "#catalog-review-search_sku",
                "value": "DEMO-SKU-001",
            },
            "result": "passed",
        },
        {
            "step": {
                "action": "click",
                "selector": 'button:has-text("Search")',
                "_action_before_url": "https://example.test/catalogReviewList",
                "_action_after_url": "https://example.test/catalogReviewList",
                "_action_dom_changed": True,
            },
            "result": "passed",
        },
        {
            "step": {
                "action": "click",
                "selector": 'a:has-text("View Log")',
                "_action_before_url": "https://example.test/catalogReviewList",
                "_action_after_url": "https://example.test/catalogReviewLog",
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
            "fill product id DEMO-SKU-001",
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
                "selector": "#catalog-review-search_sku",
                "value": "DEMO-SKU-001",
            },
            "result": "passed",
        },
        {
            "step": {
                "action": "click",
                "selector": 'button:has-text("Search")',
                "_action_before_url": "https://example.test/catalogReviewList",
                "_action_after_url": "https://example.test/catalogReviewList",
                "_action_dom_changed": True,
            },
            "result": "passed",
        },
        {
            "step": {
                "action": "click",
                "selector": 'a:has-text("View Log")',
                "_action_before_url": "https://example.test/catalogReviewList",
                "_action_after_url": "https://example.test/catalogReviewLog",
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
                    "selector": "#catalog-review-search_sku",
                    "value": "DEMO-SKU-001",
                }
            },
            ensure_ascii=False,
        ),
        spec={"inputs": {}},
        dom_context={},
    )

    assert decision.action == "fill"
    assert decision.selector == "#catalog-review-search_sku"
    assert decision.value == "DEMO-SKU-001"


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
        {"action": "assert_title_contains", "value": "目录审核"}
    )

    assert decision.action == "assert_title_contains"
    assert decision.value == "目录审核"


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
        _first_url("打开 https://example.test/search?q=demo；搜索框输入demo")
        == "https://example.test/search?q=demo"
    )
    assert _first_url("open https://example.test/path?q=1, then click") == (
        "https://example.test/path?q=1"
    )
    assert _first_url("visit https://example.test/path.") == (
        "https://example.test/path"
    )


def test_url_contains_accepts_percent_encoded_actual_url():
    assert _url_contains(
        "https://example.test/results?q=%E6%BC%94%E7%A4%BA",
        "q=\u6f14\u793a",
    )
    assert not _url_contains(
        "https://example.test/results?q=%E6%BC%94%E7%A4%BA",
        "q=\u5176\u4ed6",
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
                "text": "Target destination",
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
            "intent": "open target destination and complete workflow",
            "steps": [],
            "inputs": {},
            "criteria": {"final": ["target destination is visible"]},
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
                "name": "Target destination",
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
                "reason": "open target destination",
            }
        ),
        spec={
            "description": "",
            "intent": "open target destination and complete workflow",
            "steps": [],
            "inputs": {},
            "criteria": {"final": ["target destination is visible"]},
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


def test_agent_case_execute_step_skips_new_page_probe_for_non_click(tmp_path: Path):
    class FakePage:
        url = "https://example.test/inventory"

        def locator(self, selector):
            raise AssertionError(f"unexpected locator probe: {selector}")

    class FakeStepExecutor:
        step_has_error = False
        smart_resolver = None

        def __init__(self):
            self.executed_step = None

        def execute_step(self, step):
            self.executed_step = dict(step)

    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://example.test/",
        elements={"page_title": ".title"},
        modules={},
        variables={},
        test_cases=[],
        test_data={},
    )
    executor = AgentCaseExecutor(
        page=FakePage(),
        ui_helper=object(),
        elements={"page_title": ".title"},
        context=context,
    )
    step_executor = FakeStepExecutor()
    executor.step_executor = step_executor

    executor._execute_step(
        {"action": "assert_visible", "selector": "page_title"},
        spec={"guardrails": {}},
    )

    assert step_executor.executed_step["selector"] == "page_title"


def test_agent_case_execute_step_resolves_project_selector_for_new_page_probe(
    tmp_path: Path,
):
    class FakeLocator:
        @property
        def first(self):
            return self

        def evaluate(self, script):
            return False

        def inner_text(self, timeout=None):
            return ""

    class FakeContext:
        def __init__(self):
            self.pages = []

    class FakePage:
        def __init__(self, context):
            self.url = "https://example.test/start"
            self.context = context
            self.locator_calls = []

        def locator(self, selector):
            self.locator_calls.append(selector)
            return FakeLocator()

        def title(self):
            return ""

    class FakeStepExecutor:
        step_has_error = False
        smart_resolver = None

        def __init__(self, page):
            self.page = page

        def execute_step(self, step):
            return None

    page_context = FakeContext()
    page = FakePage(page_context)
    page_context.pages.append(page)
    context = ProjectContext(
        project="demo",
        test_dir=tmp_path,
        base_url="https://example.test/",
        elements={"search_button": "#search"},
        modules={},
        variables={},
        test_cases=[],
        test_data={},
    )
    executor = AgentCaseExecutor(
        page=page,
        ui_helper=object(),
        elements={"search_button": "#search"},
        context=context,
    )
    executor.step_executor = FakeStepExecutor(page)

    executor._execute_step(
        {"action": "click", "selector": "search_button"},
        spec={"guardrails": {}},
    )

    assert "#search" in page.locator_calls
    assert "search_button" not in page.locator_calls


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
        url = "https://admin.example.test/#/login"

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
                {"action": "goto", "value": "https://admin.example.test"}
            ),
            spec={
                "intent": "visit url https://admin.example.test/#/login",
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
        "intent": "visit url https://example.test/login, use OA login, expand mall menu, click catalog review",
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
        "page_summary": {"visible_text_summary": ["Mall", "Catalog Review"]},
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
        "intent": "visit url https://example.test/login, use OA login, expand mall menu, click catalog review",
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
        "page_summary": {
            "visible_text_summary": ["Login", "OA account", "OA password"]
        },
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
        "intent": "fill product id DEMO-SKU-001, click search, click view log",
        "steps": [],
        "inputs": {"product_id": "DEMO-SKU-001"},
    }
    history = [
        {
            "step": {
                "action": "fill",
                "selector": "#catalog-review-search_sku",
                "value": "DEMO-SKU-001",
                "_resolved_value_after": "DEMO-SKU-001",
            },
            "result": "passed",
            "url_after": "https://example.test/catalogReviewList",
        },
        {
            "step": {
                "action": "click",
                "selector": 'button:has-text("Search")',
                "_action_before_url": "https://example.test/catalogReviewList",
                "_action_after_url": "https://example.test/catalogReviewList",
                "_action_dom_changed": True,
            },
            "result": "passed",
            "url_after": "https://example.test/catalogReviewList",
        },
    ]
    dom_context = {
        "meta": {
            "url": "https://example.test/catalogReviewList",
            "title": "Catalog Review",
        },
        "page_summary": {"visible_text_summary": ["Catalog Review"]},
        "forms": [],
        "interactive_elements": [],
        "assertion_candidates": [],
    }

    harness = _runtime_harness_state(
        spec=spec,
        history=history,
        dom_context=dom_context,
        current_url="https://example.test/catalogReviewList",
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
        "intent": "expand mall menu, click catalog review",
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
    assert any(item.startswith("intent click: catalog review") for item in unmet)


def test_agent_case_unmatched_click_without_observed_progress_does_not_satisfy_next_intent():
    spec = {
        "description": "",
        "intent": "expand mall menu, click catalog review",
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

    assert any(item.startswith("intent click: catalog review") for item in unmet)


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

    assert (
        _unmet_intent_action_requirements(
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
        == []
    )


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
                "_action_before_url": "https://example.test/catalogReviewList",
                "_action_after_url": "https://example.test/catalogReviewList",
                "_action_dom_changed": False,
            },
            "result": "passed",
            "url_after": "https://example.test/catalogReviewList",
        }
    ]

    assert (
        _unmet_intent_action_requirements(
            spec=spec,
            history=history,
            current_url="https://example.test/catalogReviewList",
            dom_context={
                "meta": {"url": "https://example.test/catalogReviewList"},
                "forms": [],
                "interactive_elements": [],
                "assertion_candidates": [],
            },
        )
        == []
    )


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
        "intent": "expand mall menu, click catalog review",
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
        "intent click: catalog review/catalog/review"
    ]


def test_agent_case_runtime_harness_advances_after_semantic_menu_click():
    spec = {
        "description": "",
        "intent": "expand mall menu, click catalog review, fill product id DEMO-SKU-001",
        "steps": [],
        "inputs": {"product_id": "DEMO-SKU-001"},
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
                "selector": 'li[title="Catalog Review"]',
                "_action_target_text": "Catalog Review",
                "_action_before_url": "https://example.test/home",
                "_action_after_url": "https://example.test/catalogReviewList",
            },
            "result": "passed",
            "url_after": "https://example.test/catalogReviewList",
        },
    ]
    dom_context = {
        "meta": {
            "url": "https://example.test/catalogReviewList",
            "title": "Catalog Review",
        },
        "forms": [
            {
                "id": "f1",
                "type": "input",
                "input_type": "text",
                "label": "Product ID",
                "selector_candidates": ["#catalog-review-search_sku"],
            }
        ],
        "interactive_elements": [],
        "assertion_candidates": [],
    }

    harness = _runtime_harness_state(
        spec=spec,
        history=history,
        dom_context=dom_context,
        current_url="https://example.test/catalogReviewList",
    )

    assert harness["phase"] == "fill product id DEMO-SKU-001"


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
                    "near_text": "Example App 登录 显示密码 修改密码 忘记密码",
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
        "meta": {"url": "https://example.test/catalogReviewList"},
        "page_summary": {"visible_text_summary": ["Catalog Review"]},
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
