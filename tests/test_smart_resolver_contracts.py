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
        "ai_playwright.ai_runtime.smart_resolver.load_ai_config",
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
        "ai_playwright.ai_runtime.smart_resolver.load_ai_config",
        lambda: {
            "selector_registry": {"enabled": False},
            "runtime": {"smart_selector_probe_timeout_ms": 1000},
            "native_observe": {"enabled": False},
        },
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.selector_matches_target",
        lambda page, selector, target, action: True,
    )
    verify_calls = []

    def fake_verify_selector(page, selector, *, action, timeout):
        verify_calls.append((selector, timeout))
        if selector == "input[name='password']":
            raise TimeoutError("yaml selector is stale")

    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.verify_selector",
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


def test_smart_resolver_uses_target_when_explicit_selector_is_stale(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.load_ai_config",
        lambda: {
            "selector_registry": {"enabled": False},
            "runtime": {"smart_selector_probe_timeout_ms": 1000},
            "native_observe": {"enabled": False},
        },
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.selector_matches_target",
        lambda page, selector, target, action, **kwargs: (
            selector == "#new-login" and target == "login button"
        ),
    )
    verify_calls = []

    def fake_verify_selector(page, selector, *, action, timeout):
        verify_calls.append((selector, timeout))
        if selector == "#old-login":
            raise TimeoutError("yaml selector is stale")

    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.verify_selector",
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
        selector="#new-login",
        source="heuristic",
        confidence=0.9,
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = registry
    resolved = resolver.resolve(
        action="click",
        target="login button",
        selector="#old-login",
        mode="smart",
        timeout=10000,
    )

    assert resolved.selector == "#new-login"
    assert resolved.source == "registry"
    assert resolved.healed is True
    assert resolved.healing_attempted is True
    assert resolved.original_selector == "#old-login"
    assert resolved.cache_target == "login button"
    assert verify_calls == [("#old-login", 1000), ("#new-login", 1000)]


def test_smart_resolver_target_only_can_use_verified_registry_first(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.load_ai_config",
        lambda: {
            "selector_registry": {"enabled": False},
            "runtime": {"smart_selector_probe_timeout_ms": 1000},
            "native_observe": {"enabled": False},
        },
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.selector_matches_target",
        lambda page, selector, target, action: True,
    )
    verify_calls = []

    def fake_verify_selector(page, selector, *, action, timeout):
        verify_calls.append((selector, timeout))

    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.verify_selector",
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
        "ai_playwright.ai_runtime.smart_resolver.load_ai_config",
        lambda: {
            "selector_registry": {"enabled": False},
            "runtime": {
                "ai_enabled": False,
                "smart_selector_probe_timeout_ms": 750,
            },
            "native_observe": {"enabled": False},
        },
    )
    verify_calls = []

    def fake_verify_selector(page, selector, *, action, timeout):
        verify_calls.append((selector, timeout))
        raise TimeoutError("not visible")

    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.verify_selector",
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


def test_smart_resolver_limits_low_quality_self_heal_probe_candidates(monkeypatch):
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.load_ai_config",
        lambda: {
            "selector_registry": {"enabled": False},
            "runtime": {
                "ai_enabled": False,
                "self_heal_probe_limit": 3,
                "smart_selector_probe_timeout_ms": 10,
            },
            "native_observe": {"enabled": False},
        },
    )
    heuristic_candidates = [
        f'input[data-testid*="password-{index}" i]' for index in range(20)
    ]
    verify_calls: list[str] = []

    def fake_verify_selector(page, selector, *, action, timeout):
        verify_calls.append(selector)
        raise ValueError("not visible")

    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.verify_selector",
        fake_verify_selector,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.heuristic_selectors",
        lambda target, action: heuristic_candidates,
    )

    class FakePage:
        url = "https://example.test/login"

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None

    with pytest.raises(ValueError, match="AI定位未启用"):
        resolver.resolve(
            action="fill",
            target="password field",
            selector="#old-password",
            mode="smart",
            timeout=1000,
        )

    assert verify_calls == ["#old-password", *heuristic_candidates[:3]]


def test_smart_resolver_prioritizes_strong_self_heal_probe_selector(monkeypatch):
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.load_ai_config",
        lambda: {
            "selector_registry": {"enabled": False},
            "runtime": {
                "ai_enabled": False,
                "self_heal_probe_limit": 1,
                "smart_selector_probe_timeout_ms": 10,
            },
            "native_observe": {"enabled": False},
        },
    )
    verify_calls: list[str] = []
    strong_selector = 'button[data-test="save"]'

    def fake_verify_selector(page, selector, *, action, timeout):
        verify_calls.append(selector)
        if selector != strong_selector:
            raise ValueError("not visible")
        return True

    class FakeLocator:
        def __init__(self, selector):
            self.selector = selector

    class FakePage:
        url = "https://example.test/form"

        def locator(self, selector):
            return FakeLocator(selector)

    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.verify_selector",
        fake_verify_selector,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.heuristic_selectors",
        lambda target, action: [
            "main > section > div:nth-of-type(2) > button",
            strong_selector,
        ],
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.stable_selector_for_locator",
        lambda locator: locator.selector,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.selector_matches_target",
        lambda page, selector, target, action: True,
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None

    resolved = resolver.resolve(
        action="click",
        target="save button",
        selector="#old-save",
        mode="smart",
        timeout=1000,
    )

    assert resolved.selector == strong_selector
    assert verify_calls == ["#old-save", strong_selector]


def test_smart_resolver_uses_ai_when_concrete_text_is_not_visible(monkeypatch):
    ai_called = False

    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.load_ai_config",
        lambda: {
            "selector_registry": {"enabled": False},
            "runtime": {
                "ai_enabled": True,
                "allow_ai_in_smart": True,
                "smart_selector_probe_timeout_ms": 10,
            },
            "native_observe": {"enabled": False},
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
        "ai_playwright.ai_runtime.smart_resolver.verify_selector",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("not visible")),
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.heuristic_selectors",
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
        target="text=目录审核",
        selector="text=目录审核",
        mode="smart",
        timeout=1000,
    )

    assert resolved.selector == "#query"
    assert ai_called is True


def test_smart_resolver_fails_when_concrete_text_is_not_visible_without_ai(
    monkeypatch,
):
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.load_ai_config",
        lambda: {
            "selector_registry": {"enabled": False},
            "runtime": {
                "ai_enabled": False,
                "allow_ai_in_smart": False,
                "smart_selector_probe_timeout_ms": 10,
            },
            "native_observe": {"enabled": False},
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
        "ai_playwright.ai_runtime.smart_resolver.verify_selector",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("not visible")),
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.heuristic_selectors",
        lambda target, action: [],
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None

    with pytest.raises(ValueError, match="target text is not visible"):
        resolver.resolve(
            action="click",
            target="text=目录审核",
            selector="text=目录审核",
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
                "text": "移动 App 下载",
                "title": "移动 App 下载",
                "role": "link",
            }

    class FakePage:
        def locator(self, selector):
            return FakeLocator()

    assert (
        selector_matches_target(
            FakePage(), 'a:has-text("移动 App 下载")', "text=商城", "click"
        )
        is False
    )
    assert (
        selector_matches_target(
            FakePage(), "#username", "li.menu-item:has-text('商城')", "click"
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
                "id": "catalog-review-search_brandid",
                "name": "brandId",
                "placeholder": "",
                "label": "品牌",
            }

    class FakePage:
        def locator(self, selector):
            return FakeLocator()

    assert (
        selector_matches_target(
            FakePage(),
            "#catalog-review-search_brandid",
            "text=输入商品编号/商品名称",
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
                "id": "catalog-review-search_productid",
                "placeholder": "输入商品编号/商品名称",
            }

    class FakePage:
        def locator(self, selector):
            return FakeLocator()

    assert (
        selector_matches_target(
            FakePage(),
            "#catalog-review-search_productid",
            "text=输入商品编号/商品名称",
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


def test_selector_matches_target_accepts_logout_sidebar_link_for_cjk_target():
    class FakeLocator:
        @property
        def first(self):
            return self

        def evaluate(self, script):
            return {
                "tag": "a",
                "id": "logout_sidebar_link",
                "data_test": "logout-sidebar-link",
                "text": "Logout",
                "role": "link",
            }

    class FakePage:
        def locator(self, selector):
            return FakeLocator()

    assert (
        selector_matches_target(
            FakePage(),
            "#logout_sidebar_link",
            "侧边栏退出登录按钮",
            "assert_visible",
        )
        is True
    )
    assert (
        selector_matches_target(
            FakePage(),
            'a[data-test="logout-sidebar-link"]',
            "侧边栏退出登录按钮",
            "click",
        )
        is True
    )


def test_selector_matches_target_relaxed_ai_accepts_action_object_text():
    class FakeLocator:
        def __init__(self, selector):
            self.selector = selector

        @property
        def first(self):
            return self

        def evaluate(self, script):
            text = (
                "\u65e5\u5fd7"
                if "\u65e5\u5fd7" in self.selector
                else "\u5546\u54c1\u5ba1\u6838"
            )
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
        "ai_playwright.ai_runtime.smart_resolver.ChatCompletionProvider",
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
        "ai_playwright.ai_runtime.smart_resolver.ChatCompletionProvider",
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


def test_smart_resolver_ai_semantic_validation_accepts_unknown_synonym(
    monkeypatch,
):
    provider_calls: list[str] = []

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

        def evaluate(self, script):
            return {
                "tag": "button",
                "data_test": "continue-shopping",
                "text": "Continue Shopping",
                "role": "button",
            }

    class FakePage:
        url = "https://example.test/cart"

        def locator(self, selector):
            return FakeLocator()

        def title(self):
            return "Cart"

    class SemanticValidationProvider:
        settings = type("Settings", (), {"model": "test-model"})()

        def complete_model(self, messages, response_model, **kwargs):
            provider_calls.append(response_model.__name__)
            if response_model is SelectorDecision:
                return response_model.model_validate(
                    {"element_id": "e0", "confidence": 0.91}
                )
            return response_model.model_validate(
                {
                    "status": "match",
                    "confidence": 0.86,
                    "reason": "Continue Shopping 等价于继续购物",
                }
            )

    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.ChatCompletionProvider",
        SemanticValidationProvider,
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None
    resolver._collect_candidates = lambda *, limit, respect_config_max=True: [
        {
            "index": 0,
            "tag": "button",
            "selector": 'button[data-test="continue-shopping"]',
            "data_test": "continue-shopping",
            "text": "Continue Shopping",
            "role": "button",
            "visible": True,
            "enabled": True,
        }
    ]

    resolved = resolver._resolve_with_ai(
        action="click",
        target="继续购物按钮",
        timeout=1000,
    )

    assert resolved.selector == 'button[data-test="continue-shopping"]'
    assert resolved.semantic_ai_validated is True
    assert provider_calls == [
        "SelectorDecision",
        "SelectorSemanticValidationDecision",
    ]


def test_smart_resolver_ai_semantic_validation_does_not_override_risky_mismatch(
    monkeypatch,
):
    provider_calls: list[str] = []

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

        def evaluate(self, script):
            return {"tag": "a", "id": "logout", "text": "Logout", "role": "link"}

    class FakePage:
        url = "https://example.test/home"

        def locator(self, selector):
            return FakeLocator()

        def title(self):
            return "Home"

    class RiskyProvider:
        settings = type("Settings", (), {"model": "test-model"})()

        def complete_model(self, messages, response_model, **kwargs):
            provider_calls.append(response_model.__name__)
            if response_model is SelectorDecision:
                return response_model.model_validate(
                    {"element_id": "e0", "confidence": 0.91}
                )
            raise AssertionError("risky mismatch must not call semantic validator")

    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.ChatCompletionProvider",
        RiskyProvider,
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None
    resolver._collect_candidates = lambda *, limit, respect_config_max=True: [
        {
            "index": 0,
            "tag": "a",
            "selector": "#logout",
            "id": "logout",
            "text": "Logout",
            "role": "link",
            "visible": True,
            "enabled": True,
        }
    ]

    with pytest.raises(ValueError, match="AI selector semantic mismatch"):
        resolver._resolve_with_ai(action="click", target="text=Search", timeout=1000)

    assert provider_calls == ["SelectorDecision"]


def test_smart_resolver_uses_role_text_candidate_from_structural_menu_source(
    monkeypatch,
):
    captured: dict[str, Any] = {}
    monkeypatch.setenv("LLM_DATA_POLICY", "trusted_local")

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
        "ai_playwright.ai_runtime.smart_resolver.ChatCompletionProvider",
        ElementIdProvider,
    )

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None

    resolved = resolver._resolve_with_ai(
        action="click", target="text=Mall", timeout=1000
    )

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
            raise AssertionError(
                f"low-quality selector should not be verified: {selector}"
            )

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
        "ai_playwright.ai_runtime.smart_resolver.ChatCompletionProvider",
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
        data_policy="trusted_local",
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
        "ai_playwright.ai_runtime.smart_resolver.load_ai_config",
        lambda: {
            "native_observe": {
                "max_candidates": 7,
                "include_open_shadow_dom": False,
                "ignore_selectors": [".noise"],
            }
        },
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.collect_candidates",
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
        "ai_playwright.ai_runtime.smart_resolver.semantic_selectors",
        fake_semantic_selectors,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.heuristic_selectors",
        lambda target, action: [],
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.verify_selector",
        lambda page, selector, action, timeout: True,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.smart_resolver.selector_matches_target",
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
            assert kwargs["target"] == "login button"
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
        "ai_playwright.step_actions.step_executor.execute_action_with_command",
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
    assert cache_events[0]["target"] == "login button"


def test_step_executor_does_not_call_element_store_when_healed_selector_verified(
    monkeypatch,
):
    calls: list[tuple[str, str, str | None]] = []

    class FakeResolver:
        def resolve(self, **kwargs):
            assert kwargs["target"] == "login button"
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
        "ai_playwright.step_actions.step_executor.execute_action_with_command",
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


def test_step_executor_generation_records_verified_element_update(
    monkeypatch,
    tmp_path: Path,
):
    project_dir = tmp_path / "demo"
    elements_dir = project_dir / "elements"
    elements_dir.mkdir(parents=True)
    elements_file = elements_dir / "login.yaml"
    elements_file.write_text(
        "elements:\n  login_button: button:has-text('\u767b\u5f55')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_DIR", str(project_dir))
    monkeypatch.setenv("UI_GENERATION_PERSIST_VERIFIED_HEALS", "1")
    calls: list[tuple[str, str, str | None]] = []

    class FakeResolver:
        def resolve(self, **kwargs):
            return ResolvedSelector(
                selector='button:has-text("\u767b \u5f55")',
                source="heuristic",
                healed=True,
                healing_attempted=True,
                original_selector="button:has-text('\u767b\u5f55')",
                original_error="not found",
                confidence=0.0,
                cache_action="click",
                cache_target=kwargs["target"],
                cache_page_key="https://example.test/login",
            )

        def record_verified_selector(self, **kwargs):
            pass

    class FakePage:
        url = "https://example.test/login"

    class FakeUiHelper:
        pass

    def fake_execute_action_with_command(ui_helper, action, selector, value, step):
        calls.append((action, selector, value))

    monkeypatch.setattr(
        "ai_playwright.step_actions.step_executor.execute_action_with_command",
        fake_execute_action_with_command,
    )

    step_executor_module.discard_pending_selector_cache("test setup")
    step_executor_module.pop_persisted_selector_updates()
    executor = StepExecutor(
        FakePage(),
        FakeUiHelper(),
        elements={"login_button": "button:has-text('\u767b\u5f55')"},
    )
    executor.smart_resolver = FakeResolver()

    executor.execute_step(
        {
            "action": "click",
            "selector": "login_button",
            "target": "\u767b\u5f55\u6309\u94ae",
            "mode": "smart",
        }
    )
    step_executor_module.commit_pending_selector_cache()
    updates = step_executor_module.pop_persisted_selector_updates()

    assert calls == [("click", 'button:has-text("\u767b \u5f55")', None)]
    assert updates[0]["source_key"] == "login_button"
    assert updates[0]["persisted_key"] == "login_button"
    assert updates[0]["selector"] == "//button[normalize-space()='\u767b \u5f55']"
    assert (
        "login_button: //button[normalize-space()='\u767b \u5f55']"
        in elements_file.read_text(encoding="utf-8")
    )


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
        selector.startswith("//span") and "normalize-space()='商城'" in selector
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
        "ai_playwright.ai_runtime.playwright_selectors.collect_candidates",
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
        "ai_playwright.ai_runtime.agent_case_executor.semantic_selectors",
        lambda *args, **kwargs: ['button:has-text("Search")'],
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.agent_case_executor.heuristic_selectors",
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


def test_selector_decision_normalizes_common_status_aliases():
    decision = SelectorDecision.model_validate(
        {"status": "success", "element_id": "e1", "confidence": 0.9}
    )

    assert decision.status == "ok"


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
