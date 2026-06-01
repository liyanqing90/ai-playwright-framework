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
    _configure_runtime_for_verification,
    _default_output_name,
    _has_explicit_steps,
    _payload_from_explicit_spec,
    _payload_with_referenced_context_modules,
    _result_paths,
    _resolve_navigation_context,
    _verification_pytest_args,
    _validate_spec_project_scope,
    _write_payload,
    generate_case_files,
    resolve_generation_spec_path,
)
from ai_playwright.ai_generation.harness import GenerationHarness
from ai_playwright.ai_generation.harness import _safe_case_name
from ai_playwright.ai_generation.pipeline import execute_compiled_payload_steps
from ai_playwright.ai_generation.project_context import (
    ProjectContext,
    load_project_context,
)
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

    with pytest.raises(ValueError, match="test_data/admin/generation"):
        _validate_spec_project_scope(
            project="demo",
            spec_path=Path("test_data/admin/generation/smoke.yaml"),
            spec={},
        )

    with pytest.raises(ValueError, match="project=admin"):
        _validate_spec_project_scope(
            project="demo",
            spec_path=Path("test_data/demo/generation/smoke.yaml"),
            spec={"project": "admin"},
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


def test_generation_spec_error_lists_available_specs(tmp_path: Path):
    spec_dir = tmp_path / "test_data" / "demo" / "generation"
    spec_dir.mkdir(parents=True)
    (spec_dir / "saucedemo_ai_intent.yaml").write_text("cases: []\n", encoding="utf-8")

    class Context:
        test_dir = tmp_path / "test_data" / "demo"

    with pytest.raises(FileNotFoundError, match="可用规格: saucedemo_ai_intent"):
        resolve_generation_spec_path(Context(), "missing_spec")


def test_generation_spec_does_not_normalize_unicode_dash(tmp_path: Path):
    spec_dir = tmp_path / "test_data" / "demo" / "generation"
    spec_dir.mkdir(parents=True)
    (spec_dir / "saucedemo_ai_intent.yaml").write_text("cases: []\n", encoding="utf-8")

    class Context:
        test_dir = tmp_path / "test_data" / "demo"

    with pytest.raises(FileNotFoundError, match="saucedemo_ai—intent"):
        resolve_generation_spec_path(Context(), "saucedemo_ai—intent")


def test_generation_project_context_honors_test_dir_env(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    explicit_test_dir = tmp_path / "external_test_data" / "demo"
    explicit_test_dir.mkdir(parents=True)
    (config_dir / "env_config.yaml").write_text(
        "projects:\n"
        "  demo:\n"
        "    test_dir: should_not_use\n"
        "    environments:\n"
        "      prod: https://example.test/\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_PLAYWRIGHT_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("TEST_DIR", str(explicit_test_dir))

    context = load_project_context("demo", env="prod")

    assert context.test_dir == explicit_test_dir.resolve()


def test_generation_spec_string_steps_still_use_ai():
    natural_spec = {
        "cases": [
            {
                "name": "example_search_keyword",
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
                "name": "example_search_keyword",
                "steps": [{"action": "goto", "value": "https://example.test"}],
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
        "ai_playwright.ai_generation.case_generator.load_ai_config",
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
        "ai_playwright.ai_generation.case_generator.ChatCompletionProvider",
        FakeProvider,
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


def test_generate_case_always_verifies_candidate_before_formal_persist(
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
        "ai_playwright.ai_generation.case_generator.load_project_context",
        lambda project, env="prod": context,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_generation.case_generator.load_ai_config",
        lambda: {
            "generation": {
                "verify_after_generate": False,
                "runtime_repair_attempts": 1,
            }
        },
    )
    monkeypatch.setattr(
        "ai_playwright.ai_generation.case_generator._build_payload",
        lambda *args, **kwargs: payload,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_generation.case_generator._normalize_validate_payload",
        lambda **kwargs: (payload, []),
    )
    monkeypatch.setattr(
        "ai_playwright.ai_generation.case_generator._write_and_verify_candidate",
        lambda **kwargs: events.append("candidate_verify"),
    )

    def fake_verify_generated_case(*, stage="生成", **kwargs):
        events.append(f"verify:{stage}")

    def fake_write_payload(result, *, overwrite, verify=None, post_verify=None):
        events.append("formal_write")
        if post_verify:
            post_verify()

    monkeypatch.setattr(
        "ai_playwright.ai_generation.case_generator._verify_generated_case",
        fake_verify_generated_case,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_generation.case_generator._write_payload",
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


def test_generate_case_rejects_preview_or_unverified_modes():
    with pytest.raises(ValueError, match="dry_run"):
        generate_case_files(project="demo", spec_path="spec.yaml", dry_run=True)

    with pytest.raises(ValueError, match="verify=False"):
        generate_case_files(project="demo", spec_path="spec.yaml", verify=False)


def test_generation_verification_loads_framework_pytest_plugin(tmp_path: Path):
    args = _verification_pytest_args(tmp_path / "cases" / "generated.yaml")

    plugin_index = args.index("ai_playwright.pytest_plugin")

    assert args[plugin_index - 1] == "-p"
    assert "--skip-yaml-schema" in args
    assert plugin_index < args.index("--skip-yaml-schema")


def test_generation_verification_defaults_to_headed_browser(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv("PWHEADED", raising=False)

    args = _verification_pytest_args(tmp_path / "cases" / "generated.yaml")

    assert "--browser" in args
    assert args[args.index("--browser") + 1] == "chromium"
    assert "--headed" in args


def test_generation_verification_can_run_headless_from_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PWHEADED", "0")

    args = _verification_pytest_args(tmp_path / "cases" / "generated.yaml")

    assert "--headed" not in args


def test_generation_runtime_config_keeps_headless_verification_env(
    monkeypatch, tmp_path: Path
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
    monkeypatch.setenv("PWHEADED", "0")
    monkeypatch.setenv("PWSLOWMO", "125")
    monkeypatch.setenv("BROWSER", "chromium")

    _configure_runtime_for_verification(context=context, env="prod")

    assert os.environ["PWHEADED"] == "0"
    assert os.environ["PWSLOWMO"] == "125"


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
        "ai_playwright.ai_generation.case_generator.load_project_context",
        lambda project, env="prod": context,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_generation.case_generator.load_ai_config",
        lambda: {"generation": {}},
    )
    monkeypatch.setattr(
        "ai_playwright.ai_generation.case_generator._build_payload",
        lambda *args, **kwargs: payload,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_generation.case_generator._normalize_validate_payload",
        lambda **kwargs: (payload, []),
    )

    def fail_candidate(**kwargs):
        events.append("candidate_verify")
        raise AssertionError("candidate failed")

    def fail_if_written(*args, **kwargs):
        raise AssertionError("formal write must not happen")

    monkeypatch.setattr(
        "ai_playwright.ai_generation.case_generator._write_and_verify_candidate",
        fail_candidate,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_generation.case_generator._write_payload",
        fail_if_written,
    )
    monkeypatch.setattr(
        "ai_playwright.ai_generation.case_generator._repair_payload_with_ai",
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


def test_generation_harness_rejects_unknown_selector_key(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"page_title": ".title"},
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
                        {"action": "click", "selector": "shopping_cart_badge"},
                        {"action": "assert_visible", "selector": "page_title"},
                    ],
                }
            },
        }
    )

    with pytest.raises(ValueError, match="selector 未在 elements"):
        harness.validate(payload)


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
        output_name="admin/order/create",
    )

    payload = harness.normalize(
        {
            "cases": [{"name": "test_generated"}],
            "data": {
                "test_generated": {
                    "mode": "smart",
                    "steps": [
                        {"action": "assert_title_contains", "value": "Catalog Review"},
                        {"action": "fill", "selector": "search_input", "value": "x"},
                        {"action": "click", "target": "search_input"},
                    ],
                }
            },
            "elements": {"search_input": "#generated-search"},
        }
    )

    assert payload["elements"] == {"search_input_Catalog_Review": "#generated-search"}
    steps = payload["data"]["test_generated"]["steps"]
    assert steps[1]["selector"] == "search_input_Catalog_Review"
    assert steps[2] == {
        "action": "click",
        "selector": "search_input_Catalog_Review",
        "target": "search input Catalog Review",
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
    spec_file = context.test_dir / "generation" / "admin" / "order" / "create.yaml"
    payload = {
        "cases": [{"name": "test_generated"}],
        "data": {"test_generated": {"steps": []}},
        "elements": {"confirm_button": "#confirm"},
        "modules": {
            "confirm_flow": [{"action": "click", "selector": "confirm_button"}]
        },
        "vars": {"username": "qa"},
    }

    output_name = _default_output_name(spec_file, context=context)
    result = _result_paths(context, payload, output_name=output_name)

    assert output_name == "admin/order/create"
    assert result["case_file"] == context.test_dir / "cases" / "admin/order/create.yaml"
    assert result["data_file"] == context.test_dir / "data" / "admin/order/create.yaml"
    assert (
        result["elements_file"]
        == context.test_dir / "elements" / "admin/order/create.yaml"
    )
    assert (
        result["modules_file"]
        == context.test_dir / "modules" / "admin/order/create.yaml"
    )
    assert result["vars_file"] == context.test_dir / "vars" / "admin/order/create.yaml"


def test_generation_write_payload_merges_existing_vars_file(tmp_path: Path):
    vars_file = tmp_path / "vars" / "generated.yaml"
    vars_file.parent.mkdir(parents=True)
    vars_file.write_text("username: qa\npassword: secret\n", encoding="utf-8")
    result = {
        "payload": {
            "cases": [{"name": "test_generated"}],
            "data": {"test_generated": {"steps": []}},
            "vars": {"product_id": "DEMO-SKU-123"},
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
        "product_id": "DEMO-SKU-123",
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
        output_name="admin/order/create",
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
        output_name="admin/order/cancel",
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


def test_generation_harness_maps_title_assertion_to_visible_title_element(
    tmp_path: Path,
):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="https://example.test/",
            elements={"page_title": ".title"},
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
                        {"action": "assert_title", "value": "Products"},
                        {"action": "assert_title_contains", "value": "Cart"},
                    ],
                }
            },
        }
    )

    assert payload["data"]["test_generated"]["steps"] == [
        {
            "action": "assert_text",
            "selector": "page_title",
            "target": "page title",
            "value": "Products",
        },
        {
            "action": "assert_text_contains",
            "selector": "page_title",
            "target": "page title",
            "value": "Cart",
        },
    ]
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
        "target": "known search input",
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
            "cases": [{"name": "test_generated"}, {"name": "test_generated_again"}],
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
                },
                "test_generated_again": {
                    "mode": "smart",
                    "steps": [
                        {
                            "use_module": "login",
                            "params": {"username": "${standard_username}"},
                        },
                        {"action": "assert_visible", "selector": "title"},
                    ],
                },
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


def test_generation_harness_rejects_composite_context_module_reuse(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"title": ".title"},
            modules={
                "login_and_open_catalog": [
                    {"action": "goto", "value": "https://example.test/login"},
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
                        {"use_module": "login_and_open_catalog"},
                        {"action": "assert_visible", "selector": "title"},
                    ],
                }
            },
        }
    )

    with pytest.raises(ValueError, match="module颗粒度过粗"):
        harness.validate(payload)


def test_generation_harness_rejects_generated_composite_module_assets(
    tmp_path: Path,
):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"title": ".title"},
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
                        {"use_module": "search_and_view_log"},
                        {"action": "assert_visible", "selector": "title"},
                    ],
                },
            },
            "modules": {
                "search_and_view_log": [
                    {"action": "click", "selector": "title", "target": "title"},
                ]
            },
        }
    )

    with pytest.raises(ValueError, match="颗粒度过粗"):
        harness.validate(payload)


def test_generation_harness_rejects_generated_single_step_module_assets(
    tmp_path: Path,
):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"log_link": "a.log"},
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
                        {"use_module": "open_log"},
                        {"action": "assert_visible", "selector": "log_link"},
                    ],
                }
            },
            "modules": {
                "open_log": [
                    {"action": "click", "selector": "log_link", "target": "日志入口"},
                ]
            },
        }
    )

    with pytest.raises(ValueError, match="single-step module"):
        harness.validate(payload)


def test_generation_harness_rejects_generated_module_without_reuse_value(
    tmp_path: Path,
):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={
                "username": "#user-name",
                "login_btn": "#login",
                "title": ".title",
            },
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
                            "use_module": "login",
                            "params": {"username": "${standard_username}"},
                        },
                        {"action": "assert_visible", "selector": "title"},
                    ],
                },
                "test_generated_again": {
                    "mode": "smart",
                    "steps": [
                        {
                            "use_module": "login",
                            "params": {"username": "${standard_username}"},
                        },
                        {"action": "assert_visible", "selector": "title"},
                    ],
                },
            },
            "modules": {
                "login": [
                    {"action": "fill", "selector": "username", "value": "${username}"},
                    {"action": "click", "selector": "login_btn", "target": "登录按钮"},
                ]
            },
        }
    )

    with pytest.raises(ValueError, match="actual reuse value"):
        harness.validate(payload)


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
                },
                "test_generated_again": {
                    "mode": "smart",
                    "steps": [
                        {
                            "use_module": "login",
                            "params": {"username": "${standard_username}"},
                        },
                        {"action": "assert_visible", "selector": "title"},
                    ],
                },
            },
        }
    )

    assert payload["data"]["test_generated"]["steps"][0]["params"] == {
        "username": "${username}",
        "password": "${password}",
    }


def test_generation_harness_infers_single_module_for_module_action(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"title": ".title"},
            modules={
                "login": [
                    {"action": "fill", "selector": "title", "value": "${username}"},
                ]
            },
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec={"mode": "smart", "inputs": {"username": "qa_user"}},
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
                            "params": {"username": "qa_user"},
                        },
                        {"action": "assert_visible", "selector": "title"},
                    ],
                },
                "test_generated_again": {
                    "mode": "smart",
                    "steps": [
                        {
                            "use_module": "login",
                            "params": {"username": "${standard_username}"},
                        },
                        {"action": "assert_visible", "selector": "title"},
                    ],
                },
            },
        }
    )

    assert payload["data"]["test_generated"]["steps"][0] == {
        "use_module": "login",
        "params": {"username": "qa_user"},
    }
    assert harness.validate(payload) == []


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
        spec={"mode": "smart", "inputs": {"product_id": "DEMO-SKU-001"}},
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

    assert payload["data"]["test_generated"]["steps"][0]["value"] == ("DEMO-SKU-001")
    assert harness.validate(payload) == []


def test_generation_harness_inlines_nested_spec_inputs(tmp_path: Path):
    harness = GenerationHarness(
        context=ProjectContext(
            project="demo",
            test_dir=tmp_path,
            base_url="",
            elements={"checkout_first_name_input": "#first-name", "title": ".title"},
            modules={},
            variables={},
            test_cases=[],
            test_data={},
        ),
        spec={
            "mode": "smart",
            "inputs": {"checkout_info": {"first_name": "Test"}},
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
                            "selector": "checkout_first_name_input",
                            "value": "${checkout_info.first_name}",
                        },
                        {"action": "assert_visible", "selector": "title"},
                    ],
                }
            },
        }
    )

    assert payload["data"]["test_generated"]["steps"][0]["value"] == "Test"
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
            "steps": ["输入商品编号 DEMO-SKU-001", "点击查询"],
            "final": ["商品编号输入框已输入 DEMO-SKU-001"],
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
                            "value": "DEMO-SKU-001",
                        },
                        {
                            "action": "assert_text",
                            "selector": "title",
                            "value": "DEMO-SKU-001",
                        },
                    ],
                }
            },
        }
    )

    assert payload["data"]["test_generated"]["steps"][0]["value"] == ("DEMO-SKU-001")
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
        spec={"mode": "smart", "final": ["商品编号输入框已输入 DEMO-SKU-001"]},
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
                            "value": "DEMO-SKU-001",
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
        "target": "product ID",
        "value": "DEMO-SKU-001",
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
                "商品编号输入框已输入 DEMO-SKU-001",
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
                            "value": "DEMO-SKU-001",
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
            "steps": ["输入商品编号 DEMO-SKU-001", "点击查询"],
            "final": ["商品编号输入框已输入 DEMO-SKU-001"],
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
                            "value": "DEMO-SKU-001",
                        },
                        {
                            "action": "assert_text_contains",
                            "selector": "audit_link",
                            "value": "DEMO-SKU-001",
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
        "target": "product ID",
        "value": "DEMO-SKU-001",
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
            "steps": ["输入商品编号 DEMO-SKU-001", "点击查询"],
            "final": ["商品编号输入框已输入 DEMO-SKU-001"],
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
            "final": ["current page remains on catalog review page"],
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
                        {"action": "assert_title_contains", "value": "Catalog Review"},
                        {"action": "assert_url_contains", "value": "catalog-review"},
                    ],
                }
            },
        }
    )

    steps = payload["data"]["test_generated"]["steps"]
    assert steps == [
        {
            "action": "assert_text_contains",
            "selector": "title",
            "target": "title",
            "value": "Catalog Review",
        }
    ]


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
        "target": "title",
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


def test_generation_write_payload_restores_files_when_interrupted(tmp_path: Path):
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

    with pytest.raises(KeyboardInterrupt):
        _write_payload(
            result,
            overwrite=True,
            post_verify=lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
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
            "cases": [{"name": "test_generated"}, {"name": "test_generated_again"}],
            "data": {
                "test_generated": {
                    "mode": "smart",
                    "steps": [
                        {"use_module": "login"},
                        {"action": "assert_visible", "selector": "title"},
                    ],
                },
                "test_generated_again": {
                    "mode": "smart",
                    "steps": [
                        {"use_module": "login"},
                        {"action": "assert_visible", "selector": "title"},
                    ],
                },
            },
            "modules": {
                "login": [
                    {"action": "fill", "selector": "username", "value": "${username}"},
                    {"action": "click", "selector": "title", "target": "title"},
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
