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
        "ai_playwright.ai_runtime.provider.requests.post",
        lambda *a, **k: FakeResponse(),
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.provider.get_token_usage_tracker", lambda: tracker
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
        "ai_playwright.ai_runtime.provider.requests.post",
        lambda *a, **k: FakeResponse(),
    )
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.provider.get_token_usage_tracker", lambda: tracker
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

    monkeypatch.setattr("ai_playwright.ai_runtime.provider.requests.post", fake_post)
    monkeypatch.setattr(
        "ai_playwright.ai_runtime.provider.get_token_usage_tracker", lambda: tracker
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
