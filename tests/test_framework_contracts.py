import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from src.ai_generation.case_generator import (
    _GenerationArtifacts,
    _has_explicit_steps,
    _validate_spec_project_scope,
    _write_payload,
    resolve_generation_spec_path,
)
from src.ai_generation.harness import GenerationHarness
from src.ai_generation.harness import _safe_case_name
from src.ai_generation.project_context import ProjectContext
from src.ai_runtime.contracts import (
    AiStepDecision,
    GeneratedCasePayload,
    SelectorDecision,
    VisionFindResult,
)
from src.ai_runtime.element_store import ElementDefinitionStore
from src.ai_runtime.playwright_selectors import heuristic_selectors, redact_value
from src.ai_runtime.provider import (
    ChatCompletionProvider,
    LLMSettings,
    build_response_format,
    openai_strict_schema,
    parse_json_object,
    parse_model_response,
)
from src.ai_runtime.selector_registry import SelectorRegistry
from src.ai_runtime.smart_resolver import (
    AiStepOperation,
    ResolvedSelector,
    SmartResolver,
)
from src.ai_runtime.vision_client import (
    VisionClient,
    VisionServiceUnavailable,
    VisionSettings,
)
from src.ai_runtime.vision_resolver import VisionResolution, VisionResolver
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


def test_case_and_step_ai_modes_are_native_executor_defaults():
    executor = StepExecutor(page=None, ui_helper=None, elements={}, default_mode="ai")

    assert executor._resolve_mode({}) == "ai"
    assert executor._resolve_mode({"mode": "smart"}) == "smart"


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
            if self.selector != "#login-button":
                raise TimeoutError("not found")

        def is_enabled(self):
            return True

        def evaluate(self, script):
            return self.selector

    class FakePage:
        url = "https://example.test/login"

        def locator(self, selector):
            return FakeLocator(selector)

        def evaluate(self, script, limit):
            return []

    resolver = SmartResolver(FakePage(), project="demo", env="test")
    resolver.registry = None

    resolved = resolver.resolve(
        action="click",
        selector="#old-login",
        target="登录按钮",
        mode="smart",
        timeout=1000,
    )

    assert resolved.selector == "#login-button"
    assert resolved.source == "heuristic"
    assert resolved.healed is True
    assert resolved.healing_attempted is True
    assert resolved.original_selector == "#old-login"
    assert "not found" in (resolved.original_error or "")


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
                confidence=0.8,
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
        "action",
        "selector",
        "value",
        "key",
        "wait_ms",
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

    with pytest.raises(ValueError, match="mode: smart/ai"):
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
                            "use_module": "login",
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
    assert "#password" in selectors[:2]


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
            "mode": "ai",
            "steps": [{"action": "ai_step", "instruction": "打开购物车"}],
        },
    )

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
