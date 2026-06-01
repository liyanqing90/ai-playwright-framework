import json

from ai_playwright.ai_runtime.payload_compactor import build_dom_context
from ai_playwright.ai_runtime.playwright_selectors import redact_value
from ai_playwright.utils.token_usage import TokenUsageTracker


def test_local_ai_candidate_values_are_not_redacted():
    raw = "phone=13812345678 email=qa@example.com token=abcdefghijklmnopqrstuvwxyzABCDEF123456"

    assert redact_value(raw, policy="trusted_local") == raw


def test_external_ai_candidate_values_are_redacted():
    raw = "phone=13812345678 email=qa@example.com token=abcdefghijklmnopqrstuvwxyzABCDEF123456"

    redacted = redact_value(raw, policy="external")

    assert "13812345678" not in redacted
    assert "qa@example.com" not in redacted
    assert "abcdefghijklmnopqrstuvwxyzABCDEF123456" not in redacted
    assert "<redacted:phone>" in redacted
    assert "<redacted:email>" in redacted
    assert "<redacted:credential>" in redacted


def test_dom_context_external_policy_redacts_llm_visible_text():
    context = build_dom_context(
        [
            {
                "index": 1,
                "tag": "button",
                "selector": "#contact-customer",
                "text": "联系客户 13812345678 qa@example.com",
                "ancestor_text": "订单 token=abcdefghijklmnopqrstuvwxyzABCDEF123456",
                "visible": True,
                "enabled": True,
            }
        ],
        url="https://example.test/orders?token=abcdefghijklmnopqrstuvwxyzABCDEF123456",
        title="客户 qa@example.com",
        data_policy="external",
    )

    element = context["interactive_elements"][0]
    serialized = json.dumps(context, ensure_ascii=False)

    assert "13812345678" not in serialized
    assert "qa@example.com" not in serialized
    assert "abcdefghijklmnopqrstuvwxyzABCDEF123456" not in serialized
    assert element["selector_candidates"] == ["#contact-customer"]


def test_model_io_trace_is_redacted_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_DATA_POLICY", "external")
    tracker = TokenUsageTracker(base_dir=tmp_path)
    tracker.start_run(run_kind="pytest", run_id="redaction-test")

    path = tracker.record_model_io(
        operation="test.parse_error",
        request_payload={
            "content": "phone=13812345678 token=abcdefghijklmnopqrstuvwxyzABCDEF123456"
        },
        response_payload={"content": "email=qa@example.com"},
        error="authorization=Bearer abcdefghijklmnopqrstuvwxyzABCDEF123456",
    )

    payload = json.loads(tmp_path.joinpath(path).read_text(encoding="utf-8"))
    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["data_policy"] == "external"
    assert "13812345678" not in serialized
    assert "qa@example.com" not in serialized
    assert "abcdefghijklmnopqrstuvwxyzABCDEF123456" not in serialized
