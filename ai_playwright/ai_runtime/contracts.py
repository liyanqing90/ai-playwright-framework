from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SelectorDecision(StrictModel):
    status: Literal["ok", "need_more_context", "blocked", "failed"] = "ok"
    element_id: str | None = None
    selected_element_id: str | None = None
    selector: str | None = Field(default=None, min_length=1)
    selector_type: Literal["css", "xpath", "text"] = "css"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str | None = Field(default=None, max_length=120)
    expected: str | None = Field(default=None, max_length=120)

    @model_validator(mode="before")
    @classmethod
    def normalize_status_aliases(cls, data):
        if isinstance(data, dict):
            normalized = dict(data)
            status = str(normalized.get("status") or "").strip().lower()
            if status in {"success", "succeeded", "done"}:
                normalized["status"] = "ok"
            elif status in {"error", "reject", "rejected"}:
                normalized["status"] = "failed"
            return normalized
        return data

    @model_validator(mode="after")
    def validate_locator_payload(self):
        if self.status == "ok" and not (
            self.element_id or self.selected_element_id or self.selector
        ):
            raise ValueError("selector decision requires element_id or selector")
        if (
            self.status in {"need_more_context", "blocked", "failed"}
            and not self.reason
        ):
            raise ValueError(f"{self.status} requires reason")
        return self


class SelectorSemanticValidationDecision(StrictModel):
    status: Literal["match", "mismatch", "uncertain"]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def validate_reason_for_non_match(self):
        if self.status in {"mismatch", "uncertain"} and not self.reason:
            raise ValueError(f"{self.status} requires reason")
        return self


class AiStepDecision(StrictModel):
    status: Literal["ok", "need_more_context", "blocked", "failed"] = "ok"
    action: Literal["click", "fill", "press", "wait", "skip", "reject"] | None = None
    element_id: str | None = None
    selector: str | None = None
    value: str | None = None
    key: str | None = None
    wait_ms: int | None = Field(default=None, ge=0, le=60000)
    reason: str | None = Field(default=None, max_length=120)
    expected: str | None = Field(default=None, max_length=120)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_action_payload(self):
        if self.status in {"need_more_context", "blocked", "failed"}:
            if not self.reason:
                raise ValueError(f"{self.status} requires reason")
            return self
        if not self.action:
            raise ValueError("ok ai_step decision requires action")
        if self.action in {"click", "fill", "press"} and not (
            self.element_id or self.selector
        ):
            raise ValueError(f"{self.action} action requires element_id or selector")
        if self.action == "press" and not (
            self.element_id or self.selector or self.target
        ):
            raise ValueError("press action requires element_id, selector or target")
        if self.action == "press" and not self.key:
            raise ValueError("press action requires key")
        if self.action == "wait" and self.wait_ms is None:
            raise ValueError("wait action requires wait_ms")
        if self.action == "reject" and not self.reason:
            raise ValueError("reject action requires reason")
        return self


class AgentCaseRuntimeDecision(StrictModel):
    status: Literal["ok", "need_more_context", "blocked", "failed"] = "ok"
    action: (
        Literal[
            "goto",
            "click",
            "fill",
            "press",
            "wait",
            "assert_visible",
            "assert_text",
            "assert_url_contains",
            "assert_title",
            "assert_title_contains",
            "done",
            "finish",
            "fail",
        ]
        | None
    ) = None
    mode: Literal["smart"] | None = None
    element_id: str | None = None
    selector: str | None = None
    target: str | None = None
    value: str | None = None
    key: str | None = None
    wait_ms: int | None = Field(default=None, ge=0, le=60000)
    reason: str | None = Field(default=None, max_length=120)
    expected: str | None = Field(default=None, max_length=120)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    criteria_update: dict[str, list[str]] | None = None
    context_level: int | None = Field(default=None, ge=1, le=5)

    @model_validator(mode="after")
    def validate_action_payload(self):
        if self.status in {"need_more_context", "blocked", "failed"}:
            if not self.reason:
                raise ValueError(f"{self.status} requires reason")
            return self
        if not self.action:
            raise ValueError("ok decision requires action")
        if self.action == "goto" and not self.value:
            raise ValueError("goto action requires value")
        if self.action in {"click", "assert_visible"} and not (
            self.element_id or self.selector or self.target
        ):
            raise ValueError(
                f"{self.action} action requires element_id, selector or target"
            )
        if self.action == "fill" and not (
            self.element_id or self.selector or self.target
        ):
            raise ValueError("fill action requires element_id, selector or target")
        if self.action == "fill" and self.value is None:
            raise ValueError("fill action requires value")
        if self.action == "press" and not self.key:
            raise ValueError("press action requires key")
        if self.action == "assert_text" and self.value is None:
            raise ValueError("assert_text action requires value")
        if self.action == "assert_text" and not (
            self.element_id or self.selector or self.target
        ):
            raise ValueError(
                "assert_text action requires element_id, selector or target"
            )
        if (
            self.action
            in {
                "assert_url_contains",
                "assert_title",
                "assert_title_contains",
            }
            and self.value is None
        ):
            raise ValueError(f"{self.action} action requires value")
        if self.action == "wait" and self.wait_ms is None:
            raise ValueError("wait action requires wait_ms")
        if self.action in {"done", "finish", "fail"} and not self.reason:
            raise ValueError(f"{self.action} action requires reason")
        return self


class AgentCaseDecision(AgentCaseRuntimeDecision):
    action: (
        Literal[
            "goto",
            "use_module",
            "click",
            "fill",
            "press",
            "wait",
            "assert_visible",
            "assert_text",
            "assert_url_contains",
            "assert_title",
            "assert_title_contains",
            "done",
            "finish",
            "fail",
        ]
        | None
    ) = None
    module: str | None = None
    params: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_module_payload(self):
        if self.status in {"need_more_context", "blocked", "failed"}:
            return self
        if self.action == "use_module" and not self.module:
            raise ValueError("use_module action requires module")
        return self


# Backward-compatible contract name for existing tests/config references.
ObservedOperationDecision = AiStepDecision


class GeneratedCase(StrictModel):
    name: str = Field(min_length=1)


class GeneratedCaseData(StrictModel):
    description: str | None = None
    mode: Literal["strict", "smart"] = "smart"
    inputs: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)


class GeneratedCasePayload(StrictModel):
    cases: list[GeneratedCase] = Field(default_factory=list)
    data: dict[str, GeneratedCaseData] = Field(default_factory=dict)
    elements: dict[str, Any] = Field(default_factory=dict)
    modules: dict[str, Any] = Field(default_factory=dict)
    vars: dict[str, Any] = Field(default_factory=dict)
