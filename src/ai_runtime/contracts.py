from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SelectorDecision(StrictModel):
    selector: str = Field(min_length=1)
    selector_type: Literal["css", "xpath", "text"] = "css"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class AiStepDecision(StrictModel):
    action: Literal["click", "fill", "press", "wait", "skip"]
    selector: str | None = None
    value: str | None = None
    key: str | None = None
    wait_ms: int | None = Field(default=None, ge=0, le=60000)

    @model_validator(mode="after")
    def validate_action_payload(self):
        if self.action in {"click", "fill", "press"} and not self.selector:
            raise ValueError(f"{self.action} action requires selector")
        if self.action == "press" and not self.key:
            raise ValueError("press action requires key")
        if self.action == "wait" and self.wait_ms is None:
            raise ValueError("wait action requires wait_ms")
        return self


# Backward-compatible contract name for existing tests/config references.
ObservedOperationDecision = AiStepDecision


class GeneratedCase(StrictModel):
    name: str = Field(min_length=1)


class GeneratedCaseData(StrictModel):
    description: str = ""
    mode: Literal["strict", "smart", "ai"] = "strict"
    steps: list[dict[str, Any]] = Field(default_factory=list)


class GeneratedCasePayload(StrictModel):
    cases: list[GeneratedCase] = Field(default_factory=list)
    data: dict[str, GeneratedCaseData] = Field(default_factory=dict)
    elements: dict[str, Any] = Field(default_factory=dict)
    modules: dict[str, Any] = Field(default_factory=dict)
    vars: dict[str, Any] = Field(default_factory=dict)


class VisionFindResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    found: bool = False
    target: str | None = None
    type: str | None = None
    text: str | None = None
    selector: str | None = None
    selected_candidate_index: int | None = Field(default=None, ge=0)
    selected_candidate_id: int | None = Field(default=None, ge=0)
    box: list[float] | None = None
    center: list[float] | None = None
    box_norm: list[float] | None = None
    center_norm: list[float] | None = None
    clickable: bool | None = None
    enabled: bool | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    method: str | None = None
    reason: str | None = None
    error_code: str | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)
