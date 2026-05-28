from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


ActionName = Literal[
    "click",
    "fill",
    "select",
    "press",
    "press_key",
    "hover",
    "wait",
    "assert_visible",
    "assert_be_hidden",
    "assert_text",
    "assert_text_contains",
    "assert_url",
    "extract_text",
]


@dataclass(frozen=True)
class ActionIntent:
    action: str
    instruction: str
    target: str | None = None
    value: str | None = None
    area: str | None = None
    expected: Any | None = None
    operator: str | None = None
    index: int | None = None
    timeout_ms: int = 10_000
    strict: bool = False
    source_step_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectorCandidate:
    selector: str
    selector_type: str
    score: float
    reason: str | None = None
    source: str = "native_observe"
    is_verified: bool = False
    match_count: int | None = None
    visible_count: int | None = None
    last_error: str | None = None


@dataclass(frozen=True)
class ElementCandidate:
    element_id: str
    tag: str
    role: str | None = None
    type: str | None = None
    text: str | None = None
    accessible_name: str | None = None
    aria_label: str | None = None
    placeholder: str | None = None
    label: str | None = None
    title: str | None = None
    id_attr: str | None = None
    name_attr: str | None = None
    class_name: str | None = None
    test_id: str | None = None
    visible: bool = True
    enabled: bool = True
    rect: dict[str, Any] = field(default_factory=dict)
    near_text: list[str] = field(default_factory=list)
    parent_text: str | None = None
    section_heading: str | None = None
    row_text: str | None = None
    frame_id: str | None = None
    selector_candidates: list[SelectorCandidate] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedAction:
    intent: ActionIntent
    method: str
    selectors: list[SelectorCandidate]
    confidence: float
    source: str
    selected_element_id: str | None = None
    ai_called: bool = False
    llm_skip_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectorValidation:
    selector: str
    action: str
    ok: bool
    match_count: int | None = None
    visible_count: int | None = None
    enabled: bool | None = None
    action_compatible: bool | None = None
    error: str | None = None
    locator: Any | None = None


@dataclass(frozen=True)
class NativeObserveSettings:
    enabled: bool = True
    include_open_shadow_dom: bool = True
    include_iframes: bool = False
    max_candidates: int = 120
    max_text_length: int = 180
    ignore_selectors: tuple[str, ...] = ()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        default_limit: int = 120,
    ) -> "NativeObserveSettings":
        native_cfg = config.get("native_observe")
        if not isinstance(native_cfg, Mapping):
            native_cfg = (config.get("ai_resolver") or {}).get("native_observe", {})
        if not isinstance(native_cfg, Mapping):
            native_cfg = {}
        ignore_selectors = native_cfg.get("ignore_selectors") or ()
        if isinstance(ignore_selectors, str):
            ignore_selectors = (ignore_selectors,)
        return cls(
            enabled=bool(native_cfg.get("enabled", True)),
            include_open_shadow_dom=bool(
                native_cfg.get("include_open_shadow_dom", True)
            ),
            include_iframes=bool(native_cfg.get("include_iframes", False)),
            max_candidates=int(native_cfg.get("max_candidates", default_limit)),
            max_text_length=int(native_cfg.get("max_text_length", 180)),
            ignore_selectors=tuple(str(item) for item in ignore_selectors if item),
        )


def build_selector_candidates(candidate: Mapping[str, Any]) -> list[SelectorCandidate]:
    """Build ordered selector candidates from a native DOM candidate."""

    result: list[SelectorCandidate] = []
    tag = str(candidate.get("tag") or "").lower()

    def add(
        selector: str | None, selector_type: str, score: float, reason: str
    ) -> None:
        if not selector:
            return
        text = str(selector).strip()
        if not text or any(item.selector == text for item in result):
            return
        result.append(
            SelectorCandidate(
                selector=text,
                selector_type=selector_type,
                score=score,
                reason=reason,
            )
        )

    data_test = candidate.get("data_test")
    data_testid = candidate.get("data_testid")
    element_id = candidate.get("id")
    name = candidate.get("name")
    aria_label = candidate.get("aria_label")
    placeholder = candidate.get("placeholder")
    text = _trim_text(candidate.get("text") or "", 60)
    raw_selector = candidate.get("selector")

    add(_attr_selector(tag, "data-test", data_test), "testid", 0.98, "data-test")
    add(_attr_selector(tag, "data-testid", data_testid), "testid", 0.98, "data-testid")
    add(f"#{_css_ident(element_id)}" if element_id else None, "id", 0.82, "id")
    add(_attr_selector(tag, "name", name), "css", 0.80, "name")
    add(_attr_selector(tag, "aria-label", aria_label), "css", 0.78, "aria-label")
    add(_attr_selector(tag, "placeholder", placeholder), "css", 0.76, "placeholder")
    if text and tag in {"button", "a"}:
        add(f'{tag}:has-text("{_css_attr(text)}")', "text", 0.72, "element text")
    add(str(raw_selector) if raw_selector else None, "css", 0.65, "stable selector")
    return result


def _attr_selector(tag: str, attr: str, value: Any) -> str | None:
    if not value:
        return None
    return f'{tag or "*"}[{attr}="{_css_attr(value)}"]'


def _css_attr(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _css_ident(value: Any) -> str:
    return re.sub(r"([^a-zA-Z0-9_-])", r"\\\1", str(value))


def _trim_text(value: Any, limit: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."
