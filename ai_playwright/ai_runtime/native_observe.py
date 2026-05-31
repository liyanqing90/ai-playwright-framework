from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


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
    title = candidate.get("title")
    role = candidate.get("role")
    text = _trim_text(candidate.get("text") or "", 60)
    raw_selector = candidate.get("selector")

    add(_attr_selector(tag, "data-test", data_test), "testid", 0.98, "data-test")
    add(_attr_selector(tag, "data-testid", data_testid), "testid", 0.98, "data-testid")
    add(
        f"#{_css_ident(element_id)}" if _is_stable_id(element_id) else None,
        "id",
        0.82,
        "stable id",
    )
    add(_attr_selector(tag, "name", name), "css", 0.80, "name")
    add(_attr_selector(tag, "aria-label", aria_label), "css", 0.78, "aria-label")
    add(_attr_selector(tag, "placeholder", placeholder), "css", 0.76, "placeholder")
    add(_attr_selector(tag, "title", title), "css", 0.74, "title")
    if text:
        add(_role_text_selector(tag, role, text), "text", 0.73, "role text")
        if tag in {"button", "a", "label"}:
            add(f'{tag}:has-text("{_css_attr(text)}")', "text", 0.72, "element text")
    if _selector_safe_for_ai_candidate(raw_selector):
        add(str(raw_selector), "css", 0.65, "stable selector")
    return result


def _attr_selector(tag: str, attr: str, value: Any) -> str | None:
    if not value:
        return None
    return f'{tag or "*"}[{attr}="{_css_attr(value)}"]'


def _css_attr(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _css_ident(value: Any) -> str:
    return re.sub(r"([^a-zA-Z0-9_-])", r"\\\1", str(value))


def _role_text_selector(tag: str, role: Any, text: str) -> str | None:
    role_text = str(role or "").strip()
    if not role_text:
        return None
    text = _trim_text(text, 80)
    if not text:
        return None
    tag_prefix = f"{tag}" if tag else "*"
    return f'{tag_prefix}[role="{_css_attr(role_text)}"]:has-text("{_css_attr(text)}")'


def _is_stable_id(value: Any) -> bool:
    text = str(value or "").strip()
    if not text or len(text) > 80:
        return False
    if re.fullmatch(r"\d+", text):
        return False
    if re.fullmatch(r"[a-f0-9]{8,}", text, flags=re.IGNORECASE):
        return False
    if re.search(r"\d{8,}", text):
        return False
    if re.fullmatch(r"ember\d+", text, flags=re.IGNORECASE):
        return False
    return True


def _selector_safe_for_ai_candidate(selector: Any) -> bool:
    text = str(selector or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered in {
        "a",
        "button",
        "div",
        "input",
        "label",
        "li",
        "main",
        "section",
        "select",
        "span",
        "textarea",
        "ul",
    }:
        return False
    if re.fullmatch(
        r"(?:[a-z][\w-]*(?::nth-of-type\(\d+\))?\s*>\s*)+"
        r"[a-z][\w-]*(?::nth-of-type\(\d+\))?",
        lowered,
    ):
        return False
    if ":nth-of-type(" in lowered and not any(
        marker in lowered
        for marker in (
            "[data-testid=",
            "[data-test=",
            "[data-qa=",
            "[data-cy=",
            "[data-ui=",
            "[aria-label=",
            "[placeholder=",
            "[title=",
            "[name=",
            ":has-text(",
            "#",
        )
    ):
        return False
    return True


def _trim_text(value: Any, limit: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."
