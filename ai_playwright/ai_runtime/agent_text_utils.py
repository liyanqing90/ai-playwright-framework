from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse


_URL_RE = re.compile(r"https?://[^\s\"'<>，,。；;、)）\]】]+", re.IGNORECASE)


def _dedupe_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _normalize_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _short_list(value: Any, *, limit: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    result = [str(item).strip() for item in value if str(item).strip()]
    if len(result) <= limit:
        return result
    return result[:limit] + [f"...(+{len(result) - limit})"]


def _flatten_agent_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        result: list[str] = []
        for key, item in value.items():
            result.append(str(key))
            result.extend(_flatten_agent_text(item))
        return result
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_flatten_agent_text(item))
        return result
    return [str(value)]


def _normalized_goal_text(value: Any) -> str:
    text = str(value or "").lower()
    return " ".join(re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text).split())


def _normalize_assertion_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_identifier_words(value: Any) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
    text = re.sub(r"[^0-9A-Za-z]+", " ", text).lower()
    replacements = {
        "username": "user name",
        "firstname": "first name",
        "lastname": "last name",
        "zipcode": "zip code",
        "postalcode": "postal code",
    }
    for source, replacement in replacements.items():
        text = re.sub(rf"\b{source}\b", replacement, text)
    return " ".join(text.split())


def _step_action(step: dict[str, Any] | None) -> str:
    if not isinstance(step, dict):
        return ""
    return str(step.get("action") or ("use_module" if step.get("use_module") else ""))


def _is_assertion_step(step: dict[str, Any] | None) -> bool:
    return _step_action(step).lower().startswith("assert_")


def _text_query_from_selector(selector: str | None) -> str:
    text = str(selector or "").strip()
    if not text:
        return ""
    match = re.search(r":has-text\(\s*(['\"])(.*?)\1\s*\)", text, flags=re.I)
    if match:
        return match.group(2).strip()
    match = re.search(r"\btext\s*=\s*(['\"]?)([^'\"]+?)\1\s*$", text, flags=re.I)
    if match:
        return match.group(2).strip()
    if not re.search(
        r"(?:text\s*\(|normalize-space\s*\(|contains\s*\(|translate\s*\()",
        text,
        flags=re.I,
    ):
        return ""
    for pattern in (
        r"contains\s*\(\s*(?:text\s*\(\)|normalize-space\s*\([^)]*\)|\.)\s*,\s*(['\"])(.*?)\1\s*\)",
        r"(?:text\s*\(\)|normalize-space\s*\([^)]*\)|translate\s*\([^)]*\))\s*=\s*(['\"])(.*?)\1",
    ):
        match = re.search(pattern, text, flags=re.I)
        if match and _selector_text_literal_looks_meaningful(match.group(2)):
            return match.group(2).strip()
    literals = [
        match.group(2).strip()
        for match in re.finditer(r"(['\"])(.*?)\1", text)
        if _selector_text_literal_looks_meaningful(match.group(2))
    ]
    if literals:
        return literals[-1]
    return ""


def _selector_text_literal_looks_meaningful(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text):
        return False
    return True


def _url_state_changed(before: str, after: str) -> bool:
    if not before or not after:
        return False
    before_parsed = urlparse(str(before))
    after_parsed = urlparse(str(after))
    if (
        before_parsed.netloc
        and after_parsed.netloc
        and before_parsed.netloc != after_parsed.netloc
    ):
        return True
    return (
        before_parsed.path.rstrip("/") != after_parsed.path.rstrip("/")
        or before_parsed.fragment != after_parsed.fragment
    )


def _first_url(value: Any) -> str:
    if isinstance(value, str):
        match = _URL_RE.search(value)
        return _clean_extracted_url(match.group(0)) if match else ""
    if isinstance(value, list):
        for item in value:
            url = _first_url(item)
            if url:
                return url
    if isinstance(value, dict):
        for item in value.values():
            url = _first_url(item)
            if url:
                return url
    return ""


def _clean_extracted_url(url: str) -> str:
    cleaned = str(url or "").strip()
    cleaned = cleaned.rstrip(".,;:!?，。；：！？、")
    if cleaned != "https://" and cleaned != "http://":
        cleaned = cleaned.rstrip("/")
    return cleaned


def _first_module_url(modules: dict[str, Any]) -> str:
    if not isinstance(modules, dict):
        return ""
    for steps in modules.values():
        url = _first_module_url_from_steps(steps)
        if url:
            return url
    return ""


def _first_module_url_from_steps(steps: Any) -> str:
    if isinstance(steps, dict):
        return _first_module_url_from_steps(steps.get("steps") or [])
    if not isinstance(steps, list):
        return ""
    for step in steps:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "").lower()
        value = step.get("value") or step.get("url")
        if action in {"goto", "navigate", "open", "打开", "访问"}:
            url = _first_url(value)
            if url:
                return url
    return ""


def _is_external_url(*, current_url: str, next_url: str, fallback_url: str) -> bool:
    next_host = urlparse(next_url).netloc
    if not next_host:
        return False
    base_host = urlparse(
        current_url if current_url != "about:blank" else fallback_url
    ).netloc
    fallback_host = urlparse(fallback_url).netloc
    if next_host and fallback_host and next_host == fallback_host:
        return False
    return bool(base_host and next_host and next_host != base_host)


def _is_truncated_current_url_prefix(*, current_url: str, next_url: str) -> bool:
    current = str(current_url or "").strip()
    candidate = str(next_url or "").strip()
    if not current or not candidate or current == "about:blank":
        return False
    parsed_candidate = urlparse(candidate)
    parsed_current = urlparse(current)
    if not parsed_candidate.scheme or not parsed_candidate.netloc:
        return False
    if not parsed_current.scheme or not parsed_current.netloc:
        return False
    if candidate == current:
        return False
    return current.startswith(candidate) and len(candidate) < len(current)
