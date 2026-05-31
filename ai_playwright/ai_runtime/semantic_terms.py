from __future__ import annotations

import re


SEMANTIC_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        "查询",
        "搜索",
        "检索",
        "查找",
        "筛选",
        "过滤",
        "query",
        "search",
        "find",
        "lookup",
        "filter",
    ),
    ("输入", "键入", "填写", "填入", "录入", "type", "enter", "input", "fill"),
    (
        "点击",
        "单击",
        "点选",
        "选择",
        "打开",
        "按下",
        "click",
        "tap",
        "select",
        "open",
        "press",
    ),
    ("查看", "浏览", "展示", "显示", "view", "show", "display"),
    ("日志", "记录", "log", "logs"),
    ("登录", "登陆", "login", "sign in"),
)

_GENERIC_PREFIXES = (
    "请输入",
    "请",
    "点击",
    "单击",
    "点选",
    "选择",
    "打开",
    "查看",
    "浏览",
    "输入",
    "键入",
    "填写",
    "填入",
    "录入",
    "click",
    "tap",
    "select",
    "open",
    "view",
    "show",
    "type",
    "enter",
    "input",
    "fill",
)

_GENERIC_SUFFIXES = (
    "输入框",
    "文本框",
    "搜索框",
    "查询框",
    "按钮",
    "链接",
    "字段",
    "button",
    "link",
    "input",
    "field",
    "textbox",
)


def strip_selector_prefix(value: str | None) -> str:
    text = str(value or "").strip()
    lowered = text.lower()
    for prefix in ("text=", "label=", "placeholder="):
        if lowered.startswith(prefix):
            return text.split("=", 1)[1].strip()
    return text


def strip_generic_target_words(value: str | None) -> str:
    text = strip_selector_prefix(value).strip(" \"'[]()（）:：,，.。;；")
    changed = True
    while changed:
        changed = False
        for suffix in _GENERIC_SUFFIXES:
            if _ends_with_term(text, suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)].strip()
                changed = True
                break
        if changed:
            continue
        for prefix in _GENERIC_PREFIXES:
            if _starts_with_term(text, prefix) and len(text) > len(prefix):
                remainder = text[len(prefix) :].strip()
                if remainder and remainder not in _GENERIC_SUFFIXES:
                    text = remainder
                    changed = True
                    break
    return text.strip(" \"'[]()（）:：,，.。;；")


def semantic_text_variants(value: str | None) -> list[str]:
    base = strip_generic_target_words(value)
    if not base:
        return []
    variants: list[str] = []
    _append_unique(variants, base)
    for group in SEMANTIC_SYNONYM_GROUPS:
        matched_terms = [term for term in group if _contains_term(base, term)]
        for matched in matched_terms:
            for synonym in group:
                replaced = _replace_term(base, matched, synonym)
                normalized = strip_generic_target_words(replaced)
                if normalized:
                    _append_unique(variants, normalized)
    return variants


def _append_unique(values: list[str], value: str) -> None:
    normalized = str(value or "").strip()
    if normalized and normalized not in values:
        values.append(normalized)


def _contains_term(value: str, term: str) -> bool:
    if not value or not term:
        return False
    if _has_cjk(term):
        return term in value
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", value, re.I))


def _replace_term(value: str, old: str, new: str) -> str:
    if _has_cjk(old):
        return value.replace(old, new)
    return re.sub(
        rf"(?<![a-z0-9]){re.escape(old)}(?![a-z0-9])",
        new,
        value,
        flags=re.I,
    )


def _starts_with_term(value: str, term: str) -> bool:
    if _has_cjk(term):
        return value.startswith(term)
    return bool(re.match(rf"{re.escape(term)}(?=\s+|[-_:：])", value, re.I))


def _ends_with_term(value: str, term: str) -> bool:
    if _has_cjk(term):
        return value.endswith(term)
    return bool(re.search(rf"(\s+|[-_:：]){re.escape(term)}$", value, re.I))


def _has_cjk(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", value))
