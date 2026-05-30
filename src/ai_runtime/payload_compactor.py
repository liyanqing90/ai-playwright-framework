from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Any

from src.ai_runtime.native_observe import build_selector_candidates
from src.ai_runtime.semantic_terms import (
    semantic_text_variants,
    strip_generic_target_words as _shared_strip_generic_target_words,
)

if TYPE_CHECKING:
    from src.ai_generation.project_context import ProjectContext


_INTERACTIVE_TAGS = {"a", "button", "input", "select", "textarea", "option"}
_IMPORTANT_TEXT_TAGS = {
    "div",
    "span",
    "label",
    "p",
    "li",
    "td",
    "th",
    "h1",
    "h2",
    "h3",
}
_SIGNAL_KEYWORDS = {
    "add",
    "back",
    "badge",
    "close",
    "continue",
    "error",
    "finish",
    "login",
    "logout",
    "menu",
    "password",
    "remove",
    "submit",
    "username",
    "登录",
    "退出",
    "菜单",
    "提交",
    "确认",
    "查询",
    "搜索",
    "检索",
    "查找",
    "筛选",
    "过滤",
    "查看",
    "日志",
    "输入",
    "键入",
    "填写",
    "点击",
    "选择",
    "打开",
}
_DOM_FIELDS = (
    "index",
    "tag",
    "selector",
    "id",
    "data_test",
    "data_testid",
    "role",
    "text",
    "aria_label",
    "placeholder",
    "title",
    "name",
    "type",
    "label",
    "ancestor_text",
    "visible",
    "enabled",
)
_TEXT_LIMITS = {
    "selector": 180,
    "id": 80,
    "data_test": 100,
    "data_testid": 100,
    "role": 60,
    "text": 100,
    "aria_label": 100,
    "placeholder": 100,
    "title": 100,
    "name": 80,
    "type": 60,
    "label": 100,
    "ancestor_text": 160,
}
_ASSERTION_TAGS = {"h1", "h2", "h3", "label", "span", "div", "p", "li"}
_ASSERTION_KEYWORDS = {
    "error",
    "badge",
    "title",
    "toast",
    "alert",
    "dialog",
    "complete",
    "错误",
    "提示",
    "标题",
    "完成",
}


def compact_dom_candidates(
    candidates: list[dict[str, Any]],
    *,
    limit: int = 40,
    hints: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Compress DOM candidates for text LLM calls.

    The text model does not need geometry or raw class names. Those fields stay in
    the vision path where coordinate mapping matters.
    """

    hint_terms = _terms(" ".join(_flatten_texts(hints or [])))
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for position, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        item = _compact_candidate(candidate)
        if not item:
            continue
        score = _candidate_score(item, hint_terms)
        if score <= 0:
            continue
        scored.append((score, position, item))

    scored.sort(key=lambda row: (-row[0], row[1]))
    selected = scored[: max(0, limit)]
    selected.sort(key=lambda row: row[1])
    return [item for _, _, item in selected]


def build_dom_context(
    candidates: list[dict[str, Any]],
    *,
    url: str = "",
    title: str = "",
    context_level: int = 2,
    limit: int = 40,
    hints: list[Any] | None = None,
) -> dict[str, Any]:
    """Build an accessibility-like DOM context for Agent/Smart LLM calls."""

    compacted = compact_dom_candidates(candidates, limit=limit, hints=hints)
    interactive = [
        _to_element(candidate, kind="interactive")
        for candidate in compacted
        if _is_interactive(candidate)
    ]
    assertion_candidates = [
        _to_element(candidate, kind="assertion")
        for candidate in compacted
        if _is_assertion_candidate(candidate)
    ]
    forms = [
        _to_form(candidate)
        for candidate in compacted
        if str(candidate.get("tag") or "").lower() in {"input", "textarea", "select"}
    ]
    business_objects = _extract_business_objects(interactive)
    return {
        "meta": {
            "url": url,
            "title": title,
            "route_hint": _route_hint(url),
        },
        "page_summary": {
            "main_heading": _main_heading(compacted),
            "visible_text_summary": _visible_text_summary(compacted),
        },
        "forms": forms,
        "business_objects": business_objects,
        "interactive_elements": interactive,
        "assertion_candidates": assertion_candidates,
        "navigation": {
            "links": [
                item
                for item in interactive
                if item.get("role") == "link" or item.get("tag") == "a"
            ][:12],
            "menu_items": [
                item
                for item in interactive
                if "menu" in _element_blob(item)
                or str(item.get("role") or "").lower() in {"menuitem", "tab"}
            ][:12],
        },
        "compression": {
            "raw_element_count": len(candidates),
            "kept_element_count": len(compacted),
            "interactive_count": len(interactive),
            "assertion_count": len(assertion_candidates),
            "context_level": context_level,
        },
    }


def build_locator_context(
    *,
    action: str,
    target: str,
    candidates: list[dict[str, Any]],
    url: str = "",
    title: str = "",
    limit: int = 12,
) -> dict[str, Any]:
    expanded_limit = min(len(candidates), max(limit * 8, limit + 160))
    dom_context = build_dom_context(
        candidates,
        url=url,
        title=title,
        context_level=1,
        limit=expanded_limit,
        hints=[action, target],
    )
    elements = (
        dom_context["interactive_elements"]
        if action in {"click", "fill", "press", "press_key"}
        else dom_context["interactive_elements"] + dom_context["assertion_candidates"]
    )
    elements = _rank_locator_elements(elements, action=action, target=target)
    return {
        "locator_task": {
            "action": action,
            "target": target,
        },
        "page": dom_context["meta"],
        "candidates": elements[:limit],
        "compression": dom_context["compression"],
    }


def selector_for_element_id(
    payload: dict[str, Any] | list[dict[str, Any]],
    element_id: str | None,
) -> str | None:
    selectors = selectors_for_element_id(payload, element_id)
    return selectors[0] if selectors else None


def selectors_for_element_id(
    payload: dict[str, Any] | list[dict[str, Any]],
    element_id: str | None,
) -> list[str]:
    if not element_id:
        return []
    for element in _iter_elements(payload):
        if element.get("id") != element_id:
            continue
        result: list[str] = []
        selectors = element.get("selector_candidates")
        if isinstance(selectors, list):
            for selector in selectors:
                if isinstance(selector, str) and selector.strip():
                    result.append(selector.strip())
        selector = element.get("selector")
        if isinstance(selector, str) and selector.strip():
            result.append(selector.strip())
        return _dedupe(result)
    return []


def looks_like_internal_element_id(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(re.fullmatch(r"[efa]\d+", text))


def _rank_locator_elements(
    elements: list[dict[str, Any]],
    *,
    action: str,
    target: str,
) -> list[dict[str, Any]]:
    terms = _terms(f"{action} {target}")
    target_texts = [
        _remove_cjk_display_spaces(item.lower())
        for item in semantic_text_variants(str(target or ""))
    ]
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for index, element in enumerate(elements):
        blob = _element_blob(element)
        normalized_blob = _remove_cjk_display_spaces(blob)
        score = 0
        if any(target_text and target_text in normalized_blob for target_text in target_texts):
            score += 80
        score += min(
            60,
            sum(
                12
                for term in terms
                if term and (term in blob or term in normalized_blob)
            ),
        )
        if _locator_action_compatible(element, action):
            score += 20
        if _is_unrequested_exit_candidate(normalized_blob, terms):
            score -= 80
        ranked.append((score, index, element))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [element for _, _, element in ranked]


def _locator_action_compatible(element: dict[str, Any], action: str) -> bool:
    tag = str(element.get("tag") or "").lower()
    role = str(element.get("role") or "").lower()
    if action == "fill":
        return tag in {"input", "textarea"} or role in {"textbox", "combobox"}
    if action in {"click", "press", "press_key"}:
        return tag in {"button", "a"} or role in {"button", "link", "menuitem", "tab"}
    return True


def normalize_model_text(value: Any, *, limit: int = 80) -> str:
    return _trim_text(value or "", limit)


def compressed_decision_summary(
    decision: Any, *, result: str = "passed"
) -> dict[str, Any]:
    action = getattr(decision, "action", None)
    return {
        "action": action,
        "element_id": getattr(decision, "element_id", None),
        "selector": getattr(decision, "selector", None),
        "target": getattr(decision, "target", None),
        "result": result,
        "reason": normalize_model_text(getattr(decision, "reason", ""), limit=80),
        "expected": normalize_model_text(getattr(decision, "expected", ""), limit=80),
        "confidence": getattr(decision, "confidence", None),
    }


def compact_history(
    history: list[dict[str, Any]], *, limit: int = 10
) -> list[dict[str, Any]]:
    if not isinstance(history, list) or limit <= 0:
        return []
    compacted: list[dict[str, Any]] = []
    for item in history[-limit:]:
        if not isinstance(item, dict):
            continue
        step = item.get("step")
        if not isinstance(step, dict):
            continue
        compact_step: dict[str, Any] = {}
        for key in ("action", "use_module", "selector", "target", "value", "key"):
            if step.get(key) not in (None, ""):
                compact_step[key] = _trim_text(step[key], 140)
        for key in (
            "_action_before_url",
            "_action_after_url",
            "_action_executed_selector",
            "_action_dom_changed",
            "_action_target_text",
            "_action_target_visible_after",
            "_resolved_value",
            "_resolved_value_after",
        ):
            if step.get(key) not in (None, "", []):
                if isinstance(step.get(key), bool):
                    compact_step[key] = bool(step.get(key))
                else:
                    field_limit = (
                        80
                        if key
                        in {
                            "_resolved_value_after",
                            "_resolved_value",
                            "_action_target_text",
                            "_action_executed_selector",
                        }
                        else 140
                    )
                    compact_step[key] = _trim_text(step[key], field_limit)
        page_errors = step.get("_action_page_errors")
        if isinstance(page_errors, list) and page_errors:
            compact_step["_action_page_errors"] = [
                _trim_text(error, 120) for error in page_errors[:2] if str(error).strip()
            ]
        history_item = {
            "source": item.get("source"),
            "step": compact_step,
            "result": item.get("result", "passed"),
            "url_after": item.get("url_after"),
        }
        decision = item.get("decision")
        if isinstance(decision, dict):
            history_item["decision"] = {
                key: decision.get(key)
        for key in (
            "action",
            "element_id",
            "target",
            "selector",
            "reason",
            "expected",
            "confidence",
        )
                if decision.get(key) not in (None, "")
            }
        compacted.append(history_item)
    return compacted


def compact_project_context(
    context: "ProjectContext",
    *,
    max_items: int = 40,
    max_modules: int = 6,
    max_module_steps: int = 8,
    hints: list[Any] | None = None,
    include_modules: bool = True,
    include_elements: bool = True,
) -> dict[str, Any]:
    hint_terms = _terms(" ".join(_flatten_texts(hints or [])))
    element_keys = (
        _ranked_keys(context.elements.keys(), hint_terms, limit=max_items)
        if include_elements
        else []
    )
    module_names = (
        _ranked_keys(context.modules.keys(), hint_terms, limit=max_items)
        if include_modules
        else []
    )
    variable_keys = _ranked_keys(context.variables.keys(), hint_terms, limit=max_items)
    return {
        "project": context.project,
        "base_url": context.base_url,
        "element_keys": element_keys,
        "module_names": module_names,
        "variable_keys": variable_keys,
        "modules": (
            _compact_modules(
                context.modules,
                module_names=module_names[:max_modules],
                max_steps=max_module_steps,
            )
            if include_modules
            else {}
        ),
    }


def compact_model_dom_context(
    dom_context: dict[str, Any],
    *,
    candidate_limit: int = 12,
    selector_limit: int = 2,
    form_limit: int = 6,
    assertion_limit: int | None = None,
    hints: list[Any] | None = None,
    include_business_objects: bool = True,
    include_compression: bool = True,
) -> dict[str, Any]:
    if not isinstance(dom_context, dict):
        return {}
    business_objects: dict[str, Any] = {}
    if include_business_objects and isinstance(
        dom_context.get("business_objects"), dict
    ):
        business_objects = dom_context.get("business_objects") or {}
        cards = business_objects.get("cards")
        if isinstance(cards, list):
            business_objects = {
                "cards": [
                    _trim_business_object(item)
                    for item in cards[: min(candidate_limit, 6)]
                    if isinstance(item, dict)
                ]
            }
    page_summary = copy.deepcopy(dom_context.get("page_summary") or {})
    if isinstance(page_summary.get("visible_text_summary"), list):
        visible_summary = [
            _trim_text(item, 80)
            for item in page_summary.get("visible_text_summary", [])[:6]
            if str(item or "").strip()
        ]
        menu_outline = _menu_text_outline(dom_context, limit=24)
        if menu_outline:
            page_summary["menu_text_outline"] = menu_outline
            visible_summary.extend(
                item for item in menu_outline[:4] if item not in visible_summary
            )
        page_summary["visible_text_summary"] = visible_summary[:8]
    if page_summary.get("main_heading"):
        page_summary["main_heading"] = _trim_text(page_summary["main_heading"], 80)

    hint_terms = _terms(" ".join(_flatten_texts(hints or [])))
    priority_terms = _priority_terms_from_hints(hints or [])
    forms = _rank_model_items(dom_context.get("forms") or [], hint_terms)
    interactive_items = [
        item
        for item in (dom_context.get("interactive_elements") or [])
        if isinstance(item, dict)
    ]
    interactive_action_items = [
        item for item in interactive_items if not _is_form_like_model_item(item)
    ]
    interactive_form_items = [
        item for item in interactive_items if _is_form_like_model_item(item)
    ]
    interactive_hint_terms = priority_terms
    interactive = _rank_model_items(interactive_action_items, interactive_hint_terms)
    if not interactive_hint_terms and _has_value_like_hint(hint_terms):
        interactive = _rank_common_action_items(interactive)
    interactive.extend(_rank_model_items(interactive_form_items, hint_terms))
    interactive = _frontload_priority_items(interactive, priority_terms)
    interactive = _backfill_menu_outline_items(
        interactive,
        interactive_items,
        limit=max(0, candidate_limit),
    )
    assertions = _rank_model_items(
        dom_context.get("assertion_candidates") or [], hint_terms
    )
    assertions = _frontload_priority_items(assertions, priority_terms)

    result = {
        "meta": copy.deepcopy(dom_context.get("meta") or {}),
        "page_summary": page_summary,
        "forms": [
            _compact_model_element(item, selector_limit=selector_limit)
            for item in forms[: max(0, form_limit)]
            if isinstance(item, dict)
        ],
        "interactive_elements": [
            _compact_model_element(item, selector_limit=selector_limit)
            for item in interactive[: max(0, candidate_limit)]
            if isinstance(item, dict)
        ],
        "assertion_candidates": [
            _compact_model_element(item, selector_limit=selector_limit)
            for item in assertions[
                : max(0, assertion_limit if assertion_limit is not None else candidate_limit)
            ]
            if isinstance(item, dict)
        ],
    }
    if business_objects:
        result["business_objects"] = business_objects
    if include_compression:
        result["compression"] = copy.deepcopy(dom_context.get("compression") or {})
    return result


def _frontload_priority_items(
    items: list[dict[str, Any]],
    priority_terms: set[str],
) -> list[dict[str, Any]]:
    if not priority_terms:
        return items
    priority: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for item in items:
        blob = _remove_cjk_display_spaces(_model_item_blob(item))
        if any(term and term in blob for term in priority_terms):
            priority.append(item)
        else:
            rest.append(item)
    return priority + rest


def _priority_terms_from_hints(hints: list[Any]) -> set[str]:
    terms: set[str] = set()
    for item in _priority_hint_texts(hints):
        text = str(item or "")
        if not text:
            continue
        terms.update(_goal_target_terms(text))
    return terms


def _priority_hint_texts(values: list[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, str):
            lowered = value.lower()
            if (
                "intent click:" in lowered
                or "intent fill:" in lowered
                or _has_explicit_priority_action(value)
            ):
                result.append(value)
            continue
        if isinstance(value, dict):
            phase = value.get("phase")
            include_observation = (
                isinstance(phase, str) and _has_explicit_priority_action(phase)
            )
            observation = value.get("phase_observation")
            if include_observation and isinstance(observation, dict):
                for key in ("target", "next_phase"):
                    item = observation.get(key)
                    if isinstance(item, str) and item.strip():
                        result.append(f"intent click: {item}")
            continue
        if isinstance(value, list):
            result.extend(_priority_hint_texts(value))
    return result


def _has_explicit_priority_action(value: Any) -> bool:
    text = str(value or "")
    if not text:
        return False
    return bool(
        re.search(
            r"(?:点击|单击|点选|按下|查看|查询|搜索|筛选|click|tap|press|select|view|expand)",
            text,
            flags=re.I,
        )
    )


def _goal_target_terms(text: str) -> set[str]:
    raw = str(text or "")
    targets: list[str] = []
    for pattern in (
        r"当前阶段:\s*([^，。；;,\n]+)",
        r"current phase:\s*([^，。；;,\n]+)",
        r"intent click:\s*([^，。；;,\n]+)",
        r"intent fill:\s*([^，。；;,\n]+)",
        r"点击\s*([^，。；;,\n]+)",
        r"查看\s*([^，。；;,\n]+)",
        r"查询\s*([^，。；;,\n]+)",
        r"click\s+([^，。；;,\n]+)",
        r"view\s+([^，。；;,\n]+)",
        r"expand\s+([^，。；;,\n]+)",
        r"select\s+([^，。；;,\n]+)",
    ):
        for match in re.finditer(pattern, raw, flags=re.I):
            targets.append(match.group(1))
    result: set[str] = set()
    for target in targets:
        cleaned = _strip_generic_target_words(target)
        for variant in semantic_text_variants(cleaned) or [cleaned]:
            normalized = _remove_cjk_display_spaces(str(variant).lower())
            if len(normalized) >= 2:
                result.add(normalized)
            for token in re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]{3,}", normalized):
                stripped = _strip_generic_target_words(token)
                normalized_token = _remove_cjk_display_spaces(str(stripped).lower())
                if len(normalized_token) >= 2:
                    result.add(normalized_token)
    return set(list(result)[:24])


def _rank_common_action_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[tuple[int, int, dict[str, Any]]] = []
    has_signal = False
    for position, item in enumerate(items):
        score = _common_action_signal_score(item)
        has_signal = has_signal or score > 0
        scored.append((score, position, item))
    if not has_signal:
        return items
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [item for _, _, item in scored]


def _common_action_signal_score(item: dict[str, Any]) -> int:
    blob = _remove_cjk_display_spaces(_model_item_blob(item))
    tag = str(item.get("tag") or "").lower()
    role = str(item.get("role") or "").lower()
    is_direct_action = tag in {"button", "a"} or role in {"button", "link"}
    if not is_direct_action:
        return 0
    score = 0
    if any(
        term in blob
        for term in (
            "查询",
            "搜索",
            "检索",
            "筛选",
            "过滤",
            "search",
            "query",
            "filter",
            "查看",
            "日志",
            "详情",
            "view",
            "log",
            "detail",
        )
    ):
        score += 40
    if tag == "button" or role == "button":
        score += 8
    if tag == "a" or role == "link":
        score += 6
    return score


def _has_value_like_hint(hint_terms: set[str]) -> bool:
    for term in hint_terms:
        text = str(term or "").lower()
        if re.search(r"[a-z]{2,}\d{2,}", text) or re.search(r"\d{5,}", text):
            return True
    return False


def _backfill_menu_outline_items(
    ranked: list[dict[str, Any]],
    all_items: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return ranked
    result = list(ranked)
    seen_ids = {str(item.get("id") or "") for item in result if isinstance(item, dict)}
    for item in all_items:
        if len(result) >= limit:
            break
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "")
        if item_id and item_id in seen_ids:
            continue
        role = str(item.get("role") or "").lower()
        tag = str(item.get("tag") or "").lower()
        has_text = any(
            str(item.get(key) or "").strip()
            for key in ("text", "name", "label", "near_text")
        )
        if role != "menuitem" and tag != "li":
            continue
        if not has_text:
            continue
        result.append(item)
        if item_id:
            seen_ids.add(item_id)
    return result


def _menu_text_outline(dom_context: dict[str, Any], *, limit: int = 24) -> list[str]:
    result: list[str] = []
    for item in dom_context.get("interactive_elements") or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").lower()
        tag = str(item.get("tag") or "").lower()
        if role != "menuitem" and tag != "li":
            continue
        text = _trim_text(
            item.get("text")
            or item.get("name")
            or item.get("label")
            or item.get("near_text")
            or "",
            40,
        )
        if not text or text in result:
            continue
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _compact_model_element(
    item: dict[str, Any], *, selector_limit: int = 2
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, limit in (
        ("id", 24),
        ("tag", 24),
        ("role", 32),
        ("type", 32),
        ("input_type", 32),
        ("name", 64),
        ("label", 64),
        ("text", 64),
        ("placeholder", 64),
        ("near_text", 96),
        ("value_state", 16),
        ("low_confidence", 16),
    ):
        value = item.get(key)
        if value in (None, "", []):
            continue
        result[key] = _trim_text(value, limit) if isinstance(value, str) else value
    selectors = item.get("selector_candidates")
    if isinstance(selectors, list) and selector_limit > 0:
        result["selector_candidates"] = [
            str(selector).strip()
            for selector in selectors[:selector_limit]
            if str(selector or "").strip()
        ]
    attributes = item.get("attributes")
    if isinstance(attributes, dict):
        compact_attrs = {
            key: _trim_text(value, 48)
            for key, value in attributes.items()
            if key in {"data-test", "data-testid", "name", "type"} and value
        }
        if compact_attrs:
            result["attributes"] = compact_attrs
    if item.get("visible") is False:
        result["visible"] = False
    if item.get("enabled") is False:
        result["enabled"] = False
    return {key: value for key, value in result.items() if value not in (None, "", [])}


def _rank_model_items(
    items: list[Any],
    hint_terms: set[str],
) -> list[dict[str, Any]]:
    dict_items = [item for item in items if isinstance(item, dict)]
    if not hint_terms:
        return dict_items
    scored: list[tuple[int, int, dict[str, Any]]] = []
    has_signal = False
    for position, item in enumerate(dict_items):
        score = _model_item_hint_score(item, hint_terms)
        has_signal = has_signal or score > 0
        scored.append((score, position, item))
    if not has_signal:
        return dict_items
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [item for _, _, item in scored]


def _model_item_hint_score(item: dict[str, Any], hint_terms: set[str]) -> int:
    blob = _model_item_blob(item)
    normalized_blob = _remove_cjk_display_spaces(blob)
    score = 0
    for term in hint_terms:
        if not term:
            continue
        normalized_term = _remove_cjk_display_spaces(str(term).lower())
        if not normalized_term:
            continue
        if normalized_term == normalized_blob:
            score += 80
        elif normalized_term in normalized_blob:
            score += max(12, min(len(normalized_term), 24))
        elif normalized_blob in normalized_term and len(normalized_blob) >= 3:
            score += 6
    tag = str(item.get("tag") or item.get("type") or "").lower()
    role = str(item.get("role") or "").lower()
    if tag in {"input", "textarea", "select", "button", "a"} or role in {
        "button",
        "link",
        "textbox",
        "combobox",
        "menuitem",
        "tab",
    }:
        score += 3 if score > 0 else 0
    return score


def _model_item_blob(item: dict[str, Any]) -> str:
    values = [
        item.get("id"),
        item.get("tag"),
        item.get("role"),
        item.get("type"),
        item.get("input_type"),
        item.get("name"),
        item.get("label"),
        item.get("text"),
        item.get("placeholder"),
        item.get("near_text"),
        item.get("selector_candidates"),
        item.get("attributes"),
    ]
    return " ".join(_flatten_texts(values)).lower()


def _is_form_like_model_item(item: dict[str, Any]) -> bool:
    tag = str(item.get("tag") or item.get("type") or "").lower()
    role = str(item.get("role") or "").lower()
    return tag in {"input", "textarea", "select"} or role in {"textbox", "combobox"}


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in _DOM_FIELDS:
        value = candidate.get(key)
        if value is None or value == "":
            continue
        if key in _TEXT_LIMITS:
            value = _trim_text(value, _TEXT_LIMITS[key])
            if not value:
                continue
        result[key] = value
    return result


def _to_element(candidate: dict[str, Any], *, kind: str) -> dict[str, Any]:
    element_id = _element_id(candidate, prefix="a" if kind == "assertion" else "e")
    name = _accessible_name(candidate)
    near_text = _near_text(candidate, name)
    item: dict[str, Any] = {
        "id": element_id,
        "tag": candidate.get("tag"),
        "role": _role(candidate),
        "name": name,
        "text": _trim_text(candidate.get("text") or "", 80),
        "near_text": near_text,
        "visible": candidate.get("visible", True),
        "enabled": candidate.get("enabled", True),
        "selector_candidates": _selector_candidates(candidate),
    }
    attributes = {
        "data-test": candidate.get("data_test"),
        "data-testid": candidate.get("data_testid"),
        "name": candidate.get("name"),
        "type": candidate.get("type"),
    }
    attributes = {key: value for key, value in attributes.items() if value}
    if attributes:
        item["attributes"] = attributes
    if kind == "assertion":
        item["type"] = _assertion_type(candidate)
    text = _trim_text(candidate.get("text") or "", 80)
    if kind == "interactive" and not (name or text):
        selector = str(candidate.get("selector") or "")
        if _selector_too_generic(selector):
            item["low_confidence"] = True
    return {key: value for key, value in item.items() if value not in (None, "", [])}


def _to_form(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _element_id(candidate, prefix="f"),
        "type": candidate.get("tag"),
        "input_type": candidate.get("type"),
        "label": _accessible_name(candidate),
        "placeholder": candidate.get("placeholder"),
        "name": candidate.get("name"),
        "value_state": "filled" if candidate.get("value") else "empty",
        "visible": candidate.get("visible", True),
        "enabled": candidate.get("enabled", True),
        "selector_candidates": _selector_candidates(candidate),
    }


def _candidate_score(candidate: dict[str, Any], hint_terms: set[str]) -> int:
    tag = str(candidate.get("tag") or "").lower()
    selector = str(candidate.get("selector") or "")
    text_blob = _candidate_blob(candidate)
    score = 0

    if tag in _INTERACTIVE_TAGS:
        score += 50
    elif tag in _IMPORTANT_TEXT_TAGS and _has_text(candidate):
        score += 15
    if candidate.get("role"):
        score += 25
    if selector:
        score += 15
    if candidate.get("id"):
        score += 8
    if candidate.get("data_test") or candidate.get("data_testid"):
        score += 20
    if (
        candidate.get("aria_label")
        or candidate.get("placeholder")
        or candidate.get("label")
    ):
        score += 15
    if _has_text(candidate):
        score += 8
    normalized_blob = _remove_cjk_display_spaces(text_blob)
    if any(
        keyword in text_blob or keyword in normalized_blob
        for keyword in _SIGNAL_KEYWORDS
    ):
        score += 20
    if _is_unrequested_exit_candidate(normalized_blob, hint_terms):
        score -= 60
    if hint_terms:
        overlap = sum(
            1
            for term in hint_terms
            if term and (term in text_blob or term in normalized_blob)
        )
        score += min(30, overlap * 6)
    if candidate.get("visible") is False:
        score -= 40
    if candidate.get("enabled") is False:
        score -= 5
    return score


def _selector_candidates(candidate: dict[str, Any]) -> list[str]:
    return [item.selector for item in build_selector_candidates(candidate)][:5]


def _selector_too_generic(selector: str) -> bool:
    text = str(selector or "").strip()
    if not text:
        return True
    if ":has-text(" in text or "[" in text or "#" in text:
        return False
    return bool(re.fullmatch(r"(?:[a-z]+(?::nth-of-type\(\d+\))?\s*>\s*)*[a-z]+(?::nth-of-type\(\d+\))?", text))


def _element_id(candidate: dict[str, Any], *, prefix: str) -> str:
    index = candidate.get("index")
    if isinstance(index, int):
        return f"{prefix}{index}"
    raw = _candidate_blob(candidate)
    return f"{prefix}{abs(hash(raw)) % 10000}"


def _role(candidate: dict[str, Any]) -> str:
    role = str(candidate.get("role") or "").strip().lower()
    if role:
        return role
    tag = str(candidate.get("tag") or "").lower()
    if tag == "button":
        return "button"
    if tag == "a":
        return "link"
    if tag in {"input", "textarea"}:
        return "textbox"
    if tag == "select":
        return "combobox"
    return tag or "element"


def _accessible_name(candidate: dict[str, Any]) -> str:
    for key in (
        "aria_label",
        "label",
        "placeholder",
        "title",
        "text",
        "name",
        "data_test",
        "id",
    ):
        value = _trim_text(candidate.get(key) or "", 100)
        if value:
            return value
    return ""


def _near_text(candidate: dict[str, Any], own_text: str) -> str:
    ancestor = _trim_text(candidate.get("ancestor_text") or "", 180)
    if not ancestor:
        return ""
    own = str(own_text or "").strip()
    if own:
        ancestor = ancestor.replace(own, " ").strip()
    return _trim_text(ancestor, 160)


def _is_interactive(candidate: dict[str, Any]) -> bool:
    tag = str(candidate.get("tag") or "").lower()
    role = str(candidate.get("role") or "").lower()
    return tag in _INTERACTIVE_TAGS or role in {
        "button",
        "link",
        "checkbox",
        "radio",
        "tab",
        "menuitem",
        "textbox",
    }


def _is_assertion_candidate(candidate: dict[str, Any]) -> bool:
    tag = str(candidate.get("tag") or "").lower()
    blob = _candidate_blob(candidate)
    if tag in {"input", "textarea", "select"}:
        return False
    return (tag in _ASSERTION_TAGS and _has_text(candidate)) or any(
        keyword in blob for keyword in _ASSERTION_KEYWORDS
    )


def _assertion_type(candidate: dict[str, Any]) -> str:
    blob = _candidate_blob(candidate)
    tag = str(candidate.get("tag") or "").lower()
    if tag in {"h1", "h2", "h3"} or "title" in blob:
        return "heading"
    if "badge" in blob:
        return "badge"
    if "error" in blob or "错误" in blob:
        return "error"
    if "dialog" in blob or "modal" in blob:
        return "dialog"
    return "text"


def _visible_text_summary(
    candidates: list[dict[str, Any]], *, limit: int = 12
) -> list[str]:
    result: list[str] = []
    for candidate in candidates:
        for key in ("text", "aria_label", "placeholder", "label", "ancestor_text"):
            text = _trim_text(candidate.get(key) or "", 80)
            if text and text not in result:
                result.append(text)
            if len(result) >= limit:
                return result
    return result


def _main_heading(candidates: list[dict[str, Any]]) -> str:
    for candidate in candidates:
        selector = str(candidate.get("selector") or "").lower()
        text = _trim_text(candidate.get("text") or "", 80)
        tag = str(candidate.get("tag") or "").lower()
        if text and (tag in {"h1", "h2"} or "title" in selector):
            return text
    return ""


def _extract_business_objects(elements: list[dict[str, Any]]) -> dict[str, Any]:
    cards: dict[str, dict[str, Any]] = {}
    for element in elements:
        near_text = str(element.get("near_text") or "")
        card_name = _extract_card_name(near_text)
        if not card_name:
            continue
        card = cards.setdefault(
            card_name,
            {
                "name": card_name,
                "summary": _trim_text(near_text, 140),
                "actions": {},
            },
        )
        action_name = _business_action_name(element)
        if action_name:
            card["actions"][action_name] = {
                "element_id": element["id"],
                "selector_candidates": element.get("selector_candidates", [])[:3],
            }
    return {"cards": list(cards.values())[:20]} if cards else {}


def _extract_card_name(text: str) -> str:
    lines = [
        _trim_text(line.strip(), 80)
        for line in re.split(r"\s{2,}|\n|\r", str(text or ""))
        if line.strip()
    ]
    if lines:
        return lines[0]
    cleaned = _trim_text(text, 80)
    return cleaned if len(cleaned) >= 2 else ""


def _business_action_name(element: dict[str, Any]) -> str:
    blob = _element_blob(element)
    if "add" in blob or "添加" in blob or "加入" in blob:
        return "add"
    if "remove" in blob or "移除" in blob:
        return "remove"
    if "continue" in blob or "下一步" in blob or "继续" in blob:
        return "continue"
    if "submit" in blob or "提交" in blob:
        return "submit"
    return ""


def _route_hint(url: str) -> str:
    path = str(url or "").split("?")[0].rstrip("/").rsplit("/", 1)[-1]
    return path.replace(".html", "") or "root"


def _iter_elements(payload: dict[str, Any] | list[dict[str, Any]]):
    if isinstance(payload, list):
        yield from payload
        return
    if not isinstance(payload, dict):
        return
    for key in ("interactive_elements", "assertion_candidates", "forms"):
        values = payload.get(key)
        if isinstance(values, list):
            yield from values
    dom_context = payload.get("dom_context")
    if isinstance(dom_context, dict):
        yield from _iter_elements(dom_context)
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        yield from candidates


def _element_blob(element: dict[str, Any]) -> str:
    return " ".join(
        _flatten_texts(
            [
                element.get("name"),
                element.get("text"),
                element.get("label"),
                element.get("placeholder"),
                element.get("near_text"),
                element.get("role"),
                element.get("selector_candidates"),
                element.get("attributes"),
            ]
        )
    ).lower()


def _css_attr(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _css_ident(value: Any) -> str:
    return re.sub(r"([^a-zA-Z0-9_-])", r"\\\1", str(value))


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _compact_modules(
    modules: dict[str, Any],
    *,
    module_names: list[str],
    max_steps: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in module_names:
        steps = _module_steps(modules.get(name))
        result[name] = [
            _compact_step(step) for step in steps[:max_steps] if isinstance(step, dict)
        ]
    return result


def _ranked_keys(values: Any, hint_terms: set[str], *, limit: int) -> list[str]:
    ranked: list[tuple[int, str]] = []
    for value in values or []:
        text = str(value or "").strip()
        if not text:
            continue
        blob = text.lower().replace("_", " ").replace("-", " ")
        score = 0
        if hint_terms:
            score += sum(3 for term in hint_terms if term and term in blob)
        if any(keyword in blob for keyword in _SIGNAL_KEYWORDS):
            score += 2
        ranked.append((score, text))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    result: list[str] = []
    for _, text in ranked:
        if text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _trim_business_object(item: dict[str, Any]) -> dict[str, Any]:
    result = {
        "name": _trim_text(item.get("name") or "", 80),
        "summary": _trim_text(item.get("summary") or "", 140),
    }
    actions = item.get("actions")
    if isinstance(actions, dict):
        compact_actions: dict[str, Any] = {}
        for action_name, action_value in list(actions.items())[:4]:
            if not isinstance(action_value, dict):
                continue
            compact_actions[action_name] = {
                "element_id": action_value.get("element_id"),
                "selector_candidates": list(
                    action_value.get("selector_candidates") or []
                )[:2],
            }
        if compact_actions:
            result["actions"] = compact_actions
    return {
        key: value for key, value in result.items() if value not in (None, "", [], {})
    }


def _module_steps(module: Any) -> list[Any]:
    if isinstance(module, dict):
        steps = module.get("steps")
        return steps if isinstance(steps, list) else []
    return module if isinstance(module, list) else []


def _compact_step(step: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("action", "use_module", "selector", "target", "value", "params"):
        value = step.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, (str, int, float, bool)):
            result[key] = _trim_text(value, 140)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _candidate_blob(candidate: dict[str, Any]) -> str:
    values = [
        candidate.get(key)
        for key in (
            "selector",
            "id",
            "data_test",
            "data_testid",
            "role",
            "text",
            "aria_label",
            "placeholder",
            "title",
            "name",
            "label",
            "ancestor_text",
        )
    ]
    return " ".join(_flatten_texts(values)).lower()


def _has_text(candidate: dict[str, Any]) -> bool:
    return any(
        str(candidate.get(key) or "").strip()
        for key in ("text", "aria_label", "placeholder", "label", "ancestor_text")
    )


def _terms(value: str) -> set[str]:
    text = _trim_text(value, 1200).lower()
    terms = {
        part
        for part in text.replace("_", " ").replace("-", " ").split()
        if len(part) >= 2
    }
    for keyword in _SIGNAL_KEYWORDS:
        normalized_keyword = str(keyword).lower()
        if normalized_keyword and normalized_keyword in text:
            terms.add(normalized_keyword)
    for variant in semantic_text_variants(text) or [text]:
        for chunk in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", variant):
            normalized = _strip_generic_target_words(chunk).lower()
            if len(normalized) >= 2:
                terms.add(normalized)
                terms.add(_remove_cjk_display_spaces(normalized))
            for token in re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]{3,}", normalized):
                terms.add(token)
                match = re.match(r"([a-z]{2,})\d{2,}", token)
                if match:
                    terms.add(match.group(1))
    return set(list(terms)[:80])


def _is_unrequested_exit_candidate(text_blob: str, hint_terms: set[str]) -> bool:
    if not any(term in text_blob for term in ("退出登录", "退出", "登出", "注销", "logout")):
        return False
    return not any(
        term in hint_terms for term in ("退出", "登出", "注销", "logout", "signout")
    )


def _strip_generic_target_words(value: str) -> str:
    return _shared_strip_generic_target_words(value)


def _remove_cjk_display_spaces(value: str) -> str:
    text = str(value or "")
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", text)
    return text


def _flatten_texts(values: list[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, str):
            result.append(value)
        elif isinstance(value, dict):
            result.extend(_flatten_texts(list(value.values())))
        elif isinstance(value, list):
            result.extend(_flatten_texts(value))
        elif value is not None:
            result.append(str(value))
    return result


def _trim_text(value: Any, limit: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."
