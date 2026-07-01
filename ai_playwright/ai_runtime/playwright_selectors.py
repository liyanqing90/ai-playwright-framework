from __future__ import annotations

import re
from typing import Any

from ai_playwright.ai_runtime.native_observe import SelectorValidation
from ai_playwright.ai_runtime.redaction import redact_value
from ai_playwright.ai_runtime.semantic_terms import (
    semantic_text_variants,
    strip_generic_target_words as _shared_strip_generic_target_words,
)

_LOGOUT_TARGET_TERMS = (
    "退出登录",
    "登出",
    "注销",
    "logout",
    "log out",
    "sign out",
    "signout",
)
_LOGOUT_CANDIDATE_TERMS = _LOGOUT_TARGET_TERMS + ("logout-sidebar",)
_QUERY_TARGET_TERMS = (
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
)
_PRODUCT_ID_TARGET_TERMS = (
    "商品id",
    "商品编号",
    "商品编码",
    "spu",
    "spucode",
    "spu code",
)
_LOG_TARGET_TERMS = (
    "查看日志",
    "日志",
    "记录",
    "view log",
    "logs",
    "log",
)
_FILLABLE_ELEMENT_SCRIPT = """el => {
    const tag = el.tagName.toLowerCase();
    if (tag === 'input') {
        const type = (el.getAttribute('type') || 'text').toLowerCase();
        return ![
            'button',
            'checkbox',
            'file',
            'hidden',
            'image',
            'radio',
            'reset',
            'submit'
        ].includes(type);
    }
    return tag === 'textarea'
        || el.isContentEditable === true
        || el.getAttribute('role') === 'textbox';
}"""


_CLEARABLE_CONTROL_TARGET_SCRIPT = """el => {
    const controlSelector = [
        '.ant-select',
        '.ant-picker',
        '.ant-input-affix-wrapper'
    ].join(',');
    const clearSelector = [
        '.ant-select-clear',
        '.ant-picker-clear',
        '.ant-input-clear-icon',
        '.anticon-close-circle',
        '[aria-label="close-circle"]',
        '[aria-label="CloseCircle"]',
        '[class*="clear" i]'
    ].join(',');
    const valueSelector = [
        '.ant-select-selection-item',
        '.ant-select-selection-item-content',
        '.ant-picker-input input',
        'input',
        'textarea'
    ].join(',');
    const isElement = node => node && node.nodeType === Node.ELEMENT_NODE;
    const isVisible = node => {
        if (!isElement(node)) return false;
        const style = window.getComputedStyle(node);
        const rect = node.getBoundingClientRect();
        return style.display !== 'none'
            && style.visibility !== 'hidden'
            && rect.width > 0
            && rect.height > 0;
    };
    if (isVisible(el)) {
        const tag = el.tagName.toLowerCase();
        if (['input', 'textarea', 'select'].includes(tag) || el.isContentEditable) {
            return true;
        }
    }
    const control = isElement(el) ? el.closest(controlSelector) : null;
    if (!control || !isVisible(control)) return false;
    let hasClearIcon = false;
    try { hasClearIcon = Boolean(control.querySelector(clearSelector)); } catch {}
    let hasValue = false;
    try {
        hasValue = Array.from(control.querySelectorAll(valueSelector)).some(node => {
            const text = String(
                node.getAttribute('title') || node.textContent || node.value || ''
            ).trim();
            return Boolean(text);
        });
    } catch {}
    return hasClearIcon || hasValue || control.classList.contains('ant-select-allow-clear');
}"""


def normalize_selector(selector: str, selector_type: str | None = None) -> str:
    selector = str(selector).strip()
    if selector.startswith("css="):
        return selector.removeprefix("css=")
    if selector.startswith("xpath="):
        return selector.removeprefix("xpath=")
    if selector_type == "xpath" and not selector.startswith(("//", "(//")):
        return f"//{selector}"
    return selector


def canonicalize_persisted_selector(selector: str) -> str:
    selector = normalize_selector(selector)
    match = re.fullmatch(
        r"([a-zA-Z][\w-]*):has-text\(\s*([\"'])(.*?)\2\s*\)",
        selector,
        flags=re.DOTALL,
    )
    if not match:
        return selector

    tag = match.group(1).lower()
    text = match.group(3).strip()
    if not _has_cjk_display_space(text):
        return selector
    return f"//{tag}[normalize-space()={_xpath_literal(text)}]"


def is_fillable_element(locator: Any) -> bool:
    return bool(locator.evaluate(_FILLABLE_ELEMENT_SCRIPT))


def is_clearable_control_target(locator: Any) -> bool:
    return bool(locator.evaluate(_CLEARABLE_CONTROL_TARGET_SCRIPT))


def is_checkable_control(locator: Any) -> bool:
    return bool(
        locator.evaluate(
            """el => {
                const tag = el.tagName.toLowerCase();
                const role = (el.getAttribute('role') || '').toLowerCase();
                if (tag === 'input') {
                    const type = (el.getAttribute('type') || '').toLowerCase();
                    return type === 'checkbox' || type === 'radio';
                }
                return role === 'checkbox' || role === 'radio';
            }"""
        )
    )


def validate_selector(
    page,
    selector: str,
    *,
    action: str,
    timeout: int,
    require_unique: bool = False,
) -> SelectorValidation:
    locator = page.locator(selector)
    first = locator.first
    normalized_action = str(action or "").lower()
    wait_state = "attached" if normalized_action == "clear" else "visible"
    try:
        first.wait_for(state=wait_state, timeout=timeout)
    except Exception as exc:
        match_count = _safe_locator_count(locator)
        return SelectorValidation(
            selector=selector,
            action=action,
            ok=False,
            match_count=match_count,
            visible_count=_safe_visible_count(locator, match_count),
            error=str(exc),
        )

    match_count = _safe_locator_count(locator)
    if match_count == 0:
        return SelectorValidation(
            selector=selector,
            action=action,
            ok=False,
            match_count=0,
            visible_count=0,
            error="selector matched no elements",
        )
    if require_unique and match_count is not None and match_count != 1:
        return SelectorValidation(
            selector=selector,
            action=action,
            ok=False,
            match_count=match_count,
            visible_count=_safe_visible_count(locator, match_count),
            error=f"selector matched {match_count} elements",
        )

    visible_count = _safe_visible_count(locator, match_count)
    enabled: bool | None = None
    action_compatible: bool | None = None
    if normalized_action in {"click", "press", "press_key"}:
        try:
            enabled = bool(first.is_enabled())
        except Exception as exc:
            return SelectorValidation(
                selector=selector,
                action=action,
                ok=False,
                match_count=match_count,
                visible_count=visible_count,
                error=str(exc),
                locator=first,
            )
        if not enabled:
            return SelectorValidation(
                selector=selector,
                action=action,
                ok=False,
                match_count=match_count,
                visible_count=visible_count,
                enabled=False,
                action_compatible=False,
                error="selector target is disabled",
                locator=first,
            )
        action_compatible = True
    elif normalized_action == "fill":
        try:
            action_compatible = is_fillable_element(first)
        except Exception as exc:
            return SelectorValidation(
                selector=selector,
                action=action,
                ok=False,
                match_count=match_count,
                visible_count=visible_count,
                enabled=enabled,
                action_compatible=False,
                error=str(exc),
                locator=first,
            )
        if not action_compatible:
            return SelectorValidation(
                selector=selector,
                action=action,
                ok=False,
                match_count=match_count,
                visible_count=visible_count,
                enabled=enabled,
                action_compatible=False,
                error="selector target is not fillable",
                locator=first,
            )
    elif normalized_action == "clear":
        try:
            action_compatible = is_clearable_control_target(first)
        except Exception as exc:
            return SelectorValidation(
                selector=selector,
                action=action,
                ok=False,
                match_count=match_count,
                visible_count=visible_count,
                enabled=enabled,
                action_compatible=False,
                error=str(exc),
                locator=first,
            )
        if not action_compatible:
            return SelectorValidation(
                selector=selector,
                action=action,
                ok=False,
                match_count=match_count,
                visible_count=visible_count,
                enabled=enabled,
                action_compatible=False,
                error="selector target is not clearable",
                locator=first,
            )
    elif normalized_action in {"check", "uncheck", "set_checked"}:
        try:
            action_compatible = is_checkable_control(first)
        except Exception as exc:
            return SelectorValidation(
                selector=selector,
                action=action,
                ok=False,
                match_count=match_count,
                visible_count=visible_count,
                enabled=enabled,
                action_compatible=False,
                error=str(exc),
                locator=first,
            )
        if not action_compatible:
            return SelectorValidation(
                selector=selector,
                action=action,
                ok=False,
                match_count=match_count,
                visible_count=visible_count,
                enabled=enabled,
                action_compatible=False,
                error="selector target is not checkable",
                locator=first,
            )
    return SelectorValidation(
        selector=selector,
        action=action,
        ok=True,
        match_count=match_count,
        visible_count=visible_count,
        enabled=enabled,
        action_compatible=action_compatible,
        locator=first,
    )


def verify_selector(
    page,
    selector: str,
    *,
    action: str,
    timeout: int,
    require_unique: bool = False,
) -> bool:
    validation = validate_selector(
        page,
        selector,
        action=action,
        timeout=timeout,
        require_unique=require_unique,
    )
    if not validation.ok:
        raise ValueError(validation.error or f"selector validation failed: {selector}")
    return True


def is_high_quality_selector(selector: str) -> bool:
    selector_l = str(selector or "").lower().strip()
    if not selector_l:
        return False
    if any(
        marker in selector_l
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
            "[role=",
            "text=",
            "#",
            "//",
            "(//",
        )
    ):
        return True
    return not _looks_like_structural_selector(selector_l)


def selector_probe_score(
    selector: str, action: str | None = None, target: str | None = None
) -> float:
    selector_l = str(selector or "").lower().strip()
    if not selector_l:
        return -100.0
    score = _selector_quality_score(selector_l)
    normalized_action = str(action or "").lower()
    has_tag_prefix = bool(re.match(r"^[a-z][\w-]*\[", selector_l))
    target_l = _remove_cjk_display_spaces(_semantic_target_text(target).lower())
    is_query_click = normalized_action in {
        "click",
        "press",
        "press_key",
    } and _contains_any(
        target_l,
        list(_QUERY_TARGET_TERMS),
    )
    target_mentions_menu = _contains_any(target_l, ["菜单", "menu", "导航", "侧边栏"])

    if _looks_like_structural_selector(selector_l):
        score -= 80
    if re.search(
        r'^[a-z][\w-]*\[type\s*=\s*["\']?(?:password|search)["\']?', selector_l
    ):
        score += 50
    if re.search(r'^[a-z][\w-]*\[type\s*=\s*["\']?(?:submit|button)["\']?', selector_l):
        score += 42
    if has_tag_prefix and "*=" in selector_l:
        score += 18
    elif selector_l.startswith("[") and "*=" in selector_l:
        score -= 10
    if normalized_action == "fill" and selector_l.startswith(("input", "textarea")):
        score += 16
    if normalized_action in {"click", "press", "press_key"} and selector_l.startswith(
        ("button", "a", "input")
    ):
        score += 10
    if is_query_click:
        menu_like = any(
            marker in selector_l
            for marker in (
                "menuitem",
                "ant-pro-menu-item-title",
                "ant-menu",
                'role="menuitem"',
                "role='menuitem'",
            )
        )
        button_like = selector_l.startswith(("button", "//button", "(//button")) or (
            "//button[" in selector_l
        )
        if menu_like and not target_mentions_menu:
            score -= 80
        if button_like:
            score += 35
            if "translate(normalize-space()" in selector_l:
                score += 12
        if selector_l.startswith('input[type="search"') or selector_l.startswith(
            "input[type='search'"
        ):
            score -= 45
    if selector_l.startswith(("text=", "*:has-text(")):
        score -= 12
    return score


def _safe_locator_count(locator) -> int | None:
    try:
        return int(locator.count())
    except Exception:
        return None


def _safe_visible_count(locator, match_count: int | None) -> int | None:
    if match_count is None:
        return None
    visible = 0
    for index in range(min(match_count, 20)):
        try:
            if locator.nth(index).is_visible():
                visible += 1
        except Exception:
            return None
    return visible


def stable_selector_for_locator(locator) -> str:
    return locator.first.evaluate(
        """el => {
            const cssEscape = value => {
                if (window.CSS && CSS.escape) return CSS.escape(value);
                return String(value).replace(/["\\\\]/g, '\\\\$&');
            };
            const quote = value => String(value).replace(/["\\\\]/g, '\\\\$&');
            const visibleText = node => (node.innerText || node.textContent || '').trim().replace(/\\s+/g, ' ');
            const safeQueryAll = selector => {
                try {
                    return Array.from(document.querySelectorAll(selector));
                } catch {
                    return [];
                }
            };
            const isUniqueCss = selector => safeQueryAll(selector).length === 1;
            const isStableId = value => {
                if (!value) return false;
                const text = String(value);
                if (text.length > 80) return false;
                if (/^[0-9]+$/.test(text)) return false;
                if (/^[a-f0-9]{8,}$/i.test(text)) return false;
                if (/[0-9]{8,}/.test(text)) return false;
                if (/^ember\\d+$/i.test(text)) return false;
                return true;
            };
            const attrSelector = (node, attr) => {
                const value = node.getAttribute(attr);
                if (!value) return null;
                return `${node.tagName.toLowerCase()}[${attr}="${quote(value)}"]`;
            };
            const stableClassSelector = node => {
                const className = typeof node.className === 'string' ? node.className : '';
                if (!className) return null;
                const tokens = className.split(/\\s+/).filter(Boolean);
                const stableToken = token => {
                    if (!token || token.length > 80) return false;
                    if (/^[0-9]+$/.test(token)) return false;
                    if (/^[a-f0-9]{8,}$/i.test(token)) return false;
                    if (/[0-9]{8,}/.test(token)) return false;
                    if (/^css-[a-z0-9]+$/i.test(token)) return false;
                    return /[a-z\u4e00-\u9fff]/i.test(token);
                };
                for (const token of tokens) {
                    if (!stableToken(token)) continue;
                    const selector = `${node.tagName.toLowerCase()}.${cssEscape(token)}`;
                    if (isUniqueCss(selector)) return selector;
                }
                return null;
            };
            const inputValueSelector = node => {
                const tag = node.tagName.toLowerCase();
                if (tag !== 'input') return null;
                const type = (node.getAttribute('type') || '').toLowerCase();
                if (!['submit', 'button', 'reset'].includes(type)) return null;
                const value = node.getAttribute('value');
                if (!value || value.length > 80) return null;
                const selector = `${tag}[type="${quote(type)}"][value="${quote(value)}"]`;
                return isUniqueCss(selector) ? selector : null;
            };
            const exactTextSelector = node => {
                const tag = node.tagName.toLowerCase();
                const text = visibleText(node);
                if (!text || text.length > 80) return null;
                const role = node.getAttribute('role');
                if (role) {
                    const roleSelector = `${tag}[role="${quote(role)}"]:has-text("${quote(text)}")`;
                    if (safeQueryAll(`${tag}[role="${quote(role)}"]`).filter(item => visibleText(item) === text).length === 1) {
                        return roleSelector;
                    }
                }
                if (!['button', 'a', 'label'].includes(tag)) return null;
                const same = safeQueryAll(tag).filter(item => visibleText(item) === text);
                if (same.length !== 1) return null;
                return `${tag}:has-text("${quote(text)}")`;
            };
            const localSelectors = (node, includeText=true) => {
                const selectors = [];
                for (const attr of ['data-testid', 'data-test', 'data-qa', 'data-cy', 'data-ui']) {
                    const selector = attrSelector(node, attr);
                    if (selector) selectors.push(selector);
                }
                if (node.id && isStableId(node.id)) selectors.push(`#${cssEscape(node.id)}`);
                for (const attr of ['name', 'aria-label', 'placeholder', 'title']) {
                    const selector = attrSelector(node, attr);
                    if (selector) selectors.push(selector);
                }
                const classSelector = stableClassSelector(node);
                if (classSelector) selectors.push(classSelector);
                const valueSelector = inputValueSelector(node);
                if (valueSelector) selectors.push(valueSelector);
                const role = node.getAttribute('role');
                const aria = node.getAttribute('aria-label');
                if (role && aria) {
                    selectors.push(`${node.tagName.toLowerCase()}[role="${quote(role)}"][aria-label="${quote(aria)}"]`);
                }
                if (includeText) {
                    const textSelector = exactTextSelector(node);
                    if (textSelector) selectors.push(textSelector);
                }
                return selectors;
            };
            for (const selector of localSelectors(el)) {
                if (selector.includes(':has-text(') || isUniqueCss(selector)) return selector;
            };
            let ancestor = el.parentElement;
            while (ancestor && ancestor !== document.body) {
                for (const ancestorSelector of localSelectors(ancestor, false)) {
                    if (!isUniqueCss(ancestorSelector)) continue;
                    for (const childSelector of localSelectors(el, false)) {
                        if (childSelector.includes(':has-text(')) continue;
                        const scoped = `${ancestorSelector} ${childSelector}`;
                        if (isUniqueCss(scoped)) return scoped;
                    }
                }
                ancestor = ancestor.parentElement;
            };
            const parts = [];
            let node = el;
            while (node && node.nodeType === Node.ELEMENT_NODE && node !== document.body) {
                let part = node.tagName.toLowerCase();
                const parent = node.parentElement;
                if (!parent) break;
                const same = Array.from(parent.children).filter(child => child.tagName === node.tagName);
                if (same.length > 1) part += `:nth-of-type(${same.indexOf(node) + 1})`;
                parts.unshift(part);
                const path = parts.join(' > ');
                if (parts.length >= 2 && isUniqueCss(path)) return path;
                node = parent;
            }
            return parts.join(' > ');
        }"""
    )


def collect_candidates_diagnostic(
    page,
    *,
    limit: int = 120,
    ignore_selectors: list[str] | tuple[str, ...] | None = None,
    include_open_shadow_dom: bool = True,
    time_budget_ms: int | None = None,
) -> dict[str, Any]:
    result = page.evaluate(
        """opts => {
            const limit = opts.limit || 120;
            const timeBudgetMs = Math.max(0, Number(opts.time_budget_ms || 0));
            const startedAt = performance.now();
            const deadlineAt = timeBudgetMs > 0 ? startedAt + timeBudgetMs : 0;
            let timedOut = false;
            let timeoutStage = '';
            const markStage = stage => {
                if (!deadlineAt || performance.now() <= deadlineAt) return false;
                timedOut = true;
                if (!timeoutStage) timeoutStage = stage;
                return true;
            };
            const candidateBudget = Math.max(limit * 4, 240);
            const textNodeScanLimit = Math.max(limit * 8, 500);
            const ignoreSelectors = Array.isArray(opts.ignore_selectors) ? opts.ignore_selectors : [];
            const includeOpenShadowDom = opts.include_open_shadow_dom !== false;
            let pageTitle = '';
            try {
                pageTitle = document.title || '';
            } catch {
                pageTitle = '';
            }
            const viewport = {
                width: window.innerWidth || document.documentElement.clientWidth || 1,
                height: window.innerHeight || document.documentElement.clientHeight || 1
            };
            const compactText = value => String(value || '').trim().replace(/\\s+/g, ' ');
            const nodeText = node => compactText(node.textContent || '');
            const cssEscape = value => {
                if (window.CSS && CSS.escape) return CSS.escape(value);
                return String(value).replace(/["\\\\]/g, '\\\\$&');
            };
            const attrSelector = (node, attr) => {
                const value = node.getAttribute(attr);
                if (!value) return null;
                return `${node.tagName.toLowerCase()}[${attr}="${String(value).replace(/["\\\\]/g, '\\\\$&')}"]`;
            };
            const safeMatches = (node, selector) => {
                try {
                    return node.matches && node.matches(selector);
                } catch {
                    return false;
                }
            };
            const safeClosest = (node, selector) => {
                try {
                    return node.closest && node.closest(selector);
                } catch {
                    return null;
                }
            };
            const ignored = el => ignoreSelectors.some(selector => {
                if (!selector) return false;
                return safeMatches(el, selector) || !!safeClosest(el, selector);
            });
            const boundedVisibleBox = el => {
                const rect = el.getBoundingClientRect();
                if (!rect.width || !rect.height) return null;
                const area = rect.width * rect.height;
                const viewportArea = Math.max(1, viewport.width * viewport.height);
                if (area > viewportArea * 0.35 || rect.width < 4 || rect.height < 4) return null;
                return rect;
            };
            const visibleTextPool = (() => {
                const result = [];
                let nodes = [];
                try {
                    nodes = Array.from(document.querySelectorAll('label, span, div'));
                } catch {
                    nodes = [];
                }
                for (const node of nodes) {
                    if (markStage('text_pool')) break;
                    if (result.length >= textNodeScanLimit) break;
                    const text = nodeText(node);
                    if (!text || text.length > 40) continue;
                    const box = boundedVisibleBox(node);
                    if (!box) continue;
                    result.push({ text, box });
                }
                return result;
            })();
            const labelText = el => {
                const labels = [];
                if (el.id) {
                    const explicit = document.querySelector(`label[for="${cssEscape(el.id)}"]`);
                    if (explicit) labels.push(nodeText(explicit));
                }
                let parent = el.parentElement;
                while (parent && parent !== document.body) {
                    if (parent.tagName && parent.tagName.toLowerCase() === 'label') {
                        labels.push(nodeText(parent));
                        break;
                    }
                    parent = parent.parentElement;
                }
                const formItem = el.closest([
                    '.ant-form-item',
                    '[class*="form-item" i]',
                    '[class*="formItem"]',
                    'fieldset'
                ].join(','));
                if (formItem) {
                    const labelNode = formItem.querySelector([
                        'label',
                        '.ant-form-item-label',
                        '[class*="label" i]'
                    ].join(','));
                    if (labelNode) labels.push(nodeText(labelNode));
                }
                const rect = el.getBoundingClientRect();
                const nearby = visibleTextPool
                    .map(item => {
                        const box = item.box;
                        const verticalOverlap = Math.min(rect.bottom, box.bottom) - Math.max(rect.top, box.top);
                        const distance = rect.left - box.right;
                        if (verticalOverlap <= 0 || distance < -8 || distance > 260) return null;
                        return { text: item.text, distance };
                    })
                    .filter(Boolean)
                    .sort((a, b) => a.distance - b.distance)[0];
                if (nearby) labels.push(nearby.text);
                return labels.join(' ').trim().slice(0, 180);
            };
            const ancestorText = el => {
                const ancestor = el.closest([
                    '[data-test]', '[data-testid]', '[data-qa]',
                    'li', 'tr', 'form', 'section', 'article', 'fieldset',
                    '[role="row"]', '[role="listitem"]',
                    '[class*="card" i]', '[class*="item" i]', '[class*="row" i]'
                ].join(','));
                if (!ancestor || ancestor === el) return '';
                return nodeText(ancestor).slice(0, 300);
            };
            const stableSelector = el => {
                const quote = value => String(value).replace(/["\\\\]/g, '\\\\$&');
                const visibleText = node => (node.innerText || node.textContent || '').trim().replace(/\\s+/g, ' ');
                const safeQueryAll = selector => {
                    try {
                        return Array.from(document.querySelectorAll(selector));
                    } catch {
                        return [];
                    }
                };
                const isUniqueCss = selector => safeQueryAll(selector).length === 1;
                const isStableId = value => {
                    if (!value) return false;
                    const text = String(value);
                    if (text.length > 80) return false;
                    if (/^[0-9]+$/.test(text)) return false;
                    if (/^[a-f0-9]{8,}$/i.test(text)) return false;
                    if (/[0-9]{8,}/.test(text)) return false;
                    if (/^ember\\d+$/i.test(text)) return false;
                    return true;
                };
                const inputValueSelector = node => {
                    const tag = node.tagName.toLowerCase();
                    if (tag !== 'input') return null;
                    const type = (node.getAttribute('type') || '').toLowerCase();
                    if (!['submit', 'button', 'reset'].includes(type)) return null;
                    const value = node.getAttribute('value');
                    if (!value || value.length > 80) return null;
                    const selector = `${tag}[type="${quote(type)}"][value="${quote(value)}"]`;
                    return isUniqueCss(selector) ? selector : null;
                };
                const exactTextSelector = node => {
                    const tag = node.tagName.toLowerCase();
                    const text = visibleText(node);
                    if (!text || text.length > 80) return null;
                    const role = node.getAttribute('role');
                    if (role) {
                        const roleSelector = `${tag}[role="${quote(role)}"]:has-text("${quote(text)}")`;
                        if (safeQueryAll(`${tag}[role="${quote(role)}"]`).filter(item => visibleText(item) === text).length === 1) {
                            return roleSelector;
                        }
                    }
                    if (!['button', 'a', 'label'].includes(tag)) return null;
                    const same = safeQueryAll(tag).filter(item => visibleText(item) === text);
                    if (same.length !== 1) return null;
                    return `${tag}:has-text("${quote(text)}")`;
                };
                const stableClassSelector = node => {
                    const className = typeof node.className === 'string' ? node.className : '';
                    if (!className) return null;
                    const tokens = className.split(/\\s+/).filter(Boolean);
                    const stableToken = token => {
                        if (!token || token.length > 80) return false;
                        if (/^[0-9]+$/.test(token)) return false;
                        if (/^[a-f0-9]{8,}$/i.test(token)) return false;
                        if (/[0-9]{8,}/.test(token)) return false;
                        if (/^css-[a-z0-9]+$/i.test(token)) return false;
                        return /[a-z\u4e00-\u9fff]/i.test(token);
                    };
                    for (const token of tokens) {
                        if (!stableToken(token)) continue;
                        const selector = `${node.tagName.toLowerCase()}.${cssEscape(token)}`;
                        if (isUniqueCss(selector)) return selector;
                    }
                    return null;
                };
                const localSelectors = (node, includeText=true) => {
                    const selectors = [];
                    for (const attr of ['data-testid', 'data-test', 'data-qa', 'data-cy', 'data-ui']) {
                        const selector = attrSelector(node, attr);
                        if (selector) selectors.push(selector);
                    }
                    if (node.id && isStableId(node.id)) selectors.push(`#${cssEscape(node.id)}`);
                    for (const attr of ['name', 'aria-label', 'placeholder', 'title']) {
                        const selector = attrSelector(node, attr);
                        if (selector) selectors.push(selector);
                    }
                    const classSelector = stableClassSelector(node);
                    if (classSelector) selectors.push(classSelector);
                    const valueSelector = inputValueSelector(node);
                    if (valueSelector) selectors.push(valueSelector);
                    const role = node.getAttribute('role');
                    const aria = node.getAttribute('aria-label');
                    if (role && aria) {
                        selectors.push(`${node.tagName.toLowerCase()}[role="${quote(role)}"][aria-label="${quote(aria)}"]`);
                    }
                    if (includeText) {
                        const textSelector = exactTextSelector(node);
                        if (textSelector) selectors.push(textSelector);
                    }
                    return selectors;
                };
                for (const selector of localSelectors(el)) {
                    if (selector.includes(':has-text(') || isUniqueCss(selector)) return selector;
                }
                let ancestor = el.parentElement;
                while (ancestor && ancestor !== document.body) {
                    for (const ancestorSelector of localSelectors(ancestor, false)) {
                        if (!isUniqueCss(ancestorSelector)) continue;
                        for (const childSelector of localSelectors(el, false)) {
                            if (childSelector.includes(':has-text(')) continue;
                            const scoped = `${ancestorSelector} ${childSelector}`;
                            if (isUniqueCss(scoped)) return scoped;
                        }
                    }
                    ancestor = ancestor.parentElement;
                }
                const parts = [];
                let node = el;
                while (node && node.nodeType === Node.ELEMENT_NODE && node !== document.body) {
                    let part = node.tagName.toLowerCase();
                    const parent = node.parentElement;
                    if (!parent) break;
                    const same = Array.from(parent.children).filter(child => child.tagName === node.tagName);
                    if (same.length > 1) part += `:nth-of-type(${same.indexOf(node) + 1})`;
                    parts.unshift(part);
                    const path = parts.join(' > ');
                    if (parts.length >= 2 && isUniqueCss(path)) return path;
                    node = parent;
                }
                return parts.join(' > ');
            };
            const candidateSelector = [
                'input', 'textarea', 'button', 'a', 'select', '[role]',
                '[aria-label]', '[placeholder]', 'label', '[onclick]',
                '[tabindex]', '[title]', '[class*="close" i]',
                '[class*="cancel" i]', '[class*="popup" i]',
                '[class*="modal" i]', '[class*="dialog" i]',
                '[data-test]', '[data-testid]', '[data-qa]', '[data-cy]', '[data-ui]',
                '[class*="badge" i]', '[class*="error" i]'
            ].join(',');
            const nodes = [];
            const seen = new Set();
            const addNode = el => {
                if (!el || seen.has(el)) return;
                if (nodes.length >= candidateBudget) return;
                seen.add(el);
                nodes.push(el);
            };
            const ownText = el => Array.from(el.childNodes || [])
                .filter(node => node.nodeType === Node.TEXT_NODE)
                .map(node => node.nodeValue || '')
                .join(' ')
                .trim()
                .replace(/\\s+/g, ' ');
            const hasStableSelector = el => {
                const selector = stableSelector(el);
                return !!selector && !/^(?:div|span|li|i)(?::nth-of-type\\(\\d+\\))?$/.test(selector);
            };
            const hasStableClassToken = el => {
                const className = typeof el.className === 'string' ? el.className : '';
                if (!className) return false;
                return className.split(/\\s+/).some(token => {
                    if (!token || token.length > 80) return false;
                    if (/^[0-9]+$/.test(token)) return false;
                    if (/^[a-f0-9]{8,}$/i.test(token)) return false;
                    if (/[0-9]{8,}/.test(token)) return false;
                    if (/^css-[a-z0-9]+$/i.test(token)) return false;
                    return /[a-z\u4e00-\u9fff]/i.test(token);
                });
            };
            const hasStableSelectorSignal = el => {
                if (el.id) return true;
                if (['data-testid', 'data-test', 'data-qa', 'data-cy', 'data-ui', 'aria-label', 'title'].some(attr => el.getAttribute(attr))) {
                    return true;
                }
                return hasStableClassToken(el);
            };
            const genericVisibleCandidate = el => {
                const tag = (el.tagName || '').toLowerCase();
                if (!['div', 'span', 'li', 'i'].includes(tag)) return false;
                if (!boundedVisibleBox(el)) return false;
                const text = ownText(el);
                if (text && text.length <= 80) return true;
                return hasStableSelectorSignal(el) && (el.children || []).length <= 8;
            };
            const scanRoot = (root, inShadow=false) => {
                if (markStage(inShadow ? 'shadow_root_scan' : 'root_scan')) return;
                let found = [];
                try {
                    found = Array.from(root.querySelectorAll(candidateSelector));
                } catch {
                    found = [];
                }
                for (const el of found) {
                    if (markStage(inShadow ? 'shadow_candidate_scan' : 'candidate_selector_scan')) break;
                    if (nodes.length >= candidateBudget) break;
                    if (inShadow) el.__uiAutoInShadow = true;
                    addNode(el);
                    if (includeOpenShadowDom && el.shadowRoot) {
                        scanRoot(el.shadowRoot, true);
                    }
                }
                if (includeOpenShadowDom) {
                    let shadowHosts = [];
                    try {
                        shadowHosts = Array.from(root.querySelectorAll('*')).filter(el => el.shadowRoot);
                    } catch {
                        shadowHosts = [];
                    }
                    for (const host of shadowHosts) {
                        if (markStage('shadow_host_scan')) break;
                        if (nodes.length >= candidateBudget) break;
                        scanRoot(host.shadowRoot, true);
                    }
                }
                if (markStage(inShadow ? 'shadow_generic_prefilter' : 'generic_prefilter')) return;
                let generic = [];
                try {
                    generic = Array.from(root.querySelectorAll('div,span,li,i'));
                } catch {
                    generic = [];
                }
                for (const el of generic) {
                    if (markStage(inShadow ? 'shadow_generic_scan' : 'generic_scan')) break;
                    if (nodes.length >= candidateBudget) break;
                    if (!genericVisibleCandidate(el)) continue;
                    if (inShadow) el.__uiAutoInShadow = true;
                    addNode(el);
                }
            };
            scanRoot(document, false);
            const visibleNodes = nodes
                .filter(el => !ignored(el))
                .filter(el => !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length))
                .slice(0, limit);
            const candidates = [];
            for (const el of visibleNodes) {
                if (markStage('candidate_serialization')) break;
                const index = candidates.length;
                try {
                    const rect = el.getBoundingClientRect();
                    const x1 = Math.max(0, rect.left);
                    const y1 = Math.max(0, rect.top);
                    const x2 = Math.min(viewport.width, rect.right);
                    const y2 = Math.min(viewport.height, rect.bottom);
                    const centerX = x1 + (x2 - x1) / 2;
                    const centerY = y1 + (y2 - y1) / 2;
                    const candidate = {
                        index,
                        tag: el.tagName.toLowerCase(),
                        selector: stableSelector(el),
                        id: el.id || null,
                        class_name: el.className || null,
                        data_test: el.getAttribute('data-test'),
                        data_testid: el.getAttribute('data-testid'),
                        text: (el.innerText || el.textContent || '').trim().slice(0, 180),
                        value: el.getAttribute('value'),
                        role: el.getAttribute('role'),
                        aria_label: el.getAttribute('aria-label'),
                        placeholder: el.getAttribute('placeholder'),
                        title: el.getAttribute('title'),
                        name: el.getAttribute('name'),
                        type: el.getAttribute('type'),
                        label: labelText(el),
                        ancestor_text: ancestorText(el),
                        visible: true,
                        enabled: !el.disabled && el.getAttribute('aria-disabled') !== 'true',
                        in_shadow: !!el.__uiAutoInShadow,
                        bbox: [x1, y1, x2, y2],
                        center: [centerX, centerY],
                        bbox_norm: [x1 / viewport.width, y1 / viewport.height, x2 / viewport.width, y2 / viewport.height],
                        center_norm: [centerX / viewport.width, centerY / viewport.height],
                    };
                    candidates.push(candidate);
                } catch {
                    continue;
                }
            }
            return {
                candidates,
                timed_out: timedOut,
                timeout_stage: timeoutStage,
                title: pageTitle,
                elapsed_ms: Math.round(performance.now() - startedAt),
                candidate_budget: candidateBudget,
            };
        }""",
        {
            "limit": limit,
            "ignore_selectors": list(ignore_selectors or ()),
            "include_open_shadow_dom": include_open_shadow_dom,
            "time_budget_ms": int(time_budget_ms or 0),
        },
    )
    if not isinstance(result, dict):
        return {
            "candidates": result if isinstance(result, list) else [],
            "timed_out": False,
            "timeout_stage": "",
            "title": "",
            "elapsed_ms": 0,
            "candidate_budget": 0,
        }
    candidates = result.get("candidates")
    return {
        "candidates": candidates if isinstance(candidates, list) else [],
        "timed_out": bool(result.get("timed_out")),
        "timeout_stage": str(result.get("timeout_stage") or ""),
        "title": str(result.get("title") or ""),
        "elapsed_ms": int(result.get("elapsed_ms") or 0),
        "candidate_budget": int(result.get("candidate_budget") or 0),
    }


def collect_candidates(
    page,
    *,
    limit: int = 120,
    ignore_selectors: list[str] | tuple[str, ...] | None = None,
    include_open_shadow_dom: bool = True,
) -> list[dict[str, Any]]:
    diagnostic = collect_candidates_diagnostic(
        page,
        limit=limit,
        ignore_selectors=ignore_selectors,
        include_open_shadow_dom=include_open_shadow_dom,
    )
    return list(diagnostic.get("candidates") or [])


def semantic_selectors(
    page,
    target: str,
    action: str,
    *,
    limit: int = 120,
    ignore_selectors: list[str] | tuple[str, ...] | None = None,
    include_open_shadow_dom: bool = True,
) -> list[str]:
    """Rank visible DOM candidates by target semantics before asking the model."""
    semantic_target = _semantic_target_text(target)
    try:
        candidates = collect_candidates(
            page,
            limit=min(max(limit * 3, limit), 500),
            ignore_selectors=ignore_selectors,
            include_open_shadow_dom=include_open_shadow_dom,
        )
    except Exception:
        return []

    scored: list[tuple[float, float, int, str]] = []
    min_score = _minimum_semantic_score(semantic_target, action)
    for index, candidate in enumerate(candidates):
        selector = str(candidate.get("selector") or "").strip()
        if not selector:
            continue
        score = _candidate_semantic_score(candidate, semantic_target, action)
        if score >= min_score:
            scored.append((score, _selector_quality_score(selector), -index, selector))

    deduped: list[str] = []
    for _, _, _, selector in sorted(
        scored,
        key=lambda item: (item[0], item[1], item[2]),
        reverse=True,
    ):
        if selector not in deduped:
            deduped.append(selector)
    return deduped


def selector_matches_target(
    page,
    selector: str,
    target: str | None,
    action: str,
    *,
    strict_text_match: bool = True,
) -> bool:
    """Reject high-risk semantic mismatches before self-healing is trusted."""
    if not target:
        return True
    semantic_target = _semantic_target_text(target)
    target_l = semantic_target.lower()
    is_password_target = _contains_any(target_l, ["密码", "password", "passwd", "pwd"])
    is_username_target = _contains_any(
        target_l,
        ["用户名", "账号", "帐号", "用户", "username", "user name", "user-name"],
    )
    is_login_target = _contains_any(target_l, ["登录", "登陆", "login", "sign in"])
    is_logout_target = _is_logout_target_text(target_l)
    is_title_target = _contains_any(target_l, ["标题", "title"])
    requires_text_match = (
        strict_text_match
        and _requires_concrete_text_match(target, semantic_target, action)
        and not is_login_target
        and not is_logout_target
    )
    requires_relaxed_text_match = (
        not strict_text_match
        and _requires_concrete_text_match(target, semantic_target, action)
        and not is_login_target
        and not is_logout_target
    )
    requires_fill_match = strict_text_match and _requires_fill_semantic_match(
        target, semantic_target, action
    )
    requires_relaxed_fill_match = (
        not strict_text_match
        and _requires_fill_semantic_match(target, semantic_target, action)
    )
    requires_destructive_guard = action in {"click", "press", "press_key"}
    if not any(
        (
            is_password_target,
            is_username_target,
            is_login_target,
            is_logout_target,
            is_title_target,
            requires_text_match,
            requires_relaxed_text_match,
            requires_fill_match,
            requires_relaxed_fill_match,
            requires_destructive_guard,
        )
    ):
        return True

    try:
        snapshot = page.locator(selector).first.evaluate(
            """el => {
                const attrEscape = value => String(value).replace(/["\\\\]/g, '\\\\$&');
                const textOf = node => (node && (node.innerText || node.textContent) || '').trim().replace(/\\s+/g, ' ');
                const labelText = node => {
                    const labels = [];
                    if (node.id) {
                        const explicit = document.querySelector(`label[for="${attrEscape(node.id)}"]`);
                        if (explicit) labels.push(textOf(explicit));
                    }
                    let parent = node.parentElement;
                    while (parent && parent !== document.body) {
                        if (parent.tagName && parent.tagName.toLowerCase() === 'label') {
                            labels.push(textOf(parent));
                            break;
                        }
                        parent = parent.parentElement;
                    }
                    const formItem = node.closest('.ant-form-item,[class*="form-item" i],[class*="formItem"],fieldset');
                    if (formItem) {
                        const labelNode = formItem.querySelector('label,.ant-form-item-label,[class*="label" i]');
                        if (labelNode) labels.push(textOf(labelNode));
                    }
                    const rect = node.getBoundingClientRect();
                    const nearby = Array.from(document.querySelectorAll('label, span, div'))
                        .map(item => {
                            const text = textOf(item);
                            if (!text || text.length > 40) return null;
                            const box = item.getBoundingClientRect();
                            if (!(box.width || box.height || item.getClientRects().length)) return null;
                            const overlap = Math.min(rect.bottom, box.bottom) - Math.max(rect.top, box.top);
                            const distance = rect.left - box.right;
                            if (overlap <= 0 || distance < -8 || distance > 260) return null;
                            return { text, distance };
                        })
                        .filter(Boolean)
                        .sort((a, b) => a.distance - b.distance)[0];
                    if (nearby) labels.push(nearby.text);
                    return labels.join(' ').trim().slice(0, 180);
                };
                const ancestor = el.closest('[data-test],[data-testid],[data-qa],li,tr,form,section,article,fieldset,[role="row"],[role="listitem"],[class*="card" i],[class*="item" i],[class*="row" i]');
                return {
                    selector: '',
                    tag: el.tagName.toLowerCase(),
                    id: el.id || null,
                    class_name: el.className || null,
                    data_test: el.getAttribute('data-test'),
                    data_testid: el.getAttribute('data-testid'),
                    text: textOf(el).slice(0, 180),
                    value: el.getAttribute('value'),
                    role: el.getAttribute('role'),
                    aria_label: el.getAttribute('aria-label'),
                    placeholder: el.getAttribute('placeholder'),
                    title: el.getAttribute('title'),
                    name: el.getAttribute('name'),
                    type: el.getAttribute('type'),
                    label: labelText(el),
                    ancestor_text: ancestor && ancestor !== el ? textOf(ancestor).slice(0, 300) : ''
                };
            }"""
        )
    except Exception:
        return True

    if not isinstance(snapshot, dict):
        snapshot = {"selector": selector}
    snapshot["selector"] = selector
    score = _candidate_semantic_score(snapshot, semantic_target, action)
    tag = str(snapshot.get("tag") or "").lower()
    input_type = str(snapshot.get("type") or "").lower()
    selector_l = str(selector or "").lower()
    blob = _candidate_blob(snapshot)
    if _is_logout_or_destructive_mismatch(blob, target_l, action):
        return False
    if is_logout_target and action in {
        "click",
        "press",
        "press_key",
        "assert_visible",
    }:
        return _candidate_is_logout_like(snapshot, blob, selector_l, score, action)
    if is_password_target:
        return score >= 6
    if is_username_target:
        return score >= 4
    if requires_text_match and not _candidate_matches_semantic_text(
        snapshot, semantic_target, include_technical=False
    ):
        return False
    if requires_relaxed_text_match and not candidate_matches_semantic_terms(
        snapshot, semantic_target, include_technical=False
    ):
        return False
    if requires_fill_match and not _candidate_matches_semantic_text(
        snapshot, semantic_target, include_technical=True
    ):
        return False
    if requires_relaxed_fill_match and not candidate_matches_semantic_terms(
        snapshot, semantic_target, include_technical=True
    ):
        return False
    if is_login_target and action in {"click", "assert_visible"}:
        login_like = score >= 4 or _contains_any(
            blob,
            ["login", "sign in", "登录", "登陆"],
        )
        if "login-button" in selector_l:
            return True
        if tag == "input" and input_type in {"submit", "button"}:
            return login_like
        if tag in {"button", "a"}:
            return login_like
        return False
    if is_title_target:
        return score >= 1.5
    return True


def heuristic_selectors(target: str, action: str) -> list[str]:
    semantic_target = _semantic_target_text(target)
    target_l = semantic_target.lower()
    semantic_targets = semantic_text_variants(semantic_target) or [semantic_target]
    selectors: list[str] = []
    if (
        "密码" in semantic_target
        or "password" in target_l
        or "passwd" in target_l
        or "pwd" in target_l
    ):
        selectors.extend(
            [
                'input[type="password"]',
                'input[name*="password" i]',
                'input[id*="password" i]',
                'input[placeholder*="Password" i]',
                'input[autocomplete*="current-password" i]',
            ]
        )
    if (
        "用户名" in semantic_target
        or "账号" in semantic_target
        or "帐号" in semantic_target
        or "username" in target_l
        or "user name" in target_l
        or "user-name" in target_l
    ):
        selectors.extend(
            [
                'input[name*="user" i]',
                'input[id*="user" i]',
                'input[placeholder*="User" i]',
                'input[autocomplete*="username" i]',
                'input[name*="account" i]',
                'input[id*="account" i]',
            ]
        )
    if (
        "登录" in semantic_target
        or "登陆" in semantic_target
        or "login" in target_l
        or "sign in" in target_l
    ):
        selectors.extend(
            [
                'input[type="submit"]',
                'button[data-test*="login" i]',
                'input[data-test*="login" i]',
                'button:has-text("Login")',
            ]
        )
    if "error" in target_l or "错误" in semantic_target or "提示" in semantic_target:
        selectors.extend(
            [
                '[role="alert"]',
                '[class*="error" i]',
                '[data-test*="error" i]',
                '[data-testid*="error" i]',
            ]
        )
    attribute_selectors: list[str] = []
    for target_variant in semantic_targets:
        attribute_selectors.extend(_target_attribute_selectors(target_variant, action))
    if action == "fill":
        selectors.extend(attribute_selectors)
    if any(term in target_l for term in _QUERY_TARGET_TERMS):
        selectors.extend(
            [
                'textarea[aria-label*="Search"]',
                'input[aria-label*="Search"]',
                'textarea[placeholder*="搜索"]',
                'textarea[placeholder*="查询"]',
                'input[placeholder*="搜索"]',
                'input[placeholder*="查询"]',
                'input[type="search"]',
            ]
        )
        if action in {"click", "press", "press_key"}:
            selectors.extend(
                [
                    'button:has-text("查询")',
                    'button:has-text("搜索")',
                    'a:has-text("查询")',
                    'a:has-text("搜索")',
                ]
            )
    if _is_log_target_text(target_l):
        selectors.extend(
            [
                'a:has-text("日志")',
                'button:has-text("日志")',
                'a:has-text("查看日志")',
                'button:has-text("查看日志")',
                'a:has-text("Logs")',
                'button:has-text("Logs")',
                'a:has-text("Log")',
                'button:has-text("Log")',
            ]
        )
    if action == "fill":
        for target_variant in semantic_targets:
            css_variant = _css_attr(target_variant)
            selectors.extend(
                [
                    f'input[placeholder*="{css_variant}"]',
                    f'textarea[placeholder*="{css_variant}"]',
                    f'input[aria-label*="{css_variant}"]',
                ]
            )
    if action in {"click", "press", "press_key"}:
        for target_variant in semantic_targets:
            css_variant = _css_attr(target_variant)
            display_variant = _remove_cjk_display_spaces(target_variant)
            display_css_variant = _css_attr(display_variant)
            selectors.extend(_text_context_selectors(target_variant, action))
            selectors.extend(
                [
                    f'button:has-text("{css_variant}")',
                    f'a:has-text("{css_variant}")',
                    f'[aria-label*="{css_variant}"]',
                    f'[title*="{css_variant}"]',
                ]
            )
            if display_variant and display_variant != target_variant:
                selectors.extend(
                    [
                        f'button:has-text("{display_css_variant}")',
                        f'a:has-text("{display_css_variant}")',
                    ]
                )
        selectors.extend(attribute_selectors)
    if action.startswith("assert"):
        for target_variant in semantic_targets:
            css_variant = _css_attr(target_variant)
            selectors.extend(_text_context_selectors(target_variant, action))
            selectors.extend(
                [
                    f'*:has-text("{css_variant}")',
                    f'[aria-label*="{css_variant}"]',
                    f'[title*="{css_variant}"]',
                ]
            )
        selectors.extend(attribute_selectors)
    if action not in {"fill", "click", "press", "press_key"} and not action.startswith(
        "assert"
    ):
        selectors.extend(attribute_selectors)

    deduped: list[str] = []
    for selector in selectors:
        if selector and selector not in deduped:
            deduped.append(selector)
    return deduped


def _semantic_target_text(target: str | None) -> str:
    text = str(target or "").strip()
    if not text:
        return ""

    extracted = _extract_selector_text(text)
    lowered = text.lower()
    for prefix in ("text=", "label=", "placeholder="):
        if lowered.startswith(prefix):
            text = text.split("=", 1)[1].strip()
            break
    else:
        if extracted and looks_like_raw_selector(text):
            text = extracted

    text = text.strip().strip(" \"'[]()（）:：,，.。;；")
    return text or str(target or "").strip()


def _extract_selector_text(value: str) -> str:
    text = str(value or "").strip()
    patterns = (
        r":has-text\(\s*([\"'])(.*?)\1\s*\)",
        r"normalize-space\(\)\s*=\s*([\"'])(.*?)\1",
        r"contains\(\s*normalize-space\(\)\s*,\s*([\"'])(.*?)\1\s*\)",
        r"\[(?:aria-label|title|placeholder)[^\]]*=\s*([\"'])(.*?)\1\]",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(2).strip()
    return ""


def _requires_concrete_text_match(
    raw_target: str | None,
    semantic_target: str,
    action: str,
) -> bool:
    if action not in {"click", "press", "press_key", "assert_visible"}:
        return False
    raw = str(raw_target or "").strip()
    if not semantic_target:
        return False
    if raw.lower().startswith(("text=", "label=")):
        return True
    if _extract_selector_text(raw):
        return True
    return bool(
        re.search(r"[\u4e00-\u9fff]", semantic_target)
        and len(semantic_target) >= 2
        and not looks_like_raw_selector(raw)
    )


def _requires_fill_semantic_match(
    raw_target: str | None,
    semantic_target: str,
    action: str,
) -> bool:
    if action != "fill" or not semantic_target:
        return False
    raw = str(raw_target or "").strip()
    lowered = raw.lower()
    if lowered.startswith(("text=", "label=", "placeholder=")):
        return True
    if _extract_selector_text(raw):
        return True
    if looks_like_raw_selector(raw):
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", semantic_target))


def _candidate_matches_semantic_text(
    candidate: dict[str, Any],
    semantic_target: str,
    *,
    include_technical: bool = True,
) -> bool:
    target_l = str(semantic_target or "").lower().strip()
    if not target_l:
        return True
    blob = (
        _candidate_blob(candidate)
        if include_technical
        else _candidate_user_facing_blob(candidate)
    )
    if target_l in blob:
        return True
    normalized_target = _remove_cjk_display_spaces(target_l)
    normalized_blob = _remove_cjk_display_spaces(blob)
    if normalized_target and normalized_target in normalized_blob:
        return True
    compact_target = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", normalized_target)
    compact_blob = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", normalized_blob)
    if compact_target and compact_target in compact_blob:
        return True
    for variant in semantic_text_variants(semantic_target):
        normalized_variant = _remove_cjk_display_spaces(variant.lower())
        if normalized_variant and normalized_variant in normalized_blob:
            return True
    terms = semantic_match_terms(semantic_target)
    if terms and any(term in normalized_blob for term in terms):
        return True
    tokens = _target_tokens(semantic_target)
    if not tokens:
        return False
    return all(token.lower() in normalized_blob for token in tokens)


def candidate_matches_semantic_terms(
    candidate: dict[str, Any],
    target: str | None,
    *,
    include_technical: bool = True,
) -> bool:
    """Return whether a candidate contains at least one precise target term."""
    terms = semantic_match_terms(target)
    if not terms:
        return True
    blob = (
        _candidate_blob(candidate)
        if include_technical
        else _candidate_user_facing_blob(candidate)
    )
    normalized_blob = _remove_cjk_display_spaces(blob)
    return any(term in normalized_blob for term in terms)


def semantic_match_terms(target: str | None) -> list[str]:
    """Extract stable semantic terms for AI result validation and scan expansion."""
    terms: list[str] = []
    semantic_target = _semantic_target_text(target)
    semantic_target_l = semantic_target.lower()
    for variant in semantic_text_variants(semantic_target):
        text = _remove_cjk_display_spaces(variant.lower())
        if not text:
            continue
        chunks = re.split(r"[\\/,，,;；:：|()\[\]\s]+|(?:或者|以及|和|或|及)", text)
        for chunk in chunks:
            normalized = _strip_generic_target_words(chunk).lower()
            normalized = _remove_cjk_display_spaces(normalized).strip()
            if not normalized:
                continue
            if normalized in {
                "text",
                "button",
                "link",
                "input",
                "field",
                "按钮",
                "链接",
                "输入框",
                "字段",
            }:
                continue
            if len(normalized) < 2:
                continue
            if normalized not in terms:
                terms.append(normalized)
            for word in re.findall(r"[a-z0-9]{3,}", normalized):
                if word in {"code", "field", "input", "name", "text", "type"}:
                    continue
                if word not in terms:
                    terms.append(word)
    if _contains_any(semantic_target_l, list(_QUERY_TARGET_TERMS)):
        _extend_unique_terms(terms, _QUERY_TARGET_TERMS)
    if _contains_any(semantic_target_l, list(_PRODUCT_ID_TARGET_TERMS)):
        _extend_unique_terms(terms, _PRODUCT_ID_TARGET_TERMS)
    if _is_log_target_text(semantic_target_l):
        _extend_unique_terms(terms, _LOG_TARGET_TERMS)
    return terms


def _extend_unique_terms(terms: list[str], values: tuple[str, ...]) -> None:
    for value in values:
        normalized = _remove_cjk_display_spaces(str(value or "").lower()).strip()
        if normalized and normalized not in terms:
            terms.append(normalized)


def _is_log_target_text(value: str) -> bool:
    text = str(value or "").lower()
    return bool(
        "日志" in text
        or "记录" in text
        or "view log" in text
        or re.search(r"(?<![a-z0-9])logs?(?![a-z0-9])", text)
    )


def _text_context_selectors(target: str, action: str) -> list[str]:
    text = str(target or "").strip()
    if not text:
        return []
    css_text = _css_attr(text)
    xpath_text = _xpath_literal(text)
    exact_text = f"normalize-space()={xpath_text}"
    has_exact_text = f".//*[{exact_text}]"
    display_exact_text = exact_text
    display_has_exact_text = has_exact_text
    if re.search(r"[\u4e00-\u9fff]", text):
        display_text = _xpath_literal(_remove_cjk_display_spaces(text))
        display_exact_text = f"translate(normalize-space(), ' ', '')={display_text}"
        display_has_exact_text = f".//*[{display_exact_text}]"
    selectors = [
        f'[role="menuitem"]:has-text("{css_text}")',
        f'li[class*="menu" i]:has-text("{css_text}")',
        f'span[class*="ant-pro-menu-item-title" i]:has-text("{css_text}")',
        f'//span[contains(@class, "ant-pro-menu-item-title") and {exact_text}]',
        f'//span[contains(@class, "ant-pro-menu-item-title") and {display_exact_text}]',
        f'//*[@role="menuitem" and {has_exact_text}]',
        f'//*[@role="menuitem" and {display_has_exact_text}]',
        f'//li[contains(@class, "menu") and {has_exact_text}]',
        f'//li[contains(@class, "menu") and {display_has_exact_text}]',
        f"//li[{has_exact_text}]//*[self::span or self::a or self::button][{exact_text}]",
        f"//li[{display_has_exact_text}]//*[self::span or self::a or self::button][{display_exact_text}]",
        f'//*[contains(@class, "menu") and {has_exact_text}]//*[self::span or self::a or self::button][{exact_text}]',
        f'//*[contains(@class, "menu") and {display_has_exact_text}]//*[self::span or self::a or self::button][{display_exact_text}]',
    ]
    if action in {"click", "press", "press_key"}:
        selectors.extend(
            [
                f'button:has-text("{css_text}")',
                f'a:has-text("{css_text}")',
                f"//button[{exact_text} or .//*[{exact_text}]]",
                f"//button[{display_exact_text} or .//*[{display_exact_text}]]",
                f"//a[{exact_text} or .//*[{exact_text}]]",
                f"//a[{display_exact_text} or .//*[{display_exact_text}]]",
                f"(//span[{exact_text}])[1]",
                f"(//span[{display_exact_text}])[1]",
            ]
        )
        if re.search(r"[\u4e00-\u9fff]", text):
            display_css_text = _css_attr(_remove_cjk_display_spaces(text))
            selectors.extend(
                [
                    f'button:has-text("{display_css_text}")',
                    f'a:has-text("{display_css_text}")',
                ]
            )
    else:
        selectors.extend(
            [
                f"//button[{exact_text} or .//*[{exact_text}]]",
                f"//button[{display_exact_text} or .//*[{display_exact_text}]]",
                f"//a[{exact_text} or .//*[{exact_text}]]",
                f"//a[{display_exact_text} or .//*[{display_exact_text}]]",
                f"//*[self::span or self::div or self::p or self::td][{exact_text}]",
                f"//*[self::span or self::div or self::p or self::td][{display_exact_text}]",
            ]
        )
    return selectors


def _target_attribute_selectors(target: str, action: str) -> list[str]:
    tokens = _target_tokens(target)
    selectors: list[str] = []
    attrs = (
        "data-testid",
        "data-test",
        "data-qa",
        "data-cy",
        "data-ui",
        "id",
        "name",
        "aria-label",
        "title",
    )
    tags = (
        ("input", "textarea")
        if action == "fill"
        else ("button", "a", "input", "textarea", "select")
    )
    for token in tokens[:5]:
        quoted = _css_attr(token)
        for attr in attrs:
            selectors.append(f'[{attr}*="{quoted}" i]')
            for tag in tags:
                selectors.append(f'{tag}[{attr}*="{quoted}" i]')
    if len(tokens) >= 2:
        primary = [_css_attr(token) for token in tokens[:3]]
        for attr in ("data-testid", "data-test", "data-qa", "data-cy", "id", "name"):
            selectors.append("".join(f'[{attr}*="{token}" i]' for token in primary))
    return selectors


def _target_tokens(target: str) -> list[str]:
    stop_words = {
        "button",
        "link",
        "input",
        "field",
        "text",
        "click",
        "fill",
        "press",
        "assert",
        "visible",
        "page",
        "header",
        "footer",
        "按钮",
        "链接",
        "输入框",
        "点击",
        "输入",
        "断言",
        "可见",
        "页面",
    }
    tokens: list[str] = []
    for raw_value in semantic_text_variants(target) or [_semantic_target_text(target)]:
        raw = raw_value.replace("_", " ").replace("-", " ")
        chunks = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", raw)
        for chunk in chunks:
            normalized = _strip_generic_target_words(chunk).lower()
            if not normalized or normalized in stop_words:
                continue
            tokens.append(normalized)
            tokens.extend(
                token.lower()
                for token in re.findall(
                    r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{3,}", normalized
                )
            )
    result: list[str] = []
    for token in tokens:
        normalized = token.lower()
        if normalized in stop_words:
            continue
        if _is_ambiguous_attribute_token(normalized, tokens):
            continue
        if normalized not in result:
            result.append(normalized)
    return result


def _remove_cjk_display_spaces(value: str) -> str:
    text = str(value or "")
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", text)
    return text


def _is_logout_or_destructive_mismatch(blob: str, target_l: str, action: str) -> bool:
    if action not in {"click", "press", "press_key"}:
        return False
    target_logout = _contains_any(
        target_l, ["退出", "登出", "注销", "logout", "sign out"]
    )
    if target_logout:
        return False
    return _contains_any(
        blob,
        [
            "退出登录",
            "登出",
            "注销",
            "logout",
            "sign out",
        ],
    )


def _strip_generic_target_words(value: str) -> str:
    return _shared_strip_generic_target_words(value)


def _is_ambiguous_attribute_token(token: str, all_tokens: list[str]) -> bool:
    if token not in {"id", "no", "num", "name", "text"}:
        return False
    return any(other != token and len(other) >= 3 for other in all_tokens)


def _css_attr(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _xpath_literal(value: str) -> str:
    text = str(value)
    if "'" not in text:
        return f"'{text}'"
    if '"' not in text:
        return f'"{text}"'
    parts = text.split("'")
    return "concat(" + ', "\'", '.join(f"'{part}'" for part in parts) + ")"


def _has_cjk_display_space(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]\s+[\u4e00-\u9fff]", str(value or "")))


def _selector_quality_score(selector: str) -> float:
    selector_l = str(selector or "").lower().strip()
    if not selector_l:
        return -100.0

    score = 0.0
    if any(attr in selector_l for attr in ("[data-testid=", "[data-test=")):
        score += 45
    elif any(attr in selector_l for attr in ("[data-qa=", "[data-cy=", "[data-ui=")):
        score += 42
    elif selector_l.startswith("#"):
        score += 36
        if _looks_like_generated_selector_id(selector_l[1:]):
            score -= 18
    elif "[name=" in selector_l:
        score += 30
    elif "[aria-label=" in selector_l:
        score += 28
    elif "[placeholder=" in selector_l:
        score += 24
    elif "[title=" in selector_l:
        score += 20
    elif ":has-text(" in selector_l:
        score += 18
    elif "[role=" in selector_l:
        score += 16

    if selector_l.startswith(("//", "(//")):
        score += 18
        if "normalize-space()" in selector_l:
            score += 8
        if "contains(@class" in selector_l or "@role=" in selector_l:
            score += 8
    if any(
        marker in selector_l
        for marker in (
            "ant-pro-menu-item-title",
            "ant-menu-item",
            "menuitem",
            'role="menuitem"',
            "role='menuitem'",
        )
    ):
        score += 10
    if selector_l.startswith("text="):
        score -= 28
    if selector_l in {"input", "textarea", "button", "a", "select"}:
        score -= 30
    score -= selector_l.count(":nth-of-type") * 8
    score -= selector_l.count(" > ") * 2
    score -= min(len(selector_l) / 40, 12)
    return score


def _looks_like_structural_selector(selector: str) -> bool:
    text = str(selector or "").strip().lower()
    if not text:
        return True
    if any(marker in text for marker in ("#", "[", ":has-text(", "text=", "//", "(//")):
        return False
    return bool(
        re.fullmatch(
            r"(?:[a-z]+(?::nth-of-type\(\d+\))?\s*>\s*)+"
            r"[a-z]+(?::nth-of-type\(\d+\))?",
            text,
        )
    )


def _looks_like_generated_selector_id(value: str) -> bool:
    value = str(value or "")
    return bool(
        re.fullmatch(r"\d+", value)
        or re.fullmatch(r"[a-f0-9]{8,}", value, flags=re.IGNORECASE)
        or re.search(r"\d{8,}", value)
        or re.fullmatch(r"ember\d+", value, flags=re.IGNORECASE)
    )


def _candidate_semantic_score(
    candidate: dict[str, Any], target: str, action: str
) -> float:
    target_l = str(target or "").lower()
    blob = _candidate_blob(candidate)
    selector = str(candidate.get("selector") or "").lower()
    tag = str(candidate.get("tag") or "").lower()
    role = str(candidate.get("role") or "").lower()
    input_type = str(candidate.get("type") or "").lower()
    score = 0.0

    if action in {"fill", "clear"}:
        if tag in {"input", "textarea", "select"} or role in {
            "textbox",
            "combobox",
            "spinbutton",
        }:
            score += 2
        else:
            score -= 4
    elif action in {"click", "press", "press_key"}:
        if tag in {"button", "a"} or role in {"button", "link"}:
            score += 1

    if target_l and target_l in blob:
        score += 5

    blob_tokens = set(re.findall(r"[a-z0-9]+", blob))
    generic_terms = {
        "button",
        "link",
        "input",
        "field",
        "text",
        "select",
        "combobox",
        "textbox",
        "click",
        "clear",
        "view",
    }
    for term in semantic_match_terms(target_l):
        term_l = str(term or "").lower()
        if not term_l:
            continue
        if re.fullmatch(r"[a-z0-9]+", term_l):
            if len(term_l) < 3 or term_l in generic_terms:
                continue
            if term_l in blob_tokens:
                score += 4
            continue
        if term_l in blob:
            score += 4

    for word in re.findall(r"[a-z0-9]{3,}", target_l):
        if word in blob:
            score += 1.5

    if _contains_any(target_l, ["密码", "password", "passwd", "pwd"]):
        if input_type == "password":
            score += 8
        if _contains_any(blob, ["password", "passwd", "pwd", "密码"]):
            score += 6
        if _contains_any(blob, ["user-name", "username", "user name"]):
            score -= 8

    if _contains_any(
        target_l,
        ["用户名", "账号", "帐号", "用户", "username", "user name", "user-name"],
    ):
        if input_type == "password":
            score -= 8
        if _contains_any(
            blob, ["user-name", "username", "user name", "login", "用户", "账号"]
        ):
            score += 6
        if _contains_any(blob, ["password", "passwd", "pwd"]):
            score -= 6

    if _contains_any(target_l, ["登录", "登陆", "login", "sign in"]):
        if _contains_any(blob, ["login", "sign in", "登录", "登陆"]):
            score += 5

    if _is_logout_target_text(target_l):
        if _contains_any(blob, list(_LOGOUT_CANDIDATE_TERMS)):
            score += 5

    if _contains_any(target_l, ["错误", "提示", "error"]):
        if _contains_any(blob, ["error", "sadface", "错误"]):
            score += 5

    if _contains_any(target_l, ["添加", "加入", "add"]):
        if _contains_any(blob, ["add", "添加", "加入"]):
            score += 4

    if selector and looks_like_raw_selector(selector) and selector in blob:
        score += 0.25
    return score


def _candidate_blob(candidate: dict[str, Any]) -> str:
    keys = (
        "selector",
        "id",
        "class_name",
        "data_test",
        "data_testid",
        "text",
        "value",
        "role",
        "aria_label",
        "placeholder",
        "title",
        "name",
        "type",
        "label",
        "ancestor_text",
    )
    return " ".join(str(candidate.get(key) or "").lower() for key in keys).replace(
        "_", "-"
    )


def _candidate_user_facing_blob(candidate: dict[str, Any]) -> str:
    keys = (
        "text",
        "aria_label",
        "placeholder",
        "title",
        "label",
        "ancestor_text",
        "value",
    )
    return " ".join(str(candidate.get(key) or "").lower() for key in keys).replace(
        "_", "-"
    )


def _contains_any(value: str, terms: list[str]) -> bool:
    return any(term.lower() in value for term in terms)


def _is_logout_target_text(value: str) -> bool:
    return _contains_any(value, list(_LOGOUT_TARGET_TERMS))


def _candidate_is_logout_like(
    candidate: dict[str, Any],
    blob: str,
    selector_l: str,
    score: float,
    action: str,
) -> bool:
    logout_like = score >= 4 or _contains_any(
        f"{blob} {selector_l}",
        list(_LOGOUT_CANDIDATE_TERMS),
    )
    if not logout_like:
        return False
    if action == "assert_visible":
        return True
    tag = str(candidate.get("tag") or "").lower()
    role = str(candidate.get("role") or "").lower()
    input_type = str(candidate.get("type") or "").lower()
    return (
        tag in {"button", "a"}
        or role in {"button", "link", "menuitem"}
        or (tag == "input" and input_type in {"submit", "button"})
    )


def _minimum_semantic_score(target: str, action: str) -> float:
    target_l = str(target or "").lower()
    if _contains_any(
        target_l, ["密码", "password", "用户名", "账号", "帐号", "username"]
    ):
        return 4
    if action in {"fill", "clear"}:
        return 3
    return 4


def looks_like_raw_selector(value: str) -> bool:
    value = str(value).strip()
    if value.startswith(
        (
            "#",
            ".",
            "//",
            "(//",
            "text=",
            "[",
            "css=",
            "xpath=",
            "role=",
            "label=",
            "placeholder=",
        )
    ):
        return True
    html_tags = {
        "a",
        "button",
        "div",
        "form",
        "img",
        "input",
        "label",
        "li",
        "option",
        "select",
        "span",
        "textarea",
        "ul",
    }
    if value.lower() in html_tags:
        return True
    match = re.match(r"^([a-zA-Z][a-zA-Z0-9_-]*)(\[|\.|#|:|\s+|>)", value)
    if not match:
        return False
    first_token = match.group(1).lower()
    marker = match.group(2)
    if first_token in html_tags:
        return True
    if "-" in first_token:
        return True
    return bool(marker and not marker.isspace())
