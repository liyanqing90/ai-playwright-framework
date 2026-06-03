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


def normalize_selector(selector: str, selector_type: str | None = None) -> str:
    selector = str(selector).strip()
    if selector.startswith("css="):
        return selector.removeprefix("css=")
    if selector.startswith("xpath="):
        return selector.removeprefix("xpath=")
    if selector_type == "xpath" and not selector.startswith(("//", "(//")):
        return f"//{selector}"
    return selector


def validate_selector(
    page,
    selector: str,
    *,
    action: str,
    timeout: int,
    require_unique: bool = False,
) -> SelectorValidation:
    locator = page.locator(selector)
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

    first = locator.first
    try:
        first.wait_for(state="visible", timeout=timeout)
    except Exception as exc:
        return SelectorValidation(
            selector=selector,
            action=action,
            ok=False,
            match_count=match_count,
            visible_count=_safe_visible_count(locator, match_count),
            error=str(exc),
        )

    visible_count = _safe_visible_count(locator, match_count)
    enabled: bool | None = None
    action_compatible: bool | None = None
    normalized_action = str(action or "").lower()
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
            action_compatible = bool(
                first.evaluate(
                    """el => {
                        const tag = el.tagName.toLowerCase();
                        return tag === 'input'
                            || tag === 'textarea'
                            || el.isContentEditable === true
                            || el.getAttribute('role') === 'textbox';
                    }"""
                )
            )
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


def selector_probe_score(selector: str, action: str | None = None) -> float:
    selector_l = str(selector or "").lower().strip()
    if not selector_l:
        return -100.0
    score = _selector_quality_score(selector_l)
    normalized_action = str(action or "").lower()
    has_tag_prefix = bool(re.match(r"^[a-z][\w-]*\[", selector_l))

    if _looks_like_structural_selector(selector_l):
        score -= 80
    if re.search(
        r'^[a-z][\w-]*\[type\s*=\s*["\']?(?:password|search)["\']?',
        selector_l,
    ):
        score += 50
    if re.search(
        r'^[a-z][\w-]*\[type\s*=\s*["\']?(?:submit|button)["\']?',
        selector_l,
    ):
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


def collect_candidates(
    page,
    *,
    limit: int = 120,
    ignore_selectors: list[str] | tuple[str, ...] | None = None,
    include_open_shadow_dom: bool = True,
) -> list[dict[str, Any]]:
    candidates = page.evaluate(
        """opts => {
            const limit = opts.limit || 120;
            const ignoreSelectors = Array.isArray(opts.ignore_selectors) ? opts.ignore_selectors : [];
            const includeOpenShadowDom = opts.include_open_shadow_dom !== false;
            const viewport = {
                width: window.innerWidth || document.documentElement.clientWidth || 1,
                height: window.innerHeight || document.documentElement.clientHeight || 1
            };
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
            const labelText = el => {
                const labels = [];
                if (el.id) {
                    const explicit = document.querySelector(`label[for="${cssEscape(el.id)}"]`);
                    if (explicit) labels.push(explicit.innerText || explicit.textContent || '');
                }
                let parent = el.parentElement;
                while (parent && parent !== document.body) {
                    if (parent.tagName && parent.tagName.toLowerCase() === 'label') {
                        labels.push(parent.innerText || parent.textContent || '');
                        break;
                    }
                    parent = parent.parentElement;
                }
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
                return (ancestor.innerText || ancestor.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 300);
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
                seen.add(el);
                nodes.push(el);
            };
            const scanRoot = (root, inShadow=false) => {
                let found = [];
                try {
                    found = Array.from(root.querySelectorAll(candidateSelector));
                } catch {
                    found = [];
                }
                for (const el of found) {
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
                        scanRoot(host.shadowRoot, true);
                    }
                }
            };
            scanRoot(document, false);
            return nodes
                .filter(el => !ignored(el))
                .filter(el => !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length))
                .slice(0, limit)
                .map((el, index) => {
                    const rect = el.getBoundingClientRect();
                    const x1 = Math.max(0, rect.left);
                    const y1 = Math.max(0, rect.top);
                    const x2 = Math.min(viewport.width, rect.right);
                    const y2 = Math.min(viewport.height, rect.bottom);
                    const centerX = x1 + (x2 - x1) / 2;
                    const centerY = y1 + (y2 - y1) / 2;
                    return {
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
                });
        }""",
        {
            "limit": limit,
            "ignore_selectors": list(ignore_selectors or ()),
            "include_open_shadow_dom": include_open_shadow_dom,
        },
    )
    return candidates


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
            """el => ({
                selector: '',
                tag: el.tagName.toLowerCase(),
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
                label: '',
                ancestor_text: ''
            })"""
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
    if any(
        term in target_l
        for term in (
            "搜索",
            "查询",
            "检索",
            "查找",
            "筛选",
            "过滤",
            "search",
            "query",
            "find",
            "lookup",
            "filter",
        )
    ):
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
    for variant in semantic_text_variants(_semantic_target_text(target)):
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
    return terms


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

    if action == "fill":
        if tag in {"input", "textarea"} or role == "textbox":
            score += 2
        else:
            score -= 4
    elif action in {"click", "press", "press_key"}:
        if tag in {"button", "a"} or role in {"button", "link"}:
            score += 1

    if target_l and target_l in blob:
        score += 5

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
    if action == "fill":
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
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*(\[|\.|#|:|\s|>)", value))
