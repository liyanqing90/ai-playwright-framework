from __future__ import annotations

import re
from typing import Any

from src.ai_runtime.native_observe import SelectorValidation

_SENSITIVE_PATTERNS = [
    (re.compile(r"\b1[3-9]\d{9}\b"), "<phone>"),
    (re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), "<email>"),
    (re.compile(r"\b\d{17}[\dXx]\b"), "<id_card>"),
    (re.compile(r"\b[A-Za-z0-9_-]{32,}\b"), "<token>"),
]


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
            const exactTextSelector = node => {
                const tag = node.tagName.toLowerCase();
                if (!['button', 'a', 'label'].includes(tag)) return null;
                const text = visibleText(node);
                if (!text || text.length > 80) return null;
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
                const exactTextSelector = node => {
                    const tag = node.tagName.toLowerCase();
                    if (!['button', 'a', 'label'].includes(tag)) return null;
                    const text = visibleText(node);
                    if (!text || text.length > 80) return null;
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
    min_score = _minimum_semantic_score(target, action)
    for index, candidate in enumerate(candidates):
        selector = str(candidate.get("selector") or "").strip()
        if not selector:
            continue
        score = _candidate_semantic_score(candidate, target, action)
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
    page, selector: str, target: str | None, action: str
) -> bool:
    """Reject high-risk semantic mismatches before self-healing is trusted."""
    if not target:
        return True
    target_l = str(target).lower()
    is_password_target = _contains_any(target_l, ["密码", "password", "passwd", "pwd"])
    is_username_target = _contains_any(
        target_l,
        ["用户名", "账号", "帐号", "用户", "username", "user name", "user-name"],
    )
    is_login_target = _contains_any(target_l, ["登录", "登陆", "login", "sign in"])
    is_title_target = _contains_any(target_l, ["标题", "title"])
    if not any(
        (
            is_password_target,
            is_username_target,
            is_login_target,
            is_title_target,
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
    score = _candidate_semantic_score(snapshot, target, action)
    tag = str(snapshot.get("tag") or "").lower()
    input_type = str(snapshot.get("type") or "").lower()
    selector_l = str(selector or "").lower()
    blob = _candidate_blob(snapshot)
    if is_password_target:
        return score >= 6
    if is_username_target:
        return score >= 4
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


def sanitize_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for candidate in candidates:
        item = {}
        for key, value in candidate.items():
            item[key] = redact_value(value)
        sanitized.append(item)
    return sanitized


def redact_value(value: Any) -> Any:
    # Local model usage intentionally keeps raw UI text and attributes so the
    # resolver can reason over real data. Keep this function as a compatibility
    # no-op for existing imports.
    return value


def heuristic_selectors(target: str, action: str) -> list[str]:
    target_l = target.lower()
    selectors: list[str] = []
    if (
        "密码" in target
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
        "用户名" in target
        or "账号" in target
        or "帐号" in target
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
        "登录" in target
        or "登陆" in target
        or "login" in target_l
        or "sign in" in target_l
    ):
        selectors.extend(
            [
                'input[type="submit"]',
                'button[data-test*="login" i]',
                'input[data-test*="login" i]',
                'button:has-text("Login")',
                "text=Login",
            ]
        )
    if "error" in target_l or "错误" in target or "提示" in target:
        selectors.extend(
            [
                '[role="alert"]',
                '[class*="error" i]',
                '[data-test*="error" i]',
                '[data-testid*="error" i]',
            ]
        )
    selectors.extend(_target_attribute_selectors(target, action))
    if "搜索" in target or "search" in target_l:
        selectors.extend(
            [
                'textarea[aria-label*="Search"]',
                'input[aria-label*="Search"]',
                'textarea[placeholder*="搜索"]',
                'input[placeholder*="搜索"]',
                'input[type="search"]',
            ]
        )
    if action == "fill":
        selectors.extend(
            [
                f'input[placeholder*="{target}"]',
                f'textarea[placeholder*="{target}"]',
                f'input[aria-label*="{target}"]',
                "input",
                "textarea",
            ]
        )
    if action in {"click", "press", "press_key"}:
        selectors.extend(
            [
                f'button:has-text("{target}")',
                f'a:has-text("{target}")',
                f"text={target}",
                f'[aria-label*="{target}"]',
                f'[title*="{target}"]',
            ]
        )
    if action.startswith("assert"):
        selectors.extend([f"text={target}", f'[aria-label*="{target}"]'])

    deduped: list[str] = []
    for selector in selectors:
        if selector and selector not in deduped:
            deduped.append(selector)
    return deduped


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
    raw = str(target or "").replace("_", " ").replace("-", " ")
    tokens = re.findall(r"[A-Za-z0-9]{2,}|[\u4e00-\u9fff]{2,}", raw)
    result: list[str] = []
    for token in tokens:
        normalized = token.lower()
        if normalized in stop_words:
            continue
        if normalized not in result:
            result.append(normalized)
    return result


def _css_attr(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


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
    elif ":has-text(" in selector_l or selector_l.startswith("text="):
        score += 18
    elif "[role=" in selector_l:
        score += 16

    if selector_l in {"input", "textarea", "button", "a", "select"}:
        score -= 30
    score -= selector_l.count(":nth-of-type") * 8
    score -= selector_l.count(" > ") * 2
    score -= min(len(selector_l) / 40, 12)
    return score


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


def _contains_any(value: str, terms: list[str]) -> bool:
    return any(term.lower() in value for term in terms)


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
