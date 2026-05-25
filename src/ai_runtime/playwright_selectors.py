from __future__ import annotations

import re
from typing import Any

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


def verify_selector(page, selector: str, *, action: str, timeout: int) -> bool:
    locator = page.locator(selector).first
    locator.wait_for(state="visible", timeout=timeout)
    if action in {"click", "press", "press_key"}:
        return bool(locator.is_enabled())
    if action == "fill":
        return bool(
            locator.evaluate(
                """el => {
                    const tag = el.tagName.toLowerCase();
                    return tag === 'input'
                        || tag === 'textarea'
                        || el.isContentEditable === true
                        || el.getAttribute('role') === 'textbox';
                }"""
            )
        )
    return True


def stable_selector_for_locator(locator) -> str:
    return locator.first.evaluate(
        """el => {
            const cssEscape = value => {
                if (window.CSS && CSS.escape) return CSS.escape(value);
                return String(value).replace(/["\\\\]/g, '\\\\$&');
            };
            const attrSelector = (node, attr) => {
                const value = node.getAttribute(attr);
                if (!value) return null;
                return `${node.tagName.toLowerCase()}[${attr}="${String(value).replace(/["\\\\]/g, '\\\\$&')}"]`;
            };
            if (el.id) return `#${cssEscape(el.id)}`;
            for (const attr of ['data-testid', 'data-test', 'data-qa', 'name', 'aria-label', 'placeholder', 'title']) {
                const selector = attrSelector(el, attr);
                if (selector) return selector;
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
                node = parent;
            }
            return parts.join(' > ');
        }"""
    )


def collect_candidates(page, *, limit: int = 120) -> list[dict[str, Any]]:
    candidates = page.evaluate(
        """limit => {
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
                const ancestor = el.closest('.inventory_item, .cart_item, li, tr, form, section, article, [data-test], [data-testid]');
                if (!ancestor || ancestor === el) return '';
                return (ancestor.innerText || ancestor.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 300);
            };
            const stableSelector = el => {
                if (el.id) return `#${cssEscape(el.id)}`;
                for (const attr of ['data-testid', 'data-test', 'data-qa', 'name', 'aria-label', 'placeholder', 'title']) {
                    const selector = attrSelector(el, attr);
                    if (selector) return selector;
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
                    node = parent;
                }
                return parts.join(' > ');
            };
            const nodes = Array.from(document.querySelectorAll([
                'input', 'textarea', 'button', 'a', 'select', '[role]',
                '[aria-label]', '[placeholder]', 'label', '[onclick]',
                '[tabindex]', '[title]', '[class*="close" i]',
                '[class*="cancel" i]', '[class*="popup" i]',
                '[class*="modal" i]', '[class*="dialog" i]',
                '[data-test]', '[data-testid]', '[data-qa]',
                '[class*="cart" i]', '[class*="badge" i]',
                '[class*="error" i]', '[class*="inventory" i]'
            ].join(',')));
            return nodes
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
                        bbox: [x1, y1, x2, y2],
                        center: [centerX, centerY],
                        bbox_norm: [x1 / viewport.width, y1 / viewport.height, x2 / viewport.width, y2 / viewport.height],
                        center_norm: [centerX / viewport.width, centerY / viewport.height],
                    };
                });
        }""",
        limit,
    )
    return candidates


def semantic_selectors(
    page, target: str, action: str, *, limit: int = 120
) -> list[str]:
    """Rank visible DOM candidates by target semantics before asking the model."""
    try:
        candidates = collect_candidates(page, limit=limit)
    except Exception:
        return []

    scored: list[tuple[float, str]] = []
    min_score = _minimum_semantic_score(target, action)
    for candidate in candidates:
        selector = str(candidate.get("selector") or "").strip()
        if not selector:
            continue
        score = _candidate_semantic_score(candidate, target, action)
        if score >= min_score:
            scored.append((score, selector))

    deduped: list[str] = []
    for _, selector in sorted(scored, key=lambda item: item[0], reverse=True):
        if selector not in deduped:
            deduped.append(selector)
    return deduped


def selector_matches_target(
    page, selector: str, target: str | None, action: str
) -> bool:
    """Reject high-risk semantic mismatches such as password target -> username input."""
    if not target:
        return True
    target_l = str(target).lower()
    is_password_target = _contains_any(target_l, ["密码", "password", "passwd", "pwd"])
    is_username_target = _contains_any(
        target_l,
        ["用户名", "账号", "帐号", "用户", "username", "user name", "user-name"],
    )
    if not is_password_target and not is_username_target:
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

    snapshot["selector"] = selector
    score = _candidate_semantic_score(snapshot, target, action)
    if is_password_target:
        return score >= 6
    if is_username_target:
        return score >= 4
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
                "#password",
                'input[name*="password" i]',
                'input[id*="password" i]',
                'input[placeholder*="Password" i]',
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
                "#user-name",
                'input[name*="user" i]',
                'input[id*="user" i]',
                'input[placeholder*="User" i]',
                'input[autocomplete*="username" i]',
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
                "#login-button",
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
                '[data-test="error"]',
                'h3[data-test="error"]',
                '[class*="error" i]',
            ]
        )
    if "购物车" in target or "cart" in target_l:
        if (
            "角标" in target
            or "数量" in target
            or "badge" in target_l
            or "count" in target_l
        ):
            selectors.extend(
                [
                    '[data-test="shopping-cart-badge"]',
                    ".shopping_cart_badge",
                    '[class*="cart_badge" i]',
                ]
            )
        if "入口" in target or "图标" in target or "link" in target_l:
            selectors.extend(
                [
                    '[data-test="shopping-cart-link"]',
                    ".shopping_cart_link",
                    '[class*="shopping_cart_link" i]',
                ]
            )
        if "列表" in target or "区域" in target or "list" in target_l:
            selectors.extend(
                [
                    '[data-test="cart-list"]',
                    ".cart_list",
                    '[class*="cart_list" i]',
                ]
            )
    if ("backpack" in target_l or "sauce labs backpack" in target_l) and (
        "添加" in target or "add" in target_l
    ):
        selectors.extend(
            [
                '[data-test="add-to-cart-sauce-labs-backpack"]',
                "#add-to-cart-sauce-labs-backpack",
            ]
        )
    if "搜索" in target or "search" in target_l:
        selectors.extend(
            [
                'textarea[name="q"]',
                'input[name="q"]',
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

    if _contains_any(target_l, ["购物车", "cart"]):
        if _contains_any(blob, ["cart", "shopping-cart", "购物车"]):
            score += 4
        if _contains_any(
            target_l, ["角标", "数量", "badge", "count"]
        ) and _contains_any(
            blob, ["badge", "count", "shopping-cart-badge", "shopping-cart-badge"]
        ):
            score += 5
        if _contains_any(target_l, ["入口", "图标", "link"]) and _contains_any(
            blob, ["shopping-cart-link", "cart link"]
        ):
            score += 5
        if _contains_any(target_l, ["列表", "区域", "list"]) and _contains_any(
            blob, ["cart-list", "cart item", "inventory"]
        ):
            score += 4

    if _contains_any(target_l, ["错误", "提示", "error"]):
        if _contains_any(blob, ["error", "sadface", "错误"]):
            score += 5

    if _contains_any(target_l, ["添加", "加入", "add"]):
        if _contains_any(blob, ["add-to-cart", "add to cart", "添加"]):
            score += 4

    if "backpack" in target_l and "backpack" in blob:
        score += 4
    if "sauce" in target_l and "sauce" in blob:
        score += 1
    if "labs" in target_l and "labs" in blob:
        score += 1

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
