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
                '[class*="modal" i]', '[class*="dialog" i]'
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
                        text: (el.innerText || el.textContent || '').trim().slice(0, 180),
                        role: el.getAttribute('role'),
                        aria_label: el.getAttribute('aria-label'),
                        placeholder: el.getAttribute('placeholder'),
                        title: el.getAttribute('title'),
                        name: el.getAttribute('name'),
                        type: el.getAttribute('type'),
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
