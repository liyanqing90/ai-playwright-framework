from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

from ai_playwright.ai_generation.project_context import ProjectContext


_URL_RE = re.compile(r"https?://[^\s\"'<>),;]+", re.IGNORECASE)
_NAVIGATION_ACTIONS = {"goto", "open", "navigate"}


def normalize_entry_url(url: str, *, base_url: str = "") -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""

    resolved = raw
    if base_url and not urlsplit(raw).scheme:
        resolved = urljoin(str(base_url or "").strip(), raw)

    parsed = urlsplit(resolved)
    if not parsed.scheme or not parsed.netloc:
        return raw.rstrip("/")

    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    if port and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname

    path = re.sub(r"/+", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((scheme, netloc, path or "/", "", ""))


def resolve_entry_scope(
    *,
    spec: dict[str, Any],
    modules: dict[str, Any],
    base_url: str,
    spec_source_name: str = "spec",
    priority: list[str] | None = None,
) -> dict[str, Any]:
    spec_urls = extract_spec_urls(spec)
    module_entries = module_goto_entries(modules, base_url=base_url)
    project_base_url = str(base_url or "").strip()

    resolved: dict[str, Any] | None = None
    if spec_urls:
        url = spec_urls[0]
        resolved = {
            "source": spec_source_name,
            "value": url,
            "url": url,
            "normalized_url": normalize_entry_url(url, base_url=project_base_url),
        }
    elif module_entries:
        resolved = {"source": "module", **dict(module_entries[0])}
    elif project_base_url:
        resolved = {
            "source": "project_config",
            "value": project_base_url,
            "url": project_base_url,
            "normalized_url": normalize_entry_url(project_base_url),
        }

    normalized_entry_url = ""
    if resolved:
        normalized_entry_url = str(
            resolved.get("normalized_url")
            or normalize_entry_url(
                str(resolved.get("url") or resolved.get("value") or ""),
                base_url=project_base_url,
            )
        )

    return {
        "priority": priority
        or [
            "spec.steps_or_description_url",
            "project_context.module_goto",
            "project_context.base_url",
        ],
        "resolved": resolved,
        "normalized_entry_url": normalized_entry_url,
        "spec_urls": spec_urls,
        "module_goto_entries": module_entries[:20],
        "project_base_url": project_base_url,
    }


def extract_spec_urls(spec: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    cases = spec.get("cases")
    if isinstance(cases, list):
        for case in cases:
            if isinstance(case, dict):
                urls.extend(extract_urls(case.get("steps")))
                urls.extend(extract_urls(case.get("intent")))
    urls.extend(extract_urls(spec.get("steps")))
    urls.extend(extract_urls(spec.get("intent")))
    if isinstance(cases, list):
        for case in cases:
            if isinstance(case, dict):
                urls.extend(extract_urls(case.get("description")))
            elif isinstance(case, str):
                urls.extend(extract_urls(case))
    urls.extend(extract_urls(spec.get("description")))
    return dedupe_preserve_order(urls)


def module_goto_entries(
    modules: dict[str, Any], *, base_url: str = ""
) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for module_name, raw_steps in (modules or {}).items():
        steps = raw_steps.get("steps") if isinstance(raw_steps, dict) else raw_steps
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            action = str(step.get("action") or "").lower()
            if action not in _NAVIGATION_ACTIONS:
                continue
            value = step.get("value") or step.get("url")
            if not value:
                continue

            entry = {
                "module": str(module_name),
                "action": action,
                "value": str(value),
            }
            url = first_url(value)
            if not url and isinstance(value, str) and value.strip().startswith("/"):
                url = value.strip()
            if url:
                normalized = normalize_entry_url(url, base_url=base_url)
                if normalized:
                    entry["url"] = normalized
                    entry["normalized_url"] = normalized
            entries.append(entry)
    return entries


def extract_urls(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [match.group(0).rstrip(".,;") for match in _URL_RE.finditer(value)]
    if isinstance(value, list):
        urls: list[str] = []
        for item in value:
            urls.extend(extract_urls(item))
        return urls
    if isinstance(value, dict):
        urls: list[str] = []
        for item in value.values():
            urls.extend(extract_urls(item))
        return urls
    return []


def first_url(value: Any) -> str:
    urls = extract_urls(value)
    return urls[0] if urls else ""


def context_asset_fingerprint(context: ProjectContext) -> dict[str, str]:
    elements = context.elements or {}
    modules = context.modules or {}
    variables = context.variables or {}
    return {
        "element_keys": hash_payload(sorted(elements.keys())),
        "element_values": hash_payload(elements),
        "module_keys": hash_payload(sorted(modules.keys())),
        "module_values": hash_payload(modules),
        "variable_keys": hash_payload(sorted(variables.keys())),
        "variable_values": hash_payload(variables),
    }


def dedupe_preserve_order(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def hash_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode(
            "utf-8"
        )
    ).hexdigest()
