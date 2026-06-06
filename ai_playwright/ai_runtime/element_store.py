from __future__ import annotations

import os
import atexit
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from ai_playwright.ai_runtime.playwright_selectors import (
    canonicalize_persisted_selector,
)
from ai_playwright.ai_runtime.semantic_terms import (
    semantic_text_variants,
    strip_generic_target_words,
)


_WRITE_LOCK = threading.Lock()
_PENDING_LOCK = threading.Lock()
_PENDING_THREADS: list[threading.Thread] = []


@dataclass(frozen=True)
class ElementUpdateResult:
    key: str
    new_selector: str
    updated: bool
    path: Path | None = None
    old_selector: str | None = None
    reason: str | None = None


class ElementDefinitionStore:
    """Find and update the source YAML file for a merged element key."""

    def __init__(self, test_dir: str | Path | None = None):
        self.test_dir = Path(test_dir or os.environ.get("TEST_DIR", "")).resolve()
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.indent(mapping=2, sequence=4, offset=2)

    def update_selector(
        self,
        key: str,
        new_selector: str,
        *,
        identifier: str | None = None,
        allow_semantic_generic_update: bool = False,
    ) -> ElementUpdateResult:
        key = str(key or "").strip()
        new_selector = canonicalize_persisted_selector(new_selector)
        if not key:
            return ElementUpdateResult(
                key=key,
                new_selector=new_selector,
                updated=False,
                reason="empty_key",
            )
        if not new_selector:
            return ElementUpdateResult(
                key=key,
                new_selector=new_selector,
                updated=False,
                reason="empty_selector",
            )

        located = self._locate_element_file(key)
        qualified_key = self._qualified_key(key, identifier)
        qualified_located = (
            self._locate_element_file(qualified_key) if qualified_key != key else None
        )
        if located is None:
            if qualified_located is not None:
                return self._update_located_selector(
                    qualified_key,
                    new_selector,
                    qualified_located,
                    reason="updated_existing_key",
                )
            existing = self._locate_key_with_selector(qualified_key, new_selector)
            if existing is not None:
                path, old_selector = existing
                return ElementUpdateResult(
                    key=qualified_key,
                    new_selector=new_selector,
                    updated=False,
                    path=path,
                    old_selector=old_selector,
                    reason="unchanged",
                )
            path, data = self._default_elements_file()
            new_key = self._unique_element_key(
                qualified_key,
                self._all_project_element_keys() | self._all_element_keys(data),
            )
            data["elements"][new_key] = new_selector
            with _WRITE_LOCK:
                self._atomic_dump(path, data)
            return ElementUpdateResult(
                key=new_key,
                new_selector=new_selector,
                updated=True,
                path=path,
                reason="created",
            )

        path, data, current_value, occurrence_count = located
        old_selector = self._extract_selector(current_value)
        if old_selector == new_selector:
            return ElementUpdateResult(
                key=key,
                new_selector=new_selector,
                updated=False,
                path=path,
                old_selector=old_selector,
                reason="unchanged",
            )

        if self._should_update_existing_key(
            key,
            occurrence_count,
            identifier=identifier,
            allow_semantic_generic_update=allow_semantic_generic_update,
        ):
            if not self._replace_selector(data["elements"], key, new_selector):
                return ElementUpdateResult(
                    key=key,
                    new_selector=new_selector,
                    updated=False,
                    path=path,
                    old_selector=old_selector,
                    reason="unsupported_element_shape",
                )

            with _WRITE_LOCK:
                self._atomic_dump(path, data)

            return ElementUpdateResult(
                key=key,
                new_selector=new_selector,
                updated=True,
                path=path,
                old_selector=old_selector,
                reason="updated_existing_key",
            )

        existing = self._locate_key_with_selector(qualified_key, new_selector)
        if existing is not None:
            existing_path, existing_selector = existing
            return ElementUpdateResult(
                key=qualified_key,
                new_selector=new_selector,
                updated=False,
                path=existing_path,
                old_selector=existing_selector,
                reason="unchanged",
            )

        if qualified_located is not None:
            return self._update_located_selector(
                qualified_key,
                new_selector,
                qualified_located,
                reason="updated_existing_key",
            )

        new_key = self._unique_element_key(
            qualified_key,
            self._all_project_element_keys() | self._all_element_keys(data),
        )
        data["elements"][new_key] = new_selector

        with _WRITE_LOCK:
            self._atomic_dump(path, data)

        return ElementUpdateResult(
            key=new_key,
            new_selector=new_selector,
            updated=True,
            path=path,
            old_selector=old_selector,
            reason="added_scoped_key",
        )

    def _locate_element_file(self, key: str) -> tuple[Path, Any, Any, int] | None:
        elements_dir = self.test_dir / "elements"
        if not elements_dir.exists():
            return None

        found: tuple[Path, Any, Any, int] | None = None
        occurrence_count = 0
        yaml_files = sorted(
            list(elements_dir.glob("**/*.yaml")) + list(elements_dir.glob("**/*.yml")),
            key=lambda file: file.as_posix(),
        )
        for path in yaml_files:
            with path.open("r", encoding="utf-8") as file:
                data = self.yaml.load(file) or {}
            if not isinstance(data, dict):
                continue
            elements = data.get("elements")
            if isinstance(elements, dict) and key in elements:
                occurrence_count += 1
                found = (path, data, elements[key], occurrence_count)
        if found is not None:
            path, data, value, _ = found
            return (path, data, value, occurrence_count)
        return found

    def _locate_key_with_selector(
        self,
        key: str,
        selector: str,
    ) -> tuple[Path, str] | None:
        elements_dir = self.test_dir / "elements"
        if not elements_dir.exists():
            return None
        yaml_files = sorted(
            list(elements_dir.glob("**/*.yaml")) + list(elements_dir.glob("**/*.yml")),
            key=lambda file: file.as_posix(),
        )
        for path in yaml_files:
            with path.open("r", encoding="utf-8") as file:
                data = self.yaml.load(file) or {}
            elements = data.get("elements") if isinstance(data, dict) else {}
            if not isinstance(elements, dict) or key not in elements:
                continue
            old_selector = self._extract_selector(elements[key])
            if old_selector == selector:
                return path, old_selector
        return None

    def _update_located_selector(
        self,
        key: str,
        new_selector: str,
        located: tuple[Path, Any, Any, int],
        *,
        reason: str,
    ) -> ElementUpdateResult:
        path, data, current_value, _ = located
        old_selector = self._extract_selector(current_value)
        if old_selector == new_selector:
            return ElementUpdateResult(
                key=key,
                new_selector=new_selector,
                updated=False,
                path=path,
                old_selector=old_selector,
                reason="unchanged",
            )
        if not self._replace_selector(data["elements"], key, new_selector):
            return ElementUpdateResult(
                key=key,
                new_selector=new_selector,
                updated=False,
                path=path,
                old_selector=old_selector,
                reason="unsupported_element_shape",
            )
        with _WRITE_LOCK:
            self._atomic_dump(path, data)
        return ElementUpdateResult(
            key=key,
            new_selector=new_selector,
            updated=True,
            path=path,
            old_selector=old_selector,
            reason=reason,
        )

    @staticmethod
    def _extract_selector(value: Any) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("selector"), str):
            return value["selector"]
        return None

    @staticmethod
    def _replace_selector(elements: Any, key: str, new_selector: str) -> bool:
        current = elements[key]
        if isinstance(current, str):
            elements[key] = new_selector
            return True
        if isinstance(current, dict) and "selector" in current:
            current["selector"] = new_selector
            return True
        return False

    def _default_elements_file(self) -> tuple[Path, Any]:
        elements_dir = self.test_dir / "elements"
        elements_dir.mkdir(parents=True, exist_ok=True)
        path = elements_dir / "generated.yaml"
        if path.exists():
            with path.open("r", encoding="utf-8") as file:
                data = self.yaml.load(file) or {}
        else:
            data = {}
        if not isinstance(data, dict):
            data = {}
        if not isinstance(data.get("elements"), dict):
            data["elements"] = {}
        return path, data

    @staticmethod
    def _qualified_key(key: str, identifier: str | None) -> str:
        identifier = _safe_element_key(identifier or "")
        base = _safe_element_key(key)
        if not identifier:
            return base
        if base.lower().startswith(f"{identifier.lower()}_"):
            return base
        return f"{identifier}_{base}"

    @staticmethod
    def _all_element_keys(data: Any) -> set[str]:
        elements = data.get("elements") if isinstance(data, dict) else {}
        if not isinstance(elements, dict):
            return set()
        return {str(key) for key in elements}

    def _all_project_element_keys(self) -> set[str]:
        elements_dir = self.test_dir / "elements"
        if not elements_dir.exists():
            return set()
        keys: set[str] = set()
        yaml_files = sorted(
            list(elements_dir.glob("**/*.yaml")) + list(elements_dir.glob("**/*.yml")),
            key=lambda file: file.as_posix(),
        )
        for path in yaml_files:
            with path.open("r", encoding="utf-8") as file:
                data = self.yaml.load(file) or {}
            elements = data.get("elements") if isinstance(data, dict) else {}
            if isinstance(elements, dict):
                keys.update(str(key) for key in elements)
        return keys

    @staticmethod
    def _should_update_existing_key(
        key: str,
        occurrence_count: int,
        *,
        identifier: str | None = None,
        allow_semantic_generic_update: bool = False,
    ) -> bool:
        if occurrence_count != 1:
            return False
        if not _is_generic_element_key(key):
            return True
        return bool(
            allow_semantic_generic_update
            and _semantic_identifier_matches_key(key, identifier)
        )

    @staticmethod
    def _unique_element_key(candidate: str, used_keys: set[str]) -> str:
        base = _safe_element_key(candidate)
        if base not in used_keys:
            return base
        index = 2
        while f"{base}_{index}" in used_keys:
            index += 1
        return f"{base}_{index}"

    def _atomic_dump(self, path: Path, data: Any) -> None:
        tmp_path = path.with_name(f"{path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            self.yaml.dump(data, file)
        tmp_path.replace(path)


def _safe_element_key(value: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_\u4e00-\u9fff]+", "_", str(value).strip()).strip("_")
    return key or "element"


def _is_generic_element_key(value: str) -> bool:
    key = _safe_element_key(value)
    lowered = key.lower()
    generic_exact = {
        "button",
        "input",
        "select",
        "login",
        "login_button",
        "username",
        "username_input",
        "password",
        "password_input",
        "submit",
        "submit_button",
        "search",
        "search_button",
        "query",
        "query_button",
        "登录",
        "登录按钮",
        "用户名",
        "用户名输入框",
        "密码",
        "密码输入框",
        "查询",
        "查询按钮",
        "搜索",
        "搜索按钮",
    }
    if lowered in generic_exact or key in generic_exact:
        return True
    if all(ord(char) < 128 for char in key):
        parts = [part for part in re.split(r"[_\s]+", lowered) if part]
        return len(parts) <= 2
    return False


def _semantic_identifier_matches_key(key: str, identifier: str | None) -> bool:
    identifier_keys = _semantic_match_keys(identifier)
    if not identifier_keys:
        return False
    return bool(identifier_keys.intersection(_semantic_match_keys(key)))


def _semantic_match_keys(value: str | None) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    candidates = {
        text,
        _target_from_identifier(text),
        strip_generic_target_words(text),
    }
    result: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        result.add(_semantic_compare_key(candidate))
        for variant in semantic_text_variants(candidate):
            result.add(_semantic_compare_key(variant))
    return {item for item in result if item}


def _target_from_identifier(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    replacements = {
        "btn": "button",
        "ipt": "input",
        "pwd": "password",
    }
    chunks = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", text.replace("-", "_"))
    words: list[str] = []
    for chunk in chunks:
        for part in re.split(r"_+", chunk):
            if not part:
                continue
            camel_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", part)
            for item in camel_parts or [part]:
                lowered = item.lower()
                words.append(replacements.get(lowered, item))
    return " ".join(words).strip() or text


def _semantic_compare_key(value: str | None) -> str:
    return re.sub(r"[\s_\-]+", "", str(value or "").strip().lower())


def register_element_update_thread(thread: threading.Thread) -> None:
    with _PENDING_LOCK:
        _PENDING_THREADS.append(thread)


def wait_for_pending_element_updates(timeout_seconds: float = 2.0) -> None:
    deadline = time.monotonic() + max(timeout_seconds, 0)
    with _PENDING_LOCK:
        threads = list(_PENDING_THREADS)

    for thread in threads:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        thread.join(timeout=remaining)

    with _PENDING_LOCK:
        _PENDING_THREADS[:] = [
            thread for thread in _PENDING_THREADS if thread.is_alive()
        ]


atexit.register(wait_for_pending_element_updates)
