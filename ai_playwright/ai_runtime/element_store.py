from __future__ import annotations

import os
import atexit
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from ai_playwright.ai_runtime.playwright_selectors import (
    canonicalize_persisted_selector,
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
        if located is None:
            return ElementUpdateResult(
                key=key,
                new_selector=new_selector,
                updated=False,
                reason="element_key_not_found",
            )

        path, data, current_value = located
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
        )

    def _locate_element_file(self, key: str) -> tuple[Path, Any, Any] | None:
        elements_dir = self.test_dir / "elements"
        if not elements_dir.exists():
            return None

        found: tuple[Path, Any, Any] | None = None
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
                found = (path, data, elements[key])
        return found

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

    def _atomic_dump(self, path: Path, data: Any) -> None:
        tmp_path = path.with_name(f"{path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            self.yaml.dump(data, file)
        tmp_path.replace(path)


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
