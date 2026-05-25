from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SelectorRecord:
    id: str
    project: str
    env: str
    page_key: str
    action: str
    target: str
    selector: str
    source: str
    confidence: float
    success_count: int
    fail_count: int
    status: str
    prompt_version: str | None = None
    schema_version: str | None = None
    model: str | None = None
    candidate_hash: str | None = None
    candidate_count: int | None = None
    last_error: str | None = None


class SelectorRegistry:
    def __init__(self, sqlite_path: str | Path):
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.sqlite_path), timeout=30, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS selectors (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    env TEXT NOT NULL,
                    page_key TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target TEXT NOT NULL,
                    selector TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    fail_count INTEGER NOT NULL DEFAULT 0,
                    last_success_at TEXT,
                    last_failed_at TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_selectors_lookup
                ON selectors(project, env, page_key, action, target, status)
                """
            )
            self._ensure_columns()
            self._conn.commit()

    def _ensure_columns(self) -> None:
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(selectors)").fetchall()
        }
        columns = {
            "prompt_version": "TEXT",
            "schema_version": "TEXT",
            "model": "TEXT",
            "candidate_hash": "TEXT",
            "candidate_count": "INTEGER",
            "last_error": "TEXT",
        }
        for name, definition in columns.items():
            if name not in existing:
                self._conn.execute(
                    f"ALTER TABLE selectors ADD COLUMN {name} {definition}"
                )

    def find(
        self,
        *,
        project: str,
        env: str,
        page_key: str,
        action: str,
        target: str,
    ) -> SelectorRecord | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM selectors
                WHERE project = ?
                  AND env = ?
                  AND page_key = ?
                  AND action = ?
                  AND target = ?
                  AND status IN ('active', 'unstable')
                ORDER BY
                  CASE status WHEN 'active' THEN 0 ELSE 1 END,
                  success_count DESC,
                  fail_count ASC,
                  COALESCE(last_success_at, '') DESC,
                  confidence DESC
                LIMIT 1
                """,
                (project, env, page_key, action, target),
            ).fetchone()
        return self._to_record(row) if row else None

    def save(
        self,
        *,
        project: str,
        env: str,
        page_key: str,
        action: str,
        target: str,
        selector: str,
        source: str,
        confidence: float = 1.0,
        prompt_version: str | None = None,
        schema_version: str | None = None,
        model: str | None = None,
        candidate_hash: str | None = None,
        candidate_count: int | None = None,
        replace_active: bool = False,
    ) -> SelectorRecord:
        now = _now()
        record_id = str(uuid.uuid4())
        with self._lock:
            if replace_active:
                self._conn.execute(
                    """
                    UPDATE selectors
                    SET status = 'deprecated', updated_at = ?
                    WHERE project = ? AND env = ? AND page_key = ?
                      AND action = ? AND target = ? AND status = 'active'
                    """,
                    (now, project, env, page_key, action, target),
                )
            self._conn.execute(
                """
                INSERT INTO selectors (
                    id, project, env, page_key, action, target, selector, source,
                    confidence, success_count, fail_count, last_success_at,
                    status, created_at, updated_at, prompt_version, schema_version,
                    model, candidate_hash, candidate_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?, 'active', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    project,
                    env,
                    page_key,
                    action,
                    target,
                    selector,
                    source,
                    confidence,
                    now,
                    now,
                    now,
                    prompt_version,
                    schema_version,
                    model,
                    candidate_hash,
                    candidate_count,
                ),
            )
            self._conn.commit()
        found = self.find(
            project=project, env=env, page_key=page_key, action=action, target=target
        )
        assert found is not None
        return found

    def mark_success(self, record_id: str) -> None:
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                UPDATE selectors
                SET success_count = success_count + 1,
                    last_success_at = ?,
                    last_error = NULL,
                    status = CASE WHEN status = 'unstable' THEN 'active' ELSE status END,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, record_id),
            )
            self._conn.commit()

    def mark_failed(
        self,
        record_id: str,
        *,
        unstable_threshold: int,
        last_error: str | None = None,
    ) -> None:
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                UPDATE selectors
                SET fail_count = fail_count + 1,
                    last_failed_at = ?,
                    last_error = ?,
                    status = CASE
                        WHEN fail_count + 1 >= ? THEN 'unstable'
                        ELSE status
                    END,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, last_error, unstable_threshold, now, record_id),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @staticmethod
    def _to_record(row: sqlite3.Row) -> SelectorRecord:
        values: dict[str, Any] = dict(row)
        return SelectorRecord(
            id=values["id"],
            project=values["project"],
            env=values["env"],
            page_key=values["page_key"],
            action=values["action"],
            target=values["target"],
            selector=values["selector"],
            source=values["source"],
            confidence=float(values["confidence"]),
            success_count=int(values["success_count"]),
            fail_count=int(values["fail_count"]),
            status=values["status"],
            prompt_version=values.get("prompt_version"),
            schema_version=values.get("schema_version"),
            model=values.get("model"),
            candidate_hash=values.get("candidate_hash"),
            candidate_count=(
                int(values["candidate_count"])
                if values.get("candidate_count") is not None
                else None
            ),
            last_error=values.get("last_error"),
        )
