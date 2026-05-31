from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


DEFAULT_AI_CACHE_SQLITE_PATH = ".ui_auto/ai_cache.sqlite3"


class AiCacheStore:
    """Shared SQLite-backed cache for AI intermediate and final artifacts."""

    def __init__(self, sqlite_path: str | Path = DEFAULT_AI_CACHE_SQLITE_PATH):
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

    def get_payload(self, *, namespace: str, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT payload_json FROM ai_cache_records
                WHERE namespace = ? AND cache_key = ?
                  AND status IN ('active', 'verified')
                LIMIT 1
                """,
                (namespace, key),
            ).fetchone()
        if not row:
            return None
        return _loads_dict(row["payload_json"])

    def find_latest_payload(
        self,
        *,
        namespace: str,
        project: str,
        env: str,
        case_name: str = "",
        input_type: str = "",
        model: str = "",
        entry_scope: str = "",
        prompt_version: str = "",
        schema_version: str = "",
    ) -> dict[str, Any] | None:
        sql = """
            SELECT payload_json FROM ai_cache_records
            WHERE namespace = ?
              AND project = ?
              AND env = ?
              AND case_name = ?
              AND input_type = ?
              AND entry_scope = ?
              AND prompt_version = ?
              AND schema_version = ?
              AND status IN ('active', 'verified')
        """
        params: list[Any] = [
            namespace,
            project,
            env,
            case_name,
            input_type,
            entry_scope,
            prompt_version,
            schema_version,
        ]
        if model:
            sql += " AND model = ?"
            params.append(model)
        sql += " ORDER BY updated_at DESC LIMIT 1"
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        if not row:
            return None
        return _loads_dict(row["payload_json"])

    def put_payload(
        self,
        *,
        namespace: str,
        key: str,
        project: str,
        env: str,
        entry_scope: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        case_name: str = "",
        input_type: str = "",
        model: str = "",
        prompt_version: str = "",
        schema_version: str = "",
        spec_hash: str = "",
        asset_hash: str = "",
        status: str = "active",
    ) -> None:
        now = int(time.time())
        metadata = metadata or {}
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO ai_cache_records (
                    namespace, cache_key, project, env, entry_scope, case_name,
                    input_type, model, prompt_version, schema_version, spec_hash,
                    asset_hash, status, payload_json, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, cache_key) DO UPDATE SET
                    project = excluded.project,
                    env = excluded.env,
                    entry_scope = excluded.entry_scope,
                    case_name = excluded.case_name,
                    input_type = excluded.input_type,
                    model = excluded.model,
                    prompt_version = excluded.prompt_version,
                    schema_version = excluded.schema_version,
                    spec_hash = excluded.spec_hash,
                    asset_hash = excluded.asset_hash,
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    namespace,
                    key,
                    project,
                    env,
                    entry_scope,
                    case_name,
                    input_type,
                    model,
                    prompt_version,
                    schema_version,
                    spec_hash,
                    asset_hash,
                    status,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def mark_stale(self, *, namespace: str, key: str, reason: str = "") -> None:
        now = int(time.time())
        with self._lock:
            self._conn.execute(
                """
                UPDATE ai_cache_records
                SET status = 'stale',
                    metadata_json = ?,
                    updated_at = ?
                WHERE namespace = ? AND cache_key = ?
                """,
                (
                    json.dumps(
                        {"stale_reason": reason, "stale_at": now},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    now,
                    namespace,
                    key,
                ),
            )
            self._conn.commit()

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_cache_records (
                    namespace TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    project TEXT NOT NULL,
                    env TEXT NOT NULL,
                    entry_scope TEXT NOT NULL DEFAULT '',
                    case_name TEXT NOT NULL DEFAULT '',
                    input_type TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    prompt_version TEXT NOT NULL DEFAULT '',
                    schema_version TEXT NOT NULL DEFAULT '',
                    spec_hash TEXT NOT NULL DEFAULT '',
                    asset_hash TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    payload_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY(namespace, cache_key)
                )
                """
            )
            self._ensure_columns()
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ai_cache_lookup
                ON ai_cache_records(
                    namespace, project, env, case_name, input_type, entry_scope,
                    prompt_version, schema_version, updated_at
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ai_cache_scope
                ON ai_cache_records(namespace, project, env, entry_scope, updated_at)
                """
            )
            self._conn.commit()

    def _ensure_columns(self) -> None:
        existing = {
            row["name"]
            for row in self._conn.execute(
                "PRAGMA table_info(ai_cache_records)"
            ).fetchall()
        }
        if "status" not in existing:
            self._conn.execute(
                "ALTER TABLE ai_cache_records ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"
            )


def ai_cache_path_from_config(config: dict[str, Any]) -> str:
    runtime_cfg = config.get("runtime") or {}
    generation_cfg = config.get("generation") or {}
    return str(
        runtime_cfg.get("ai_cache_sqlite_path")
        or generation_cfg.get("ai_cache_sqlite_path")
        or DEFAULT_AI_CACHE_SQLITE_PATH
    )


def _loads_dict(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except Exception:
        return None
    return value if isinstance(value, dict) else None
