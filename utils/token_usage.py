from __future__ import annotations

import atexit
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


DEFAULT_TOKEN_USAGE_DIR = Path("logs") / "token_usage"
_USAGE_KEYS = {
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "input_tokens_details",
    "output_tokens_details",
    "prompt_tokens_details",
    "completion_tokens_details",
}


def normalize_token_usage(
    usage_payload: Mapping[str, Any] | None,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    payload = dict(usage_payload) if isinstance(usage_payload, Mapping) else {}
    input_details = _as_mapping(
        payload.get("input_tokens_details") or payload.get("prompt_tokens_details")
    )
    output_details = _as_mapping(
        payload.get("output_tokens_details") or payload.get("completion_tokens_details")
    )

    input_tokens = _to_int(payload.get("input_tokens", payload.get("prompt_tokens")))
    output_tokens = _to_int(
        payload.get("output_tokens", payload.get("completion_tokens"))
    )
    total_tokens = _to_int(payload.get("total_tokens"))
    if total_tokens == 0 and (input_tokens or output_tokens):
        total_tokens = input_tokens + output_tokens

    cached_input_tokens = _to_int(payload.get("cached_input_tokens"))
    if cached_input_tokens == 0 and "cached_input_tokens" not in payload:
        cached_input_tokens = _to_int(input_details.get("cached_tokens"))

    uncached_input_tokens = _to_int(payload.get("uncached_input_tokens"))
    if uncached_input_tokens == 0 and "uncached_input_tokens" not in payload:
        uncached_input_tokens = max(input_tokens - cached_input_tokens, 0)

    reasoning_tokens = _to_int(payload.get("reasoning_tokens"))
    if reasoning_tokens == 0 and "reasoning_tokens" not in payload:
        reasoning_tokens = _to_int(output_details.get("reasoning_tokens"))

    usage_available = payload.get("usage_available")
    if usage_available is None:
        usage_available = any(key in payload for key in _USAGE_KEYS)
    else:
        usage_available = bool(usage_available)

    return {
        "provider": provider or payload.get("provider"),
        "model": model or payload.get("model"),
        "usage_available": usage_available,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def format_token_usage_summary(summary: Mapping[str, Any] | None) -> str:
    if not summary:
        return "AI token usage: no summary available"
    totals = _as_mapping(summary.get("totals"))
    return (
        "AI token usage: "
        f"calls={totals.get('call_count', 0)}, "
        f"input={totals.get('input_tokens', 0)}, "
        f"output={totals.get('output_tokens', 0)}, "
        f"total={totals.get('total_tokens', 0)}, "
        f"cache_hit_input={totals.get('cached_input_tokens', 0)}, "
        f"cache_miss_input={totals.get('uncached_input_tokens', 0)}, "
        f"usage_unavailable={totals.get('calls_without_usage', 0)}"
    )


class TokenUsageTracker:
    def __init__(self, base_dir: str | Path = DEFAULT_TOKEN_USAGE_DIR):
        self.base_dir = Path(base_dir)
        self._lock = threading.RLock()
        self._active_run: dict[str, Any] | None = None
        self._last_summary: dict[str, Any] | None = None

    @property
    def last_summary(self) -> dict[str, Any] | None:
        with self._lock:
            if self._last_summary is None:
                return None
            return json.loads(json.dumps(self._last_summary, ensure_ascii=False))

    def start_run(
        self,
        *,
        run_kind: str,
        metadata: Mapping[str, Any] | None = None,
        run_id: str | None = None,
    ) -> str:
        with self._lock:
            if self._active_run is not None:
                self._merge_metadata(self._active_run["metadata"], metadata)
                return self._active_run["run_id"]
            current_time = _now()
            self._active_run = {
                "run_id": run_id or f"{run_kind}_{current_time.strftime('%Y%m%d_%H%M%S_%f')}",
                "run_kind": run_kind,
                "started_at": current_time.isoformat(timespec="seconds"),
                "metadata": dict(metadata or {}),
                "events": [],
                "totals": _empty_totals(),
            }
            return self._active_run["run_id"]

    def ensure_run(
        self,
        *,
        run_kind: str = "implicit",
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        return self.start_run(run_kind=run_kind, metadata=metadata)

    def record_event(
        self,
        *,
        operation: str,
        provider: str,
        model: str | None = None,
        usage_payload: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        usage = normalize_token_usage(usage_payload, provider=provider, model=model)
        with self._lock:
            self.ensure_run()
            assert self._active_run is not None
            event = {
                "timestamp": _now().isoformat(timespec="seconds"),
                "operation": operation,
                "provider": usage.get("provider"),
                "model": usage.get("model"),
                "usage": usage,
                "metadata": dict(metadata or {}),
            }
            self._active_run["events"].append(event)
            _merge_usage_totals(self._active_run["totals"], usage)
            return json.loads(json.dumps(event, ensure_ascii=False))

    def snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            if self._active_run is None:
                return None
            snapshot = {
                "run_id": self._active_run["run_id"],
                "run_kind": self._active_run["run_kind"],
                "started_at": self._active_run["started_at"],
                "metadata": dict(self._active_run["metadata"]),
                "totals": dict(self._active_run["totals"]),
                "events": list(self._active_run["events"]),
            }
            return json.loads(json.dumps(snapshot, ensure_ascii=False))

    def finish_run(
        self,
        *,
        status: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            if self._active_run is None:
                return self.last_summary
            self._merge_metadata(self._active_run["metadata"], metadata)
            finished_at = _now().isoformat(timespec="seconds")
            summary_path = self._summary_path(self._active_run["run_id"])
            summary = {
                "run_id": self._active_run["run_id"],
                "run_kind": self._active_run["run_kind"],
                "status": status,
                "started_at": self._active_run["started_at"],
                "finished_at": finished_at,
                "metadata": dict(self._active_run["metadata"]),
                "totals": dict(self._active_run["totals"]),
                "events": list(self._active_run["events"]),
                "summary_file": str(summary_path),
            }
            self._write_summary(summary_path, summary)
            self._write_summary(self.base_dir / "latest.json", summary)
            self._last_summary = summary
            self._active_run = None
            return json.loads(json.dumps(summary, ensure_ascii=False))

    def flush_at_exit(self) -> dict[str, Any] | None:
        with self._lock:
            if self._active_run is None:
                return None
        return self.finish_run(status="process_exit")

    def _summary_path(self, run_id: str) -> Path:
        return self.base_dir / f"{run_id}.json"

    def _write_summary(self, path: Path, summary: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _merge_metadata(
        current: dict[str, Any],
        metadata: Mapping[str, Any] | None,
    ) -> None:
        if not metadata:
            return
        for key, value in metadata.items():
            if value is not None:
                current[key] = value


def get_token_usage_tracker() -> TokenUsageTracker:
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = TokenUsageTracker()
    return _TRACKER


def _empty_totals() -> dict[str, int]:
    return {
        "call_count": 0,
        "calls_with_usage": 0,
        "calls_without_usage": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "uncached_input_tokens": 0,
        "reasoning_tokens": 0,
    }


def _merge_usage_totals(totals: dict[str, int], usage: Mapping[str, Any]) -> None:
    totals["call_count"] += 1
    if usage.get("usage_available"):
        totals["calls_with_usage"] += 1
    else:
        totals["calls_without_usage"] += 1
    totals["input_tokens"] += _to_int(usage.get("input_tokens"))
    totals["output_tokens"] += _to_int(usage.get("output_tokens"))
    totals["total_tokens"] += _to_int(usage.get("total_tokens"))
    totals["cached_input_tokens"] += _to_int(usage.get("cached_input_tokens"))
    totals["uncached_input_tokens"] += _to_int(usage.get("uncached_input_tokens"))
    totals["reasoning_tokens"] += _to_int(usage.get("reasoning_tokens"))


def _to_int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _now() -> datetime:
    return datetime.now().astimezone()


_TRACKER: TokenUsageTracker | None = None
atexit.register(lambda: get_token_usage_tracker().flush_at_exit())

