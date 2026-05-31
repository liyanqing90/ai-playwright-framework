from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ai_playwright.ai_runtime.ai_cache_store import AiCacheStore
from ai_playwright.utils.logger import logger


class AgentCasePlanCache:
    """Compiled agent_case plan cache. The plan references runtime YAML assets."""

    def __init__(self, path: str | Path):
        self.store = AiCacheStore(path)
        self.namespace = "agent_case_plan"

    def load_plan(self, key: str) -> dict[str, Any] | None:
        try:
            return self.store.get_payload(namespace=self.namespace, key=key)
        except Exception as exc:
            logger.warning(f"agent_case plan缓存读取失败，忽略缓存: {exc}")
            return None

    def save_plan(
        self,
        *,
        key: str,
        project: str,
        env: str,
        case_name: str,
        entry_scope: str,
        spec: dict[str, Any],
        payload: dict[str, Any],
        case_payload_name: str,
        steps: list[dict[str, Any]],
        prompt_version: str,
        schema_version: str,
        model: str,
        asset_hash: str,
    ) -> None:
        try:
            self.store.put_payload(
                namespace=self.namespace,
                key=key,
                project=project,
                env=env,
                entry_scope=entry_scope,
                case_name=case_name,
                input_type=str(spec.get("input_type") or ""),
                model=model,
                prompt_version=prompt_version,
                schema_version=schema_version,
                spec_hash=_hash_payload(_plan_cache_spec(spec)),
                asset_hash=asset_hash,
                payload={
                    "case_name": case_payload_name,
                    "steps": _cacheable_plan_steps(steps),
                    "payload": _cacheable_plan_payload(payload),
                },
                metadata={
                    "intent": spec.get("intent"),
                    "steps": spec.get("steps"),
                    "inputs": spec.get("inputs"),
                    "criteria": spec.get("criteria"),
                    "updated_at": int(time.time()),
                },
                status="verified",
            )
        except Exception as exc:
            logger.warning(f"agent_case plan缓存写入失败，不阻塞执行: {exc}")


def _plan_cache_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "description": spec.get("description"),
        "intent": spec.get("intent"),
        "steps": spec.get("steps"),
        "inputs": spec.get("inputs"),
        "criteria": spec.get("criteria"),
        "entry_scope": spec.get("entry_scope"),
    }


def _cacheable_plan_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_clean_plan_step(step) for step in steps if isinstance(step, dict)]


def _cacheable_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    if not isinstance(result, dict):
        return {}
    result["elements"] = {}
    result["modules"] = {}
    result["vars"] = {}
    return result


def _compiled_payload_safe_for_plan_cache(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    for key in ("elements", "modules", "vars"):
        value = payload.get(key)
        if isinstance(value, dict) and value:
            return False
    return True


def _clean_plan_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in step.items()
        if (
            (
                not str(key).startswith("_")
                or key
                in {"_resolved_selector", "_resolved_value", "_resolved_value_after"}
            )
            and not str(key).startswith("_module_")
        )
    }


def _hash_payload(data: dict[str, Any]) -> str:
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


__all__ = [
    "AgentCasePlanCache",
    "_cacheable_plan_payload",
    "_cacheable_plan_steps",
    "_compiled_payload_safe_for_plan_cache",
    "_hash_payload",
]
