from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionResult:
    success: bool
    data: Any = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    failure_kind: str | None = None

    @classmethod
    def ok(cls, data: Any = None, artifacts: dict[str, Any] | None = None):
        return cls(success=True, data=data, artifacts=artifacts or {})

    @classmethod
    def failed(cls, failure_kind: str, data: Any = None):
        return cls(success=False, data=data, failure_kind=failure_kind)

    def to_step_dict(self) -> dict[str, Any]:
        payload = {
            "success": self.success,
            "failure_kind": self.failure_kind,
            "artifacts": self.artifacts,
        }
        if self.data is not None:
            payload["data"] = _safe_data(self.data)
        return payload


def _safe_data(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_safe_data(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_data(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_data(item) for key, item in value.items()}
    return f"<{value.__class__.__name__}>"
