from __future__ import annotations

import re
from typing import Any

from ai_playwright.ai_runtime.config import (
    LLM_DATA_POLICY_EXTERNAL,
    LLM_DATA_POLICY_TRUSTED_LOCAL,
    normalize_llm_data_policy,
)


_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]{1,64}@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_CN_PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
_CN_ID_RE = re.compile(
    r"(?<![0-9A-Za-z])\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])"
    r"(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?![0-9A-Za-z])"
)
_JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
)
_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|"
    r"passwd|authorization|auth)\b\s*[:=]\s*(bearer\s+)?([^\s,;\"'<>]+)"
)
_SENSITIVE_QUERY_RE = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|"
    r"password|email|phone)=)([^&#\s]+)"
)
_LONG_TOKEN_RE = re.compile(r"(?<![\w-])[A-Za-z0-9_-]{24,}(?![\w-])")


def redact_value(
    value: Any,
    *,
    policy: str | None = None,
) -> Any:
    """Redact sensitive LLM-bound values unless the caller marks the model trusted."""

    normalized_policy = normalize_llm_data_policy(
        policy, default=LLM_DATA_POLICY_EXTERNAL
    )
    if normalized_policy == LLM_DATA_POLICY_TRUSTED_LOCAL:
        return value
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        return [redact_value(item, policy=normalized_policy) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item, policy=normalized_policy) for item in value)
    if isinstance(value, dict):
        return {
            key: redact_value(item, policy=normalized_policy)
            for key, item in value.items()
        }
    return value


def _redact_text(value: str) -> str:
    text = str(value)
    text = _CREDENTIAL_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}=<redacted:credential>",
        text,
    )
    text = _SENSITIVE_QUERY_RE.sub(
        lambda match: f"{match.group(1)}<redacted:credential>",
        text,
    )
    text = _JWT_RE.sub("<redacted:token>", text)
    text = _EMAIL_RE.sub("<redacted:email>", text)
    text = _CN_PHONE_RE.sub("<redacted:phone>", text)
    text = _CN_ID_RE.sub("<redacted:id-card>", text)
    text = _LONG_TOKEN_RE.sub("<redacted:token>", text)
    return text


__all__ = [
    "LLM_DATA_POLICY_EXTERNAL",
    "LLM_DATA_POLICY_TRUSTED_LOCAL",
    "redact_value",
]
