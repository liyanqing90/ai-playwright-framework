from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError
import requests
from dotenv import load_dotenv

from src.ai_runtime.config import load_ai_config
from utils.logger import logger
from utils.token_usage import get_token_usage_tracker


LLM_BASE_URL_ENV = "LLM_BASE_URL"
LLM_API_KEY_ENV = "LLM_API_KEY"
LLM_MODEL_ENV = "LLM_MODEL"
LLM_REASONING_EFFORT_ENV = "LLM_REASONING_EFFORT"
LLM_RESPONSE_FORMAT_ENV = "LLM_RESPONSE_FORMAT"
LLM_TIMEOUT_SECONDS_ENV = "LLM_TIMEOUT_SECONDS"


@dataclass(frozen=True)
class LLMSettings:
    url: str
    api_key: str
    model: str
    reasoning_effort: str | None = None
    omit_temperature: bool = True
    response_format: Literal["json_object", "json_schema", "text", "none"] = (
        "json_object"
    )
    timeout_seconds: int = 60
    schema_version: str = "ui-ai-schema-v1"


class LLMConfigurationError(RuntimeError):
    pass


TModel = TypeVar("TModel", bound=BaseModel)


def load_llm_settings() -> LLMSettings:
    load_dotenv(".env", override=False)
    config = load_ai_config()
    llm_cfg = config.get("llm", {})
    base_url = os.environ.get(LLM_BASE_URL_ENV)
    api_key = os.environ.get(LLM_API_KEY_ENV)
    model = os.environ.get(LLM_MODEL_ENV)
    reasoning_effort = os.environ.get(LLM_REASONING_EFFORT_ENV)
    timeout_seconds = int(
        os.environ.get(LLM_TIMEOUT_SECONDS_ENV) or llm_cfg.get("timeout_seconds", 60)
    )
    is_gguf_model = bool(model and model.lower().endswith(".gguf"))
    if is_gguf_model and timeout_seconds < 180:
        timeout_seconds = 180
    if is_gguf_model:
        reasoning_effort = None
    if not base_url or not api_key or not model:
        raise LLMConfigurationError(
            "AI模型未配置。需要设置 LLM_BASE_URL、LLM_API_KEY、LLM_MODEL。"
        )
    response_format = str(
        os.environ.get(LLM_RESPONSE_FORMAT_ENV)
        or llm_cfg.get("response_format", "json_object")
    ).lower()
    if response_format == "auto":
        response_format = "text" if model.lower().endswith(".gguf") else "json_object"
    elif response_format == "json_object" and model.lower().endswith(".gguf"):
        response_format = "text"
    if response_format not in {"json_object", "json_schema", "text", "none"}:
        raise LLMConfigurationError(
            "llm.response_format 只支持 auto/json_object/json_schema/text/none"
        )
    return LLMSettings(
        url=_chat_completions_url(base_url),
        api_key=api_key,
        model=model,
        reasoning_effort=reasoning_effort,
        omit_temperature=bool(llm_cfg.get("omit_temperature", True)),
        response_format=response_format,
        timeout_seconds=timeout_seconds,
        schema_version=str(llm_cfg.get("schema_version", "ui-ai-schema-v1")),
    )


class ChatCompletionProvider:
    def __init__(self, settings: LLMSettings | None = None):
        self.settings = settings or load_llm_settings()

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_json: bool = False,
        response_model: type[BaseModel] | None = None,
        schema_name: str | None = None,
        usage_operation: str = "llm.chat_completion",
        usage_metadata: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
        }
        if self.settings.reasoning_effort:
            payload["reasoning_effort"] = self.settings.reasoning_effort
        response_format = build_response_format(
            settings=self.settings,
            response_json=response_json,
            response_model=response_model,
            schema_name=schema_name,
        )
        if response_format:
            payload["response_format"] = response_format
        if not self.settings.omit_temperature:
            payload["temperature"] = 0.2

        tracker = get_token_usage_tracker()
        try:
            response = requests.post(
                self.settings.url,
                headers={
                    "Authorization": f"Bearer {self.settings.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.settings.timeout_seconds,
            )
        except Exception as exc:
            tracker.record_model_io(
                operation=usage_operation,
                request_payload=payload,
                error=f"{type(exc).__name__}: {exc}",
            )
            if isinstance(exc, requests.exceptions.ReadTimeout):
                raise RuntimeError(
                    "模型请求超时: "
                    f"model={self.settings.model} | timeout_seconds={self.settings.timeout_seconds}。"
                    "本地GGUF模型通常需要更长超时或更小上下文，可设置 "
                    "LLM_TIMEOUT_SECONDS，或在 config/ai_config.yaml 调整 llm.timeout_seconds。"
                ) from exc
            raise
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            if _should_retry_transient_http(response):
                logger.warning(f"模型请求返回HTTP {response.status_code}，自动重试一次")
                try:
                    response = requests.post(
                        self.settings.url,
                        headers={
                            "Authorization": f"Bearer {self.settings.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=self.settings.timeout_seconds,
                    )
                    response.raise_for_status()
                    exc = None
                except requests.HTTPError as retry_exc:
                    exc = retry_exc
                except Exception as retry_exc:
                    tracker.record_model_io(
                        operation=usage_operation,
                        request_payload=payload,
                        error=f"{type(retry_exc).__name__}: {retry_exc}",
                    )
                    raise
            if exc is None:
                pass
            elif _should_retry_with_text_response_format(response, payload):
                logger.warning(
                    "模型端不支持当前 response_format，自动切换为 text 重试一次"
                )
                payload["response_format"] = {"type": "text"}
                try:
                    response = requests.post(
                        self.settings.url,
                        headers={
                            "Authorization": f"Bearer {self.settings.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=self.settings.timeout_seconds,
                    )
                    response.raise_for_status()
                except requests.HTTPError as retry_exc:
                    message = response.text.strip()[:1000]
                    tracker.record_model_io(
                        operation=usage_operation,
                        request_payload=payload,
                        response_payload={
                            "status_code": response.status_code,
                            "text": response.text,
                        },
                        error=f"HTTP {response.status_code}: {message}",
                    )
                    raise RuntimeError(
                        f"模型请求失败: HTTP {response.status_code} {message}"
                    ) from retry_exc
            else:
                message = response.text.strip()[:1000]
                tracker.record_model_io(
                    operation=usage_operation,
                    request_payload=payload,
                    response_payload={
                        "status_code": response.status_code,
                        "text": response.text,
                    },
                    error=f"HTTP {response.status_code}: {message}",
                )
                raise RuntimeError(
                    f"模型请求失败: HTTP {response.status_code} {message}"
                ) from exc
        try:
            raw = response.json()
        except Exception as exc:
            tracker.record_model_io(
                operation=usage_operation,
                request_payload=payload,
                response_payload={
                    "status_code": response.status_code,
                    "text": response.text,
                },
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        tracker.record_model_io(
            operation=usage_operation,
            request_payload=payload,
            response_payload=raw,
        )
        tracker.record_event(
            operation=usage_operation,
            provider="chat_completions",
            model=self.settings.model,
            usage_payload=raw.get("usage"),
            metadata=usage_metadata,
        )
        try:
            message = raw["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("模型响应缺少 choices[0].message.content") from exc
        refusal = message.get("refusal")
        if refusal:
            raise RuntimeError(f"模型拒绝响应: {refusal}")
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("模型响应缺少 choices[0].message.content")
        return content

    def complete_model(
        self,
        messages: list[dict[str, str]],
        response_model: type[TModel],
        *,
        schema_name: str | None = None,
        usage_operation: str = "llm.chat_completion",
        usage_metadata: dict[str, Any] | None = None,
    ) -> TModel:
        content = self.complete(
            messages,
            response_json=True,
            response_model=response_model,
            schema_name=schema_name,
            usage_operation=usage_operation,
            usage_metadata=usage_metadata,
        )
        try:
            return parse_model_response(content, response_model)
        except Exception as exc:
            get_token_usage_tracker().record_model_io(
                operation=f"{usage_operation}.parse_error",
                request_payload={"schema_name": schema_name or response_model.__name__},
                response_payload={"content": content},
                error=f"{type(exc).__name__}: {exc}",
            )
            raise


def _chat_completions_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _should_retry_with_text_response_format(
    response: requests.Response,
    payload: dict[str, Any],
) -> bool:
    response_format = payload.get("response_format")
    if not isinstance(response_format, dict):
        return False
    if response_format.get("type") == "text":
        return False
    if response.status_code != 400:
        return False
    message = response.text.lower()
    return "response_format.type" in message and "text" in message


def _should_retry_transient_http(response: requests.Response) -> bool:
    return response.status_code in {429, 500, 502, 503, 504}


def build_response_format(
    *,
    settings: LLMSettings,
    response_json: bool,
    response_model: type[BaseModel] | None = None,
    schema_name: str | None = None,
) -> dict[str, Any] | None:
    if not response_json and response_model is None:
        return None
    if settings.response_format == "none":
        return None
    if settings.response_format == "text":
        return {"type": "text"}
    if response_model is not None and settings.response_format == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name or response_model.__name__,
                "strict": True,
                "schema": openai_strict_schema(response_model),
            },
        }
    return {"type": "json_object"}


def openai_strict_schema(response_model: type[BaseModel]) -> dict[str, Any]:
    schema = response_model.model_json_schema()
    _normalize_strict_json_schema(schema)
    return schema


def _normalize_strict_json_schema(node: Any) -> None:
    if isinstance(node, dict):
        node.pop("default", None)
        properties = node.get("properties")
        if isinstance(properties, dict):
            node["required"] = list(properties.keys())
            node["additionalProperties"] = False
        for value in node.values():
            _normalize_strict_json_schema(value)
    elif isinstance(node, list):
        for item in node:
            _normalize_strict_json_schema(item)


def parse_json_object(
    content: str,
    *,
    required_keys: set[str] | None = None,
    allowed_keys: set[str] | None = None,
) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = _parse_first_json_object(content)
    if not isinstance(data, dict):
        raise ValueError("模型响应JSON必须是对象")
    if required_keys:
        missing = sorted(required_keys - set(data))
        if missing:
            raise ValueError(f"模型响应缺少必要字段: {', '.join(missing)}")
    if allowed_keys:
        extra = sorted(set(data) - allowed_keys)
        if extra:
            raise ValueError(f"模型响应包含未声明字段: {', '.join(extra)}")
    return data


def parse_model_response(content: str, response_model: type[TModel]) -> TModel:
    data = parse_json_object(content)
    try:
        return response_model.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"模型响应不符合契约: {exc}") from exc


def _parse_first_json_object(content: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            data, _ = decoder.raw_decode(content[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise ValueError("模型响应不是JSON对象")
