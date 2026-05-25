from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv

from src.ai_runtime.config import load_ai_config
from src.ai_runtime.contracts import VisionFindResult


UI_VISION_BASE_URL_ENV = "UI_VISION_BASE_URL"
UI_VISION_API_KEY_ENV = "UI_VISION_API_KEY"
UI_VISION_ENABLED_ENV = "UI_VISION_ENABLED"
UI_VISION_COORDINATE_FALLBACK_ENV = "UI_VISION_ALLOW_COORDINATE_FALLBACK"


@dataclass(frozen=True)
class VisionSettings:
    enabled: bool = False
    service_url: str = ""
    api_key: str | None = None
    timeout_seconds: int = 30
    min_confidence: float = 0.85
    allow_coordinate_fallback: bool = False
    max_calls_per_test: int = 1
    screenshot_type: str = "png"
    screenshot_full_page: bool = False
    send_dom_candidates: bool = True


class VisionConfigurationError(RuntimeError):
    pass


class VisionServiceUnavailable(RuntimeError):
    pass


def load_vision_settings(config: dict[str, Any] | None = None) -> VisionSettings:
    load_dotenv(".env", override=False)
    cfg = config or load_ai_config()
    vision_cfg = cfg.get("vision", {})
    service_url = (
        os.environ.get(UI_VISION_BASE_URL_ENV)
        or str(vision_cfg.get("service_url") or "")
    ).strip()
    api_key = os.environ.get(UI_VISION_API_KEY_ENV)
    return VisionSettings(
        enabled=_env_bool(UI_VISION_ENABLED_ENV, bool(vision_cfg.get("enabled", False))),
        service_url=service_url.rstrip("/"),
        api_key=api_key or None,
        timeout_seconds=int(vision_cfg.get("timeout_seconds", 30)),
        min_confidence=float(vision_cfg.get("min_confidence", 0.85)),
        allow_coordinate_fallback=_env_bool(
            UI_VISION_COORDINATE_FALLBACK_ENV,
            bool(vision_cfg.get("allow_coordinate_fallback", False)),
        ),
        max_calls_per_test=int(vision_cfg.get("max_calls_per_test", 1)),
        screenshot_type=str(vision_cfg.get("screenshot_type", "png")),
        screenshot_full_page=bool(vision_cfg.get("screenshot_full_page", False)),
        send_dom_candidates=bool(vision_cfg.get("send_dom_candidates", True)),
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class VisionClient:
    def __init__(self, settings: VisionSettings | None = None):
        self.settings = settings or load_vision_settings()
        if self.settings.enabled and not self.settings.service_url:
            raise VisionConfigurationError(
                "UI Vision已启用，但未配置 UI_VISION_BASE_URL 或 vision.service_url。"
            )

    def find(
        self,
        *,
        image_bytes: bytes,
        target: str,
        action: str,
        url: str,
        candidates: list[dict[str, Any]] | None = None,
        context: str | None = None,
    ) -> VisionFindResult:
        if not self.settings.service_url:
            raise VisionConfigurationError("UI Vision服务地址未配置。")

        payload: dict[str, Any] = {
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            "target": target,
            "context": context or f"url={url}; action={action}",
            "mode": "auto",
            "return_candidates": True,
            "coordinate": "pixel",
            "action": action,
            "url": url,
        }
        if self.settings.send_dom_candidates:
            payload["dom_candidates"] = candidates or []

        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

        try:
            response = requests.post(
                f"{self.settings.service_url}/v1/ui/find",
                headers=headers,
                json=payload,
                timeout=self.settings.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise VisionServiceUnavailable(
                f"UI Vision服务不可用: {self.settings.service_url}"
            ) from exc
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            message = response.text.strip()[:1000]
            if response.status_code >= 500:
                raise VisionServiceUnavailable(
                    f"UI Vision服务异常: HTTP {response.status_code} {message}"
                ) from exc
            raise RuntimeError(
                f"UI Vision请求失败: HTTP {response.status_code} {message}"
            ) from exc
        return VisionFindResult.model_validate(response.json())
