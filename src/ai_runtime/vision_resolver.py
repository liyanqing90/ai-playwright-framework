from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.ai_runtime.contracts import VisionFindResult
from src.ai_runtime.playwright_selectors import normalize_selector, verify_selector
from src.ai_runtime.vision_client import VisionClient, VisionSettings


@dataclass(frozen=True)
class VisionResolution:
    selector: str | None
    source: str
    confidence: float
    method: str | None = None
    reason: str | None = None
    coordinate: tuple[float, float] | None = None


class VisionResolver:
    def __init__(
        self,
        page,
        *,
        settings: VisionSettings,
        client: VisionClient | None = None,
    ):
        self.page = page
        self.settings = settings
        self.client = client or VisionClient(settings)

    def resolve(
        self,
        *,
        action: str,
        target: str,
        timeout: int,
        candidates: list[dict[str, Any]],
    ) -> VisionResolution:
        screenshot = self.page.screenshot(
            type=self.settings.screenshot_type,
            full_page=self.settings.screenshot_full_page,
        )
        result = self.client.find(
            image_bytes=screenshot,
            target=target,
            action=action,
            url=getattr(self.page, "url", "") or "",
            candidates=candidates,
        )
        if not result.found:
            reason = result.reason or result.error_code or "UI Vision未找到目标元素"
            raise ValueError(reason)
        if result.confidence < self.settings.min_confidence:
            raise ValueError(
                f"UI Vision置信度不足: {result.confidence} < {self.settings.min_confidence}"
            )

        selector = self._verified_selector_from_result(
            result=result,
            action=action,
            timeout=timeout,
            candidates=candidates,
        )
        if selector:
            return VisionResolution(
                selector=selector,
                source="vision_dom",
                confidence=result.confidence,
                method=result.method,
                reason=result.reason,
            )

        coordinate = _result_coordinate(result)
        if (
            coordinate
            and self.settings.allow_coordinate_fallback
            and action in {"click", "fill", "press", "press_key"}
        ):
            return VisionResolution(
                selector=None,
                source="vision_coordinate",
                confidence=result.confidence,
                method=result.method,
                reason=result.reason,
                coordinate=coordinate,
            )

        raise ValueError(
            "UI Vision只返回坐标，当前未启用 coordinate fallback 或该 action 不支持坐标兜底。"
        )

    def _verified_selector_from_result(
        self,
        *,
        result: VisionFindResult,
        action: str,
        timeout: int,
        candidates: list[dict[str, Any]],
    ) -> str | None:
        candidate_selectors: list[str] = []
        if result.selector:
            candidate_selectors.append(result.selector)

        selected = _candidate_by_selected_index(result, candidates)
        if selected and selected.get("selector"):
            candidate_selectors.append(str(selected["selector"]))

        spatial = _candidate_by_geometry(result, candidates)
        if spatial and spatial.get("selector"):
            candidate_selectors.append(str(spatial["selector"]))

        seen: set[str] = set()
        for raw_selector in candidate_selectors:
            selector = normalize_selector(raw_selector)
            if not selector or selector in seen:
                continue
            seen.add(selector)
            try:
                verify_selector(self.page, selector, action=action, timeout=timeout)
                return selector
            except Exception:
                continue
        return None


def _candidate_by_selected_index(
    result: VisionFindResult, candidates: list[dict[str, Any]]
) -> dict[str, Any] | None:
    selected = result.selected_candidate_index
    if selected is None:
        selected = result.selected_candidate_id
    if selected is None:
        return None
    for position, candidate in enumerate(candidates):
        if position == selected or candidate.get("index") == selected:
            return candidate
    return None


def _candidate_by_geometry(
    result: VisionFindResult, candidates: list[dict[str, Any]]
) -> dict[str, Any] | None:
    center = _result_coordinate(result)
    if center:
        containing = [
            candidate
            for candidate in candidates
            if _box_contains(candidate.get("bbox"), center)
        ]
        if containing:
            return min(containing, key=lambda item: _box_area(item.get("bbox")))

    if result.box:
        scored = [
            (candidate, _iou(candidate.get("bbox"), result.box))
            for candidate in candidates
            if candidate.get("bbox")
        ]
        scored = [(candidate, score) for candidate, score in scored if score > 0]
        if scored:
            return max(scored, key=lambda item: item[1])[0]
    return None


def _result_coordinate(result: VisionFindResult) -> tuple[float, float] | None:
    if result.center and len(result.center) >= 2:
        return float(result.center[0]), float(result.center[1])
    if result.box and len(result.box) >= 4:
        return (
            float(result.box[0] + (result.box[2] - result.box[0]) / 2),
            float(result.box[1] + (result.box[3] - result.box[1]) / 2),
        )
    return None


def _box_contains(box: Any, point: tuple[float, float]) -> bool:
    if not isinstance(box, list) or len(box) < 4:
        return False
    x, y = point
    return float(box[0]) <= x <= float(box[2]) and float(box[1]) <= y <= float(box[3])


def _box_area(box: Any) -> float:
    if not isinstance(box, list) or len(box) < 4:
        return float("inf")
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def _iou(a: Any, b: Any) -> float:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) < 4 or len(b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _box_area(a) + _box_area(b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union
