"""Scene-zone helpers for zebra-crossing awareness and warning logic."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from src.config import (
    SCENE_AUTO_APPROACH_SCALE_X,
    SCENE_AUTO_APPROACH_SCALE_Y,
    SCENE_AUTO_CONFLICT_SCALE_X,
    SCENE_AUTO_CONFLICT_SCALE_Y,
    SCENE_AUTO_FOCUS_SCALE_X,
    SCENE_AUTO_FOCUS_SCALE_Y,
    SCENE_ZONE_CONFIG_PATH,
    SCENE_ZONE_ENABLE,
)
from src.zebra_crossing_detection.zebra_detector import ZebraDetectionResult


Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]

ZONE_OUTSIDE = "outside"
ZONE_APPROACH = "approach"
ZONE_ZEBRA = "on_zebra"
ZONE_CONFLICT = "conflict"


@dataclass
class ScenePolygon:
    name: str
    points: List[Point] = field(default_factory=list)
    color: Tuple[int, int, int] = (0, 0, 0)
    alpha: float = 0.0

    def contour(self) -> np.ndarray | None:
        if len(self.points) < 3:
            return None
        return np.array(self.points, dtype=np.int32).reshape(-1, 1, 2)

    def bounding_rect(self) -> Rect | None:
        contour = self.contour()
        if contour is None:
            return None
        x, y, w, h = cv2.boundingRect(contour)
        return x, y, x + w, y + h


@dataclass
class SceneContext:
    source: str = "none"
    zebra_zone: ScenePolygon | None = None
    approach_zone: ScenePolygon | None = None
    conflict_zone: ScenePolygon | None = None
    focus_roi: Rect | None = None

    def classify_point(self, point: Tuple[float, float]) -> str:
        px, py = point
        if self.zebra_zone and point_in_polygon((px, py), self.zebra_zone.points):
            return ZONE_ZEBRA
        if self.approach_zone and point_in_polygon((px, py), self.approach_zone.points):
            return ZONE_APPROACH
        if self.conflict_zone and point_in_polygon((px, py), self.conflict_zone.points):
            return ZONE_CONFLICT
        return ZONE_OUTSIDE


class SceneZoneManager:
    """Load manual scene zones or derive them from zebra detections."""

    def __init__(
        self,
        enabled: bool = SCENE_ZONE_ENABLE,
        config_path: str = SCENE_ZONE_CONFIG_PATH,
    ) -> None:
        self.enabled = enabled
        self.config_path = Path(config_path)
        self._manual_cache: Dict[str, object] | None = None
        self._load_attempted = False

    def build(self, frame_shape: Tuple[int, int, int], zebra_result: ZebraDetectionResult) -> SceneContext:
        if not self.enabled:
            return SceneContext(source="disabled")

        manual = self._load_manual_config()
        if manual:
            context = self._context_from_manual(frame_shape, manual)
            if context is not None:
                return context

        if zebra_result.boxes:
            return self._context_from_zebra(frame_shape, zebra_result)
        return SceneContext(source="none")

    def _load_manual_config(self) -> Dict[str, object] | None:
        if self._load_attempted:
            return self._manual_cache

        self._load_attempted = True
        if not self.config_path.exists():
            self._manual_cache = None
            return None

        try:
            self._manual_cache = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[SceneZones] Failed to load config {self.config_path}: {exc}")
            self._manual_cache = None
        return self._manual_cache

    def _context_from_manual(
        self,
        frame_shape: Tuple[int, int, int],
        payload: Dict[str, object],
    ) -> SceneContext | None:
        frame_h, frame_w = frame_shape[:2]
        normalized = bool(payload.get("normalized", True))

        zebra_points = self._read_polygon(payload.get("zebra_zone"), frame_w, frame_h, normalized)
        approach_points = self._read_polygon(payload.get("approach_zone"), frame_w, frame_h, normalized)
        conflict_points = self._read_polygon(payload.get("conflict_zone"), frame_w, frame_h, normalized)
        focus_roi = self._read_rect(payload.get("focus_roi"), frame_w, frame_h, normalized)

        if not zebra_points and not approach_points and not conflict_points:
            return None

        if zebra_points and not approach_points:
            approach_points = expand_polygon(
                zebra_points,
                frame_w,
                frame_h,
                scale_x=SCENE_AUTO_APPROACH_SCALE_X,
                scale_y=SCENE_AUTO_APPROACH_SCALE_Y,
            )
        if zebra_points and not conflict_points:
            conflict_points = expand_polygon(
                zebra_points,
                frame_w,
                frame_h,
                scale_x=SCENE_AUTO_CONFLICT_SCALE_X,
                scale_y=SCENE_AUTO_CONFLICT_SCALE_Y,
            )

        return SceneContext(
            source="manual",
            zebra_zone=ScenePolygon("zebra", zebra_points, (0, 220, 120), 0.14) if zebra_points else None,
            approach_zone=ScenePolygon("approach", approach_points, (255, 196, 0), 0.08) if approach_points else None,
            conflict_zone=ScenePolygon("conflict", conflict_points, (255, 96, 96), 0.08) if conflict_points else None,
            focus_roi=focus_roi or self._focus_from_polygons(frame_w, frame_h, zebra_points, approach_points, conflict_points),
        )

    def _context_from_zebra(
        self,
        frame_shape: Tuple[int, int, int],
        zebra_result: ZebraDetectionResult,
    ) -> SceneContext:
        frame_h, frame_w = frame_shape[:2]
        zebra_points = self._merge_zebra_polygons(zebra_result)
        if zebra_points:
            zebra_rect = self._rect_from_points(zebra_points, frame_w, frame_h)
        else:
            x1 = min(box[0] for box in zebra_result.boxes)
            y1 = min(box[1] for box in zebra_result.boxes)
            x2 = max(box[2] for box in zebra_result.boxes)
            y2 = max(box[3] for box in zebra_result.boxes)
            zebra_rect = clip_rect((x1, y1, x2, y2), frame_w, frame_h)
            zebra_points = rect_to_polygon(zebra_rect)

        approach_rect = expand_rect(
            zebra_rect,
            frame_w,
            frame_h,
            scale_x=SCENE_AUTO_APPROACH_SCALE_X,
            scale_y=SCENE_AUTO_APPROACH_SCALE_Y,
        )
        approach_points = expand_polygon(
            zebra_points,
            frame_w,
            frame_h,
            scale_x=SCENE_AUTO_APPROACH_SCALE_X,
            scale_y=SCENE_AUTO_APPROACH_SCALE_Y,
        )
        conflict_rect = expand_rect(
            zebra_rect,
            frame_w,
            frame_h,
            scale_x=SCENE_AUTO_CONFLICT_SCALE_X,
            scale_y=SCENE_AUTO_CONFLICT_SCALE_Y,
        )
        conflict_points = expand_polygon(
            zebra_points,
            frame_w,
            frame_h,
            scale_x=SCENE_AUTO_CONFLICT_SCALE_X,
            scale_y=SCENE_AUTO_CONFLICT_SCALE_Y,
        )
        focus_roi = expand_rect(
            zebra_rect,
            frame_w,
            frame_h,
            scale_x=SCENE_AUTO_FOCUS_SCALE_X,
            scale_y=SCENE_AUTO_FOCUS_SCALE_Y,
        )

        return SceneContext(
            source="auto",
            zebra_zone=ScenePolygon("zebra", zebra_points, (0, 220, 120), 0.14),
            approach_zone=ScenePolygon("approach", approach_points or rect_to_polygon(approach_rect), (255, 196, 0), 0.08),
            conflict_zone=ScenePolygon("conflict", conflict_points or rect_to_polygon(conflict_rect), (255, 96, 96), 0.08),
            focus_roi=focus_roi,
        )

    def _read_polygon(
        self,
        raw_points: object,
        frame_w: int,
        frame_h: int,
        normalized: bool,
    ) -> List[Point]:
        if not isinstance(raw_points, Sequence):
            return []

        points: List[Point] = []
        for item in raw_points:
            if not isinstance(item, Sequence) or len(item) < 2:
                continue
            x = float(item[0])
            y = float(item[1])
            if normalized:
                x *= frame_w
                y *= frame_h
            points.append((int(round(x)), int(round(y))))
        return points

    def _read_rect(
        self,
        raw_rect: object,
        frame_w: int,
        frame_h: int,
        normalized: bool,
    ) -> Rect | None:
        if not isinstance(raw_rect, Sequence) or len(raw_rect) < 4:
            return None
        x1, y1, x2, y2 = [float(value) for value in raw_rect[:4]]
        if normalized:
            x1 *= frame_w
            x2 *= frame_w
            y1 *= frame_h
            y2 *= frame_h
        return clip_rect((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))), frame_w, frame_h)

    def _focus_from_polygons(
        self,
        frame_w: int,
        frame_h: int,
        *polygons: List[Point],
    ) -> Rect | None:
        merged = [point for polygon in polygons for point in polygon]
        if not merged:
            return None
        xs = [point[0] for point in merged]
        ys = [point[1] for point in merged]
        rect = (min(xs), min(ys), max(xs), max(ys))
        return expand_rect(rect, frame_w, frame_h, SCENE_AUTO_FOCUS_SCALE_X, SCENE_AUTO_FOCUS_SCALE_Y)

    def _merge_zebra_polygons(self, zebra_result: ZebraDetectionResult) -> List[Point]:
        if not zebra_result.polygons:
            return []

        merged = np.array([point for polygon in zebra_result.polygons for point in polygon], dtype=np.int32)
        if len(merged) < 3:
            return []
        hull = cv2.convexHull(merged.reshape(-1, 1, 2))
        return [(int(point[0][0]), int(point[0][1])) for point in hull]

    def _rect_from_points(self, points: List[Point], frame_w: int, frame_h: int) -> Rect:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return clip_rect((min(xs), min(ys), max(xs), max(ys)), frame_w, frame_h)


def rect_to_polygon(rect: Rect) -> List[Point]:
    x1, y1, x2, y2 = rect
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def expand_rect(rect: Rect, frame_w: int, frame_h: int, scale_x: float, scale_y: float) -> Rect:
    x1, y1, x2, y2 = rect
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    expand_x = int(round(width * scale_x))
    expand_y = int(round(height * scale_y))
    return clip_rect((x1 - expand_x, y1 - expand_y, x2 + expand_x, y2 + expand_y), frame_w, frame_h)


def clip_rect(rect: Rect, frame_w: int, frame_h: int) -> Rect:
    x1, y1, x2, y2 = rect
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(x1 + 1, min(x2, frame_w))
    y2 = max(y1 + 1, min(y2, frame_h))
    return x1, y1, x2, y2


def expand_polygon(
    polygon: Sequence[Point],
    frame_w: int,
    frame_h: int,
    scale_x: float,
    scale_y: float,
) -> List[Point]:
    if len(polygon) < 3:
        return []

    points = np.array(polygon, dtype=np.float32)
    centroid = points.mean(axis=0)
    expanded = []
    for x, y in points:
        dx = x - centroid[0]
        dy = y - centroid[1]
        nx = x + dx * scale_x
        ny = y + dy * scale_y
        expanded.append(
            (
                int(np.clip(round(nx), 0, frame_w - 1)),
                int(np.clip(round(ny), 0, frame_h - 1)),
            )
        )
    return expanded


def point_in_polygon(point: Tuple[float, float], polygon: Sequence[Point]) -> bool:
    if len(polygon) < 3:
        return False
    contour = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(contour, point, False) >= 0
