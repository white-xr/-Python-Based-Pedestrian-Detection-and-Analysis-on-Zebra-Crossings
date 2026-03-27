"""Traditional zebra crossing detector based on edge and contour analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

from src.config import (
    ZEBRA_CANNY_THRESHOLDS,
    ZEBRA_MAX_ASPECT,
    ZEBRA_MIN_AREA,
    ZEBRA_MIN_ASPECT,
)


@dataclass
class ZebraDetectionResult:
    boxes: List[Tuple[int, int, int, int]]
    mask: np.ndarray | None
    polygons: List[List[Tuple[int, int]]] = field(default_factory=list)


class ZebraCrossingDetector:
    """Detect zebra crossing regions using classical computer vision."""

    def __init__(
        self,
        canny_thresholds: Tuple[int, int] = ZEBRA_CANNY_THRESHOLDS,
        min_area: int = ZEBRA_MIN_AREA,
        min_aspect: float = ZEBRA_MIN_ASPECT,
        max_aspect: float = ZEBRA_MAX_ASPECT,
    ) -> None:
        self.canny_thresholds = canny_thresholds
        self.min_area = min_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

    def detect(self, frame: np.ndarray) -> ZebraDetectionResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_thresholds[0], self.canny_thresholds[1])

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray, dtype=np.uint8)
        boxes: List[Tuple[int, int, int, int]] = []
        polygons: List[List[Tuple[int, int]]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue
            aspect = w / float(h)
            if not (self.min_aspect <= aspect <= self.max_aspect):
                continue

            polygon = self._contour_to_polygon(cnt)
            if not polygon:
                polygon = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            polygon_array = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
            bx, by, bw, bh = cv2.boundingRect(polygon_array)
            boxes.append((bx, by, bx + bw, by + bh))
            polygons.append(polygon)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

        if not boxes:
            mask = None

        return ZebraDetectionResult(boxes=boxes, mask=mask, polygons=polygons)

    @staticmethod
    def _contour_to_polygon(cnt: np.ndarray) -> List[Tuple[int, int]]:
        perimeter = cv2.arcLength(cnt, True)
        epsilon = max(2.0, 0.02 * perimeter)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if approx is None or len(approx) < 3:
            return []
        return [(int(point[0][0]), int(point[0][1])) for point in approx]
