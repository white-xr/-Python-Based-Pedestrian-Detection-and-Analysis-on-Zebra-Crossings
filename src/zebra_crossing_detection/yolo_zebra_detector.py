"""YOLO-based zebra crossing detector with polygon refinement."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from src.config import (
    YOLO_DEVICE,
    ZEBRA_IMAGE_SIZE,
    ZEBRA_CONF_THRESHOLD,
    ZEBRA_MODEL_PATH,
    ZEBRA_REFINEMENT_ENABLE,
    ZEBRA_REFINEMENT_MIN_PIXELS,
    ZEBRA_WHITE_SAT_MAX,
    ZEBRA_WHITE_VALUE_MIN,
)
from src.zebra_crossing_detection.zebra_detector import ZebraCrossingDetector, ZebraDetectionResult


class YOLOZebraCrossingDetector:
    """Detect zebra crossings using YOLO and refine them into tighter polygons."""

    def __init__(
        self,
        model_path: str = ZEBRA_MODEL_PATH,
        device: str = YOLO_DEVICE,
        conf_threshold: float = ZEBRA_CONF_THRESHOLD,
    ) -> None:
        self._fallback_detector = ZebraCrossingDetector()
        try:
            self.model = YOLO(model_path).to(device)
            try:
                self.model.fuse()
            except Exception:
                pass
            self.conf_threshold = conf_threshold
            self._zebra_class_ids = self._resolve_zebra_class_ids()
            print(f"✅ YOLO 斑马线检测模型加载成功: {model_path}")
        except Exception as exc:
            print(f"❌ YOLO 斑马线检测模型加载失败: {exc}")
            print(f"   请确保模型文件存在: {model_path}")
            raise

    def _resolve_zebra_class_ids(self) -> set[int]:
        names = getattr(self.model, "names", {}) or {}
        zebra_ids: set[int] = set()
        for class_id, class_name in names.items():
            normalized = str(class_name).strip().lower()
            if "zebra" in normalized or "crosswalk" in normalized:
                zebra_ids.add(int(class_id))
        if not zebra_ids and 0 in names:
            zebra_ids.add(0)
        return zebra_ids

    def detect(self, frame: np.ndarray) -> ZebraDetectionResult:
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            imgsz=ZEBRA_IMAGE_SIZE,
            verbose=False,
        )[0]

        boxes: List[Tuple[int, int, int, int]] = []
        polygons: List[List[Tuple[int, int]]] = []
        mask = None

        if results.boxes is None or len(results.boxes) == 0:
            # Use a classical stripe detector as a fallback when YOLO misses this frame.
            return self._fallback_detector.detect(frame)

        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for index, box in enumerate(results.boxes):
            class_id = int(box.cls)
            if self._zebra_class_ids and class_id not in self._zebra_class_ids:
                continue
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            polygon = None
            if ZEBRA_REFINEMENT_ENABLE:
                polygon = self._refine_polygon(frame, (x1, y1, x2, y2))
            if polygon is None and results.masks is not None:
                polygon = self._polygon_from_segmentation(results, index, w, h)
            if polygon is None:
                polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

            contour = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
            bx, by, bw, bh = cv2.boundingRect(contour)
            boxes.append((bx, by, bx + bw, by + bh))
            polygons.append([(int(px), int(py)) for px, py in contour.reshape(-1, 2)])
            cv2.fillPoly(mask, [contour], 255)

        if not boxes:
            return self._fallback_detector.detect(frame)
        return ZebraDetectionResult(boxes=boxes, mask=mask, polygons=polygons)

    def _refine_polygon(
        self,
        frame: np.ndarray,
        rect: Tuple[int, int, int, int],
    ) -> List[Tuple[int, int]] | None:
        x1, y1, x2, y2 = rect
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or min(roi.shape[:2]) < 24:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, ZEBRA_WHITE_VALUE_MIN], dtype=np.uint8),
            np.array([180, ZEBRA_WHITE_SAT_MAX, 255], dtype=np.uint8),
        )
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        stripe_mask = cv2.bitwise_and(white_mask, otsu_mask)

        close_size = max(3, ((min(roi.shape[:2]) // 12) | 1))
        open_size = max(3, ((min(roi.shape[:2]) // 18) | 1))
        stripe_mask = cv2.morphologyEx(
            stripe_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size)),
        )
        stripe_mask = cv2.morphologyEx(
            stripe_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (open_size, open_size)),
        )

        filtered = self._filter_components(stripe_mask)
        non_zero = cv2.findNonZero(filtered)
        if non_zero is None or len(non_zero) < ZEBRA_REFINEMENT_MIN_PIXELS:
            return None

        rect_rotated = cv2.minAreaRect(non_zero)
        box_points = cv2.boxPoints(rect_rotated)
        polygon = np.round(box_points).astype(np.int32)

        area = abs(cv2.contourArea(polygon))
        roi_area = max(1, (x2 - x1) * (y2 - y1))
        if area < roi_area * 0.12:
            return None

        polygon[:, 0] += x1
        polygon[:, 1] += y1
        return [(int(px), int(py)) for px, py in polygon]

    def _filter_components(self, mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered = np.zeros_like(mask)
        min_area = max(ZEBRA_REFINEMENT_MIN_PIXELS // 4, int(mask.shape[0] * mask.shape[1] * 0.003))
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            filtered[labels == label] = 255
        return filtered

    def _polygon_from_segmentation(
        self,
        results,
        index: int,
        width: int,
        height: int,
    ) -> List[Tuple[int, int]] | None:
        try:
            seg_mask = results.masks.data[index].cpu().numpy()
        except Exception:
            return None

        seg_mask = cv2.resize(seg_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        seg_mask = (seg_mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return [(int(point[0][0]), int(point[0][1])) for point in approx]
