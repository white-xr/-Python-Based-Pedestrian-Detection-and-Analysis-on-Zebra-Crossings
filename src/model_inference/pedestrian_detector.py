"""Pedestrian detection module powered by YOLOv8."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from ultralytics import YOLO

from src.config import (
    PEDESTRIAN_CONF_THRESHOLD,
    PEDESTRIAN_IOU_THRESHOLD,
    TRACKER_BACKEND,
    TRACK_LOW_THRESH,
    YOLO_DEVICE,
    YOLO_IMAGE_SIZE,
    YOLO_MODEL_SIZE,
)


Detection = Dict[str, object]
BBox = tuple[float, float, float, float]
Rect = tuple[int, int, int, int]


@dataclass
class TrackDetection:
    track_id: int
    bbox: BBox
    score: float


class PedestrianDetector:
    """Wrapper around YOLOv8 for pedestrian detection."""

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        weights_path: str | None = None,
        device: str = YOLO_DEVICE,
        conf_threshold: float = PEDESTRIAN_CONF_THRESHOLD,
        iou_threshold: float = PEDESTRIAN_IOU_THRESHOLD,
    ) -> None:
        if weights_path is None:
            weights_path = f"yolov8{YOLO_MODEL_SIZE}.pt"
        self.model = YOLO(weights_path).to(device)
        try:
            self.model.fuse()
        except Exception:
            pass
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._tracker_backend = TRACKER_BACKEND.lower()
        self._bytetrack_enabled = self._tracker_backend == "bytetrack"
        self._bytetrack_available = self._bytetrack_enabled
        self._tracker_config_path = Path(__file__).with_name("bytetrack.yaml")
        self._track_fail_reason = ""
        self._last_track_roi: Rect | None = None

    @property
    def bytetrack_available(self) -> bool:
        return self._bytetrack_enabled and self._bytetrack_available

    @property
    def track_fail_reason(self) -> str:
        return self._track_fail_reason

    def track(
        self,
        frame: np.ndarray,
        roi: Rect | None = None,
        image_size: int | None = None,
    ) -> List[TrackDetection]:
        """Track pedestrians with YOLO + ByteTrack and return stable IDs."""
        if not self._bytetrack_enabled or not self._bytetrack_available:
            return []

        source, offset = self._crop_frame(frame, roi)
        if source.size == 0:
            return []
        self._reset_tracker_if_needed(offset, source.shape[:2])

        try:
            results = self.model.track(
                source=source,
                conf=TRACK_LOW_THRESH,
                iou=self.iou_threshold,
                imgsz=image_size or YOLO_IMAGE_SIZE,
                classes=[self.PERSON_CLASS_ID],
                tracker=str(self._tracker_config_path),
                persist=True,
                verbose=False,
                agnostic_nms=False,
            )[0]
        except Exception as exc:
            self._bytetrack_available = False
            self._track_fail_reason = str(exc)
            # `track()` may leave a tracker-aware predictor instance behind; reset it so plain `predict()` can run safely.
            self.model.predictor = None
            print(
                "ByteTrack unavailable, fallback to SimpleTracker. "
                f"Reason: {self._track_fail_reason}"
            )
            return []

        tracks: List[TrackDetection] = []
        for box in results.boxes:
            if int(box.cls) != self.PERSON_CLASS_ID:
                continue
            if box.id is None:
                continue

            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            tracks.append(
                TrackDetection(
                    track_id=int(box.id.item()),
                    bbox=self._offset_bbox((float(x1), float(y1), float(x2), float(y2)), offset),
                    score=float(box.conf.item()) if box.conf is not None else 0.0,
                )
            )
        return tracks

    def detect(
        self,
        frame: np.ndarray,
        roi: Rect | None = None,
        conf_threshold: float | None = None,
        image_size: int | None = None,
    ) -> List[Detection]:
        """Detect pedestrians in a BGR frame."""
        source, offset = self._crop_frame(frame, roi)
        if source.size == 0:
            return []

        results = self.model.predict(
            source=source,
            conf=conf_threshold if conf_threshold is not None else self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=image_size or YOLO_IMAGE_SIZE,
            verbose=False,
            agnostic_nms=False,
        )[0]

        detections: List[Detection] = []
        for box in results.boxes:
            if int(box.cls) != self.PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            detections.append(
                {
                    "bbox": self._offset_bbox((float(x1), float(y1), float(x2), float(y2)), offset),
                    "score": float(box.conf),
                }
            )
        return detections

    def _crop_frame(self, frame: np.ndarray, roi: Rect | None) -> tuple[np.ndarray, tuple[int, int]]:
        if roi is None:
            return frame, (0, 0)
        x1, y1, x2, y2 = roi
        x1 = max(0, min(int(x1), frame.shape[1] - 1))
        y1 = max(0, min(int(y1), frame.shape[0] - 1))
        x2 = max(x1 + 1, min(int(x2), frame.shape[1]))
        y2 = max(y1 + 1, min(int(y2), frame.shape[0]))
        return frame[y1:y2, x1:x2], (x1, y1)

    def _reset_tracker_if_needed(self, offset: tuple[int, int], shape: tuple[int, int]) -> None:
        roi = (offset[0], offset[1], offset[0] + shape[1], offset[1] + shape[0])
        if self._last_track_roi is None:
            self._last_track_roi = roi
            return

        if self._roi_iou(self._last_track_roi, roi) >= 0.9:
            self._last_track_roi = roi
            return

        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and hasattr(predictor, "trackers"):
            predictor.trackers = None
        self._last_track_roi = roi

    @staticmethod
    def _offset_bbox(bbox: BBox, offset: tuple[int, int]) -> BBox:
        ox, oy = offset
        return bbox[0] + ox, bbox[1] + oy, bbox[2] + ox, bbox[3] + oy

    @staticmethod
    def _roi_iou(box_a: Rect, box_b: Rect) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter_area / float(area_a + area_b - inter_area)
