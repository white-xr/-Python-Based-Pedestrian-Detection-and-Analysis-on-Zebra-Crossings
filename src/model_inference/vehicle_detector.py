"""Vehicle detection module powered by YOLOv8 (COCO vehicle classes)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from ultralytics import YOLO

from src.config import VEHICLE_CONF_THRESHOLD, YOLO_DEVICE


Detection = Dict[str, object]


class VehicleDetector:
    """Wrapper around YOLOv8 for vehicle detection on COCO classes."""

    # COCO vehicle-related class IDs commonly used by YOLOv8
    VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck

    def __init__(
        self,
        weights_path: str = "weights/yolov8n.pt",
        device: str = YOLO_DEVICE,
        conf_threshold: float = VEHICLE_CONF_THRESHOLD,
    ) -> None:
        self.model = YOLO(weights_path).to(device)
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect vehicles in a BGR frame."""
        results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)[0]

        detections: List[Detection] = []
        for box in results.boxes:
            cls = int(box.cls)
            if cls not in self.VEHICLE_CLASS_IDS:
                continue
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            detections.append(
                {
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "score": float(box.conf),
                    "class_id": cls,
                }
            )
        return detections




