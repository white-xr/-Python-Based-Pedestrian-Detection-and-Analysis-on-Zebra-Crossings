"""Fuse pedestrian detection, scene zones and warning logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

import cv2
import numpy as np
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

from src.config import (
    ENABLE_PERCEPTION,
    KEEP_ORIGINAL_FRAME,
    PEDESTRIAN_CONF_THRESHOLD,
    PERSON_DETECT_INTERVAL,
    PREDICTION_INTERVAL,
    PRED_DRAW_POINTS,
    PRED_DRAW_STEP,
    PRED_SMOOTH_ENABLE,
    PRED_SMOOTH_WEIGHTS,
    PRED_SMOOTH_WINDOW,
    SCENE_FOCUS_IMAGE_SIZE,
    SCENE_FOCUS_TRACKING,
    SCENE_ZONE_DRAW,
    SHOW_PERFORMANCE_OVERLAY,
    SHOW_PREDICTION_TRAJECTORY,
    TRACK_BOX_KALMAN_Q,
    TRACK_BOX_KALMAN_R,
    TRACKER_BACKEND,
    TRACK_MAX_HISTORY,
    TRACK_MAX_MISSED,
    USE_YOLO_ZEBRA_DETECTOR,
    YOLO_IMAGE_SIZE,
    YOLO_MAX_FRAME_WIDTH,
    ZEBRA_DETECT_INTERVAL,
    ZEBRA_FAST_DETECT_INTERVAL,
    ZEBRA_HOLD_MAX_MISS_ATTEMPTS,
    ZEBRA_OCCLUSION_HOLD_RATIO,
    ZEBRA_RENDER_STYLE,
    ZEBRA_UPDATE_MAX_AREA_RATIO,
    ZEBRA_UPDATE_MIN_AREA_RATIO,
    ZEBRA_UPDATE_MIN_IOU,
    ZEBRA_SMOOTHING_ALPHA,
)
from src.lstm_prediction import TrajectoryPredictor
from src.model_inference import PedestrianDetector, TrackDetection
from src.pedestrian_tracking import SimpleTracker, TrackState, TrackStabilizer
from src.safety_analysis.risk_analyzer import CAUTION, DANGER, SAFE, RiskAnalyzer, TrackRisk
from src.safety_analysis.scene_zones import (
    ZONE_APPROACH,
    ZONE_CONFLICT,
    ZONE_OUTSIDE,
    ZONE_ZEBRA,
    SceneContext,
    SceneZoneManager,
)
from src.zebra_crossing_detection import ZebraCrossingDetector
from src.zebra_crossing_detection.yolo_zebra_detector import YOLOZebraCrossingDetector
from src.zebra_crossing_detection.zebra_detector import ZebraDetectionResult


@dataclass
class PerformanceStats:
    preprocess_ms: float = 0.0
    detect_ms: float = 0.0
    track_ms: float = 0.0
    predict_ms: float = 0.0
    render_ms: float = 0.0
    total_ms: float = 0.0
    fps: float = 0.0


@dataclass
class PerceptionResult:
    original_frame: np.ndarray
    annotated_frame: np.ndarray
    tracks: List[TrackState] = field(default_factory=list)
    zebra_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    person_count: int = 0
    on_zebra_count: int = 0
    status_summary: str = ""
    overall_risk: str = "安全"
    track_risks: List[TrackRisk] = field(default_factory=list)
    vehicle_tracks: List[TrackState] = field(default_factory=list)
    stats: PerformanceStats = field(default_factory=PerformanceStats)
    stable_track_count: int = 0
    high_risk_track_ids: List[int] = field(default_factory=list)
    stability_stats: Dict[str, int] = field(default_factory=dict)
    tracker_backend: str = ""
    scene_source: str = "none"
    zone_counts: Dict[str, int] = field(default_factory=dict)


class PerceptionFusion:
    """High-level perception pipeline for zebra-crossing pedestrian warning."""

    def __init__(
        self,
        pedestrian_detector: PedestrianDetector | None = None,
        zebra_detector: ZebraCrossingDetector | YOLOZebraCrossingDetector | None = None,
        trajectory_predictor: TrajectoryPredictor | None = None,
        risk_analyzer: RiskAnalyzer | None = None,
    ) -> None:
        self.pedestrian_detector = pedestrian_detector or PedestrianDetector()
        if zebra_detector is not None:
            self.zebra_detector = zebra_detector
        elif USE_YOLO_ZEBRA_DETECTOR:
            try:
                self.zebra_detector = YOLOZebraCrossingDetector()
                print("Using YOLO zebra detector")
            except Exception as exc:
                print(f"Falling back to classic zebra detector: {exc}")
                self.zebra_detector = ZebraCrossingDetector()
        else:
            self.zebra_detector = ZebraCrossingDetector()

        self.tracker = SimpleTracker()
        self.track_stabilizer = TrackStabilizer()
        self.scene_zone_manager = SceneZoneManager()
        self.trajectory_predictor = trajectory_predictor or TrajectoryPredictor()
        self.risk_analyzer = risk_analyzer or RiskAnalyzer()

        self.person_detect_interval = PERSON_DETECT_INTERVAL
        self.zebra_detect_interval = ZEBRA_DETECT_INTERVAL
        self.prediction_interval = PREDICTION_INTERVAL
        self._use_bytetrack = TRACKER_BACKEND.lower() == "bytetrack"
        self._tracker_backend = "bytetrack" if self._use_bytetrack else "simple"
        self._fallback_logged = False
        self._bytetrack_state: Dict[int, TrackState] = {}

        self._frame_index = 0
        self._cached_zebra_result: ZebraDetectionResult | None = None
        self._display_zebra_result: ZebraDetectionResult | None = None
        self._zebra_missed_frames = 0
        self._zebra_consecutive_misses = 0
        self._zebra_fast_interval = max(1, int(ZEBRA_FAST_DETECT_INTERVAL))
        self._zebra_hold_max_miss_attempts = max(1, int(ZEBRA_HOLD_MAX_MISS_ATTEMPTS))
        self._zebra_occlusion_hold_ratio = float(np.clip(ZEBRA_OCCLUSION_HOLD_RATIO, 0.0, 0.95))
        self._zebra_update_min_iou = float(np.clip(ZEBRA_UPDATE_MIN_IOU, 0.0, 1.0))
        self._zebra_update_min_area_ratio = max(0.05, float(ZEBRA_UPDATE_MIN_AREA_RATIO))
        self._zebra_update_max_area_ratio = max(
            self._zebra_update_min_area_ratio + 0.01,
            float(ZEBRA_UPDATE_MAX_AREA_RATIO),
        )
        self._last_track_boxes: List[Tuple[float, float, float, float]] = []
        self._zebra_smoothed_polygons: List[np.ndarray] = []
        self._smoothing_alpha = ZEBRA_SMOOTHING_ALPHA
        self._fps_ema = 0.0
        self._project_root = Path(__file__).resolve().parents[2]
        self._label_font = self._load_label_font(size=18)

    def process_frame(self, frame: np.ndarray, frame_interval: float = 1 / 30.0) -> PerceptionResult:
        total_start = perf_counter()
        self._frame_index += 1
        stats = PerformanceStats()

        preprocess_start = perf_counter()
        processed_frame, _ = self._prepare_frame(frame)
        original_frame = frame.copy() if KEEP_ORIGINAL_FRAME else processed_frame.copy()
        stats.preprocess_ms = (perf_counter() - preprocess_start) * 1000

        if not ENABLE_PERCEPTION:
            annotated = original_frame.copy()
            stats.total_ms = (perf_counter() - total_start) * 1000
            stats.fps = self._update_fps(stats.total_ms)
            return PerceptionResult(
                original_frame=original_frame,
                annotated_frame=annotated,
                status_summary="感知模块已关闭",
                stats=stats,
                tracker_backend=self._tracker_backend,
            )

        detect_start = perf_counter()
        zebra_result = self._update_zebra_state(processed_frame)
        display_zebra_result = self._display_zebra_result or zebra_result
        scene_context = self.scene_zone_manager.build(processed_frame.shape, display_zebra_result)
        stats.detect_ms = (perf_counter() - detect_start) * 1000

        track_start = perf_counter()
        raw_tracks = self._run_tracking(processed_frame, frame_interval, scene_context)
        stable_tracks, stabilizer_stats = self.track_stabilizer.update(raw_tracks, frame_interval)
        self._last_track_boxes = [
            tuple(map(float, (track.smoothed_bbox or track.bbox)))
            for track in stable_tracks
        ]
        stats.track_ms = (perf_counter() - track_start) * 1000

        predict_start = perf_counter()
        zone_counts = {
            ZONE_OUTSIDE: 0,
            ZONE_APPROACH: 0,
            ZONE_ZEBRA: 0,
            ZONE_CONFLICT: 0,
        }
        prediction_enabled = self._frame_index % self.prediction_interval == 0
        for track in stable_tracks:
            draw_box = track.smoothed_bbox or track.bbox
            foot_point = self._foot_point(draw_box)
            track.foot_point = foot_point
            track.zone_state = scene_context.classify_point(foot_point)
            track.on_zebra = track.zone_state == ZONE_ZEBRA
            zone_counts[track.zone_state] = zone_counts.get(track.zone_state, 0) + 1
            if prediction_enabled:
                raw_preds = self.trajectory_predictor.predict(track.history)
                track.predicted_positions = self._smooth_predictions(track, raw_preds)
        stats.predict_ms = (perf_counter() - predict_start) * 1000

        overall_risk, track_risks = self.risk_analyzer.evaluate(stable_tracks, display_zebra_result, scene_context)
        high_risk_track_ids = [risk.track_id for risk in track_risks if risk.level in (CAUTION, DANGER)]
        on_zebra_count = zone_counts.get(ZONE_ZEBRA, 0)
        summary = (
            f"行人 {len(stable_tracks)} | 接近区 {zone_counts.get(ZONE_APPROACH, 0)} | "
            f"斑马线内 {on_zebra_count} | 风险 {overall_risk}"
        )

        render_start = perf_counter()
        annotated = processed_frame.copy()
        self._draw_zebra(annotated, display_zebra_result)
        if SCENE_ZONE_DRAW:
            self._draw_scene_zones(annotated, scene_context)
        self._draw_tracks(annotated, stable_tracks, track_risks)
        stats.render_ms = (perf_counter() - render_start) * 1000

        stats.total_ms = (perf_counter() - total_start) * 1000
        stats.fps = self._update_fps(stats.total_ms)
        if SHOW_PERFORMANCE_OVERLAY:
            self._draw_perf_overlay(annotated, stats)

        return PerceptionResult(
            original_frame=original_frame,
            annotated_frame=annotated,
            tracks=stable_tracks,
            zebra_boxes=display_zebra_result.boxes,
            person_count=len(stable_tracks),
            on_zebra_count=on_zebra_count,
            status_summary=summary,
            overall_risk=overall_risk,
            track_risks=track_risks,
            vehicle_tracks=[],
            stats=stats,
            stable_track_count=stabilizer_stats.visible_tracks,
            high_risk_track_ids=high_risk_track_ids,
            stability_stats={
                "active_tracks": stabilizer_stats.active_tracks,
                "held_tracks": stabilizer_stats.held_tracks,
                "visible_tracks": stabilizer_stats.visible_tracks,
            },
            tracker_backend=self._tracker_backend,
            scene_source=scene_context.source,
            zone_counts=zone_counts,
        )

    def _run_tracking(
        self,
        frame: np.ndarray,
        frame_interval: float,
        scene_context: SceneContext,
    ) -> List[TrackState]:
        focus_roi = scene_context.focus_roi if SCENE_FOCUS_TRACKING else None
        focus_image_size = self._select_focus_image_size(frame.shape, scene_context)

        if self._use_bytetrack:
            track_detections = self.pedestrian_detector.track(
                frame,
                roi=focus_roi,
                image_size=focus_image_size,
            )
            if self.pedestrian_detector.bytetrack_available:
                self._tracker_backend = "bytetrack"
                return self._update_bytetrack_states(track_detections, frame_interval)

            if not self._fallback_logged:
                reason = self.pedestrian_detector.track_fail_reason or "unknown"
                print(f"[TrackerFallback] ByteTrack unavailable, switched to SimpleTracker. reason={reason}")
                self._fallback_logged = True

        self._tracker_backend = "simple"
        should_detect_people = (
            self._frame_index == 1 or self._frame_index % self.person_detect_interval == 0
        )
        detections = None
        if should_detect_people:
            detections = self.pedestrian_detector.detect(frame)
            if focus_roi is not None:
                roi_detections = self.pedestrian_detector.detect(
                    frame,
                    roi=focus_roi,
                    conf_threshold=max(0.18, PEDESTRIAN_CONF_THRESHOLD * 0.8),
                    image_size=focus_image_size,
                )
                detections = self._merge_detections(detections, roi_detections)
        return self.tracker.update(detections, frame_interval=frame_interval)

    def _update_bytetrack_states(
        self,
        track_detections: List[TrackDetection],
        frame_interval: float,
    ) -> List[TrackState]:
        active_tracks: List[TrackState] = []
        seen_ids = set()

        for det in track_detections:
            seen_ids.add(det.track_id)
            track = self._bytetrack_state.get(det.track_id)
            if track is None:
                track = TrackState(track_id=det.track_id, bbox=det.bbox, score=det.score)
            track.bbox = det.bbox
            track.score = det.score
            track.missed_frames = 0
            track.is_held = False
            self._append_history(track, det.bbox, frame_interval)
            active_tracks.append(track)
            self._bytetrack_state[det.track_id] = track

        stale_ids: List[int] = []
        for track_id, track in self._bytetrack_state.items():
            if track_id in seen_ids:
                continue
            track.missed_frames += 1
            if track.missed_frames > TRACK_MAX_MISSED:
                stale_ids.append(track_id)
        for track_id in stale_ids:
            self._bytetrack_state.pop(track_id, None)

        return active_tracks

    def _append_history(self, track: TrackState, bbox: Tuple[float, float, float, float], frame_interval: float) -> None:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = bbox[3]

        prev_point = track.history[-1] if track.history else None
        track.history.append((cx, cy))
        if len(track.history) > TRACK_MAX_HISTORY:
            track.history.pop(0)

        if prev_point is None or frame_interval <= 0:
            track.velocity = (0.0, 0.0)
            track.speed = 0.0
            return

        vx = (cx - prev_point[0]) / frame_interval
        vy = (cy - prev_point[1]) / frame_interval
        track.velocity = (vx, vy)
        track.speed = float(np.hypot(vx, vy))

    def _smooth_predictions(
        self,
        track: TrackState,
        raw_preds: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        if not raw_preds:
            return []
        if not PRED_SMOOTH_ENABLE:
            return raw_preds

        kalman_preds = self._kalman_filter_predictions(track, raw_preds)
        if PRED_SMOOTH_WINDOW != 3 or len(kalman_preds) < 3:
            return kalman_preds
        return self._weighted_average(kalman_preds, PRED_SMOOTH_WEIGHTS)

    def _kalman_filter_predictions(
        self,
        track: TrackState,
        raw_preds: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        if not raw_preds:
            return []

        start_x, start_y = track.history[-1] if track.history else raw_preds[0]
        vx, vy = track.velocity

        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * (TRACK_BOX_KALMAN_Q * 1.5)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * (TRACK_BOX_KALMAN_R * 2.0)
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.array([[start_x], [start_y], [vx], [vy]], dtype=np.float32)

        smoothed: List[Tuple[float, float]] = []
        for px, py in raw_preds:
            kf.predict()
            corrected = kf.correct(np.array([[px], [py]], dtype=np.float32))
            smoothed.append((float(corrected[0, 0]), float(corrected[1, 0])))
        return smoothed

    @staticmethod
    def _weighted_average(
        points: List[Tuple[float, float]],
        weights: Tuple[float, float, float],
    ) -> List[Tuple[float, float]]:
        w0, w1, w2 = weights
        smoothed = [points[0]]
        for idx in range(1, len(points) - 1):
            x = w0 * points[idx - 1][0] + w1 * points[idx][0] + w2 * points[idx + 1][0]
            y = w0 * points[idx - 1][1] + w1 * points[idx][1] + w2 * points[idx + 1][1]
            smoothed.append((x, y))
        smoothed.append(points[-1])
        return smoothed

    def _merge_detections(
        self,
        detections: List[Dict[str, object]],
        roi_detections: List[Dict[str, object]],
        iou_threshold: float = 0.45,
    ) -> List[Dict[str, object]]:
        merged = list(detections)
        for roi_det in roi_detections:
            roi_bbox = roi_det["bbox"]
            duplicate_index = None
            for index, det in enumerate(merged):
                if self._iou(det["bbox"], roi_bbox) >= iou_threshold:
                    duplicate_index = index
                    break
            if duplicate_index is None:
                merged.append(roi_det)
                continue
            if float(roi_det["score"]) > float(merged[duplicate_index]["score"]):
                merged[duplicate_index] = roi_det
        return merged

    def _prepare_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        if w <= YOLO_MAX_FRAME_WIDTH:
            return frame, 1.0

        scale = YOLO_MAX_FRAME_WIDTH / float(w)
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _select_focus_image_size(
        self,
        frame_shape: tuple[int, int, int],
        scene_context: SceneContext,
    ) -> int:
        reference_rect = None
        if scene_context.zebra_zone is not None:
            reference_rect = scene_context.zebra_zone.bounding_rect()
        if reference_rect is None:
            reference_rect = scene_context.focus_roi
        if reference_rect is None:
            return YOLO_IMAGE_SIZE

        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = reference_rect
        roi_area = max(1, (x2 - x1) * (y2 - y1))
        frame_area = max(1, frame_w * frame_h)
        area_ratio = min(1.0, roi_area / float(frame_area))

        detail_boost = float(np.clip((0.65 - area_ratio) / 0.45, 0.0, 1.0))
        target_size = int(round(YOLO_IMAGE_SIZE + detail_boost * (SCENE_FOCUS_IMAGE_SIZE - YOLO_IMAGE_SIZE)))
        target_size = max(YOLO_IMAGE_SIZE, min(SCENE_FOCUS_IMAGE_SIZE, target_size))
        return max(32, int(round(target_size / 32.0)) * 32)

    def _update_fps(self, total_ms: float) -> float:
        instant_fps = 1000.0 / total_ms if total_ms > 0 else 0.0
        if self._fps_ema == 0.0:
            self._fps_ema = instant_fps
        else:
            self._fps_ema = 0.85 * self._fps_ema + 0.15 * instant_fps
        return self._fps_ema

    def _update_zebra_state(self, frame: np.ndarray) -> ZebraDetectionResult:
        detect_interval = self._zebra_fast_interval
        if self._has_zebra(self._display_zebra_result) and self._zebra_consecutive_misses == 0:
            detect_interval = max(1, int(self.zebra_detect_interval))

        should_detect = (
            self._frame_index == 1
            or self._cached_zebra_result is None
            or self._frame_index % detect_interval == 0
        )
        if not should_detect:
            return self._cached_zebra_result or self._display_zebra_result or self._empty_zebra_result()

        try:
            detected = self.zebra_detector.detect(frame)
        except Exception as exc:
            print(f"[Zebra] detection error: {exc}")
            detected = None
        detected = detected or self._empty_zebra_result()

        occlusion_ratio = self._estimate_zebra_occlusion_ratio(self._display_zebra_result, self._last_track_boxes)
        return self._merge_zebra_detection(detected, occlusion_ratio)

    def _merge_zebra_detection(
        self,
        detected: ZebraDetectionResult,
        occlusion_ratio: float,
    ) -> ZebraDetectionResult:
        if detected.boxes:
            if self._is_outlier_zebra(detected):
                # Keep last stable zebra region when the current update is geometrically implausible.
                self._zebra_consecutive_misses += 1
                return self._display_zebra_result or self._cached_zebra_result or detected

            self._zebra_consecutive_misses = 0
            self._zebra_missed_frames = 0
            self._cached_zebra_result = detected
            self._display_zebra_result = detected
            self._update_smoothed_zebra(detected)
            return detected

        hold_condition = (
            self._has_zebra(self._display_zebra_result)
            and (
                occlusion_ratio >= self._zebra_occlusion_hold_ratio
                or self._zebra_consecutive_misses < self._zebra_hold_max_miss_attempts
            )
        )
        self._zebra_consecutive_misses += 1
        self._zebra_missed_frames += 1
        if hold_condition:
            return self._display_zebra_result or self._cached_zebra_result or self._empty_zebra_result()

        empty = self._empty_zebra_result()
        self._cached_zebra_result = empty
        self._display_zebra_result = empty
        self._zebra_smoothed_polygons = []
        return empty

    @staticmethod
    def _empty_zebra_result() -> ZebraDetectionResult:
        return ZebraDetectionResult(boxes=[], mask=None, polygons=[])

    @staticmethod
    def _has_zebra(zebra_result: ZebraDetectionResult | None) -> bool:
        return zebra_result is not None and bool(zebra_result.boxes)

    def _estimate_zebra_occlusion_ratio(
        self,
        zebra_result: ZebraDetectionResult | None,
        track_boxes: List[Tuple[float, float, float, float]],
    ) -> float:
        if not self._has_zebra(zebra_result) or not track_boxes:
            return 0.0

        zebra_boxes = zebra_result.boxes
        zebra_area = sum(max(1.0, float((x2 - x1) * (y2 - y1))) for x1, y1, x2, y2 in zebra_boxes)
        if zebra_area <= 1.0:
            return 0.0

        covered = 0.0
        for tx1, ty1, tx2, ty2 in track_boxes:
            for zx1, zy1, zx2, zy2 in zebra_boxes:
                inter_x1 = max(float(tx1), float(zx1))
                inter_y1 = max(float(ty1), float(zy1))
                inter_x2 = min(float(tx2), float(zx2))
                inter_y2 = min(float(ty2), float(zy2))
                if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                    continue
                covered += (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        return float(np.clip(covered / zebra_area, 0.0, 1.0))

    def _is_outlier_zebra(self, detected: ZebraDetectionResult) -> bool:
        if not self._has_zebra(self._display_zebra_result):
            return False

        prev_box = self._merge_boxes(self._display_zebra_result.boxes)
        curr_box = self._merge_boxes(detected.boxes)
        if prev_box is None or curr_box is None:
            return False

        iou = self._iou(prev_box, curr_box)
        prev_area = max(1.0, self._box_area(prev_box))
        curr_area = max(1.0, self._box_area(curr_box))
        area_ratio = curr_area / prev_area

        return iou < self._zebra_update_min_iou and (
            area_ratio < self._zebra_update_min_area_ratio
            or area_ratio > self._zebra_update_max_area_ratio
        )

    @staticmethod
    def _merge_boxes(boxes: List[Tuple[int, int, int, int]]) -> Tuple[float, float, float, float] | None:
        if not boxes:
            return None
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)
        return float(x1), float(y1), float(x2), float(y2)

    @staticmethod
    def _box_area(box: Tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = box
        return max(1.0, (x2 - x1) * (y2 - y1))

    def _update_smoothed_zebra(self, zebra_result: ZebraDetectionResult) -> None:
        polygons = zebra_result.polygons or [
            [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            for x1, y1, x2, y2 in zebra_result.boxes
        ]
        smoothed: List[np.ndarray] = []
        for index, polygon in enumerate(polygons):
            points = np.array(polygon, dtype=np.float32)
            if index < len(self._zebra_smoothed_polygons) and len(self._zebra_smoothed_polygons[index]) == len(points):
                prev = self._zebra_smoothed_polygons[index]
                points = self._smoothing_alpha * prev + (1 - self._smoothing_alpha) * points
            smoothed.append(points)
        self._zebra_smoothed_polygons = smoothed

    @staticmethod
    def _foot_point(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, _, x2, y2 = bbox
        return (x1 + x2) / 2.0, y2

    def _draw_zebra(self, frame: np.ndarray, zebra_result: ZebraDetectionResult) -> None:
        if not zebra_result.boxes:
            return

        overlay = frame.copy()
        points_list = self._zebra_smoothed_polygons or [
            np.array(polygon, dtype=np.float32)
            for polygon in (zebra_result.polygons or [
                [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                for x1, y1, x2, y2 in zebra_result.boxes
            ])
        ]

        for points in points_list:
            contour = points.astype(np.int32).reshape(-1, 1, 2)
            color = (0, 220, 120)
            cv2.fillPoly(overlay, [contour], color)
            cv2.polylines(frame, [contour], True, color, 2, cv2.LINE_AA)
            if ZEBRA_RENDER_STYLE == "fancy":
                self._draw_zebra_stripes(frame, contour)

        cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)

    def _draw_scene_zones(self, frame: np.ndarray, scene_context: SceneContext) -> None:
        overlay = frame.copy()
        for polygon in [scene_context.conflict_zone, scene_context.approach_zone]:
            if polygon is None:
                continue
            contour = polygon.contour()
            if contour is None:
                continue
            cv2.fillPoly(overlay, [contour], polygon.color)
            cv2.polylines(frame, [contour], True, polygon.color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

        if scene_context.focus_roi is not None:
            x1, y1, x2, y2 = scene_context.focus_roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (212, 176, 82), 1, cv2.LINE_AA)

    def _draw_zebra_stripes(self, frame: np.ndarray, contour: np.ndarray) -> None:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        y_min = int(contour[:, 0, 1].min())
        y_max = int(contour[:, 0, 1].max())
        for y in range(y_min, y_max, 16):
            stripe = np.zeros_like(mask)
            cv2.rectangle(stripe, (0, y), (frame.shape[1], min(y + 8, y_max)), 255, -1)
            stripe = cv2.bitwise_and(stripe, mask)
            frame[stripe > 0] = (180, 255, 220)

    def _draw_tracks(self, frame: np.ndarray, tracks: List[TrackState], track_risks: List[TrackRisk]) -> None:
        risk_lookup = {risk.track_id: risk for risk in track_risks}
        label_items: List[Tuple[str, int, int, Tuple[int, int, int]]] = []
        for track in tracks:
            draw_box = track.smoothed_bbox or track.bbox
            x1, y1, x2, y2 = map(int, draw_box)
            risk = risk_lookup.get(track.track_id)
            level = risk.level if risk else SAFE
            color = self._risk_color(level, track.zone_state)
            if track.is_held:
                color = (150, 176, 228)
            self._draw_clean_box(frame, (x1, y1, x2, y2), color)
            foot = (int((x1 + x2) / 2), y2)
            cv2.circle(frame, foot, 4, color, -1)
            label = f"{self._zone_label(track.zone_state)} ID {track.track_id}"
            anchor_x, anchor_y = track.label_anchor or (float(x1), float(y1 - 12))
            label_items.append((label, int(anchor_x), int(anchor_y), color))
            self._draw_history(frame, track.history, color)
            if SHOW_PREDICTION_TRAJECTORY and track.predicted_positions:
                self._draw_prediction(frame, track.predicted_positions, color)
        self._draw_labels(frame, label_items)

    def _draw_history(self, frame: np.ndarray, history: List[Tuple[float, float]], color: Tuple[int, int, int]) -> None:
        if len(history) < 2:
            return
        points = np.array(history[-10:], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [points], False, color, 1, cv2.LINE_AA)

    def _draw_prediction(
        self,
        frame: np.ndarray,
        predicted_positions: List[Tuple[float, float]],
        color: Tuple[int, int, int],
    ) -> None:
        if not predicted_positions:
            return
        points = np.array(predicted_positions[:PRED_DRAW_POINTS], dtype=np.int32)
        if len(points) > 2 and PRED_DRAW_STEP > 1:
            points = points[::PRED_DRAW_STEP]
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(frame, tuple(start), tuple(end), color, 1, cv2.LINE_AA)
            cv2.circle(frame, tuple(end), 2, color, -1)

    def _draw_clean_box(self, frame: np.ndarray, bbox: tuple[int, int, int, int], color: Tuple[int, int, int]) -> None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        corner = 12
        thickness = 3
        cv2.line(frame, (x1, y1), (x1 + corner, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2 - corner, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + corner), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1 + corner, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - corner), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2 - corner, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - corner), color, thickness, cv2.LINE_AA)

    def _candidate_font_paths(self) -> List[Path]:
        return [
            self._project_root / "src" / "fonts" / "SimHei.ttf",
            self._project_root / "src" / "gui" / "SimHei.ttf",
            self._project_root / "src" / "detect" / "SimHei.ttf",
            self._project_root / "SimHei.ttf",
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/msyhbd.ttc"),
        ]

    def _load_label_font(self, size: int):
        if ImageFont is None:
            print("[Overlay] Pillow unavailable, fallback to cv2.putText.")
            return None

        for font_path in self._candidate_font_paths():
            if not font_path.exists():
                continue
            try:
                return ImageFont.truetype(str(font_path), size=size)
            except OSError:
                continue

        print("[Overlay] Chinese font not found, fallback to cv2.putText.")
        return None

    def _draw_labels(
        self,
        frame: np.ndarray,
        label_items: List[Tuple[str, int, int, Tuple[int, int, int]]],
    ) -> None:
        if not label_items:
            return

        if self._label_font is None or Image is None or ImageDraw is None:
            for text, x, y, color in label_items:
                self._draw_label_cv2(frame, text, x, y, color)
            return

        rgba_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        draw = ImageDraw.Draw(rgba_image, "RGBA")

        for text, x, y, color in label_items:
            self._draw_label_pil(draw, frame.shape[1], text, x, y, color)

        rgb_image = np.array(rgba_image.convert("RGB"))
        frame[:] = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    def _draw_label_pil(
        self,
        draw: "ImageDraw.ImageDraw",
        frame_width: int,
        text: str,
        x: int,
        y: int,
        color: Tuple[int, int, int],
    ) -> None:
        if self._label_font is None:
            return

        padding = 6
        try:
            bbox = self._label_font.getbbox(text)
            text_w = int(bbox[2] - bbox[0])
            text_h = int(bbox[3] - bbox[1])
        except Exception:
            text_w, text_h = self._label_font.getsize(text)

        x = max(8, min(x, frame_width - text_w - padding * 2 - 8))
        top = max(8, y - text_h - padding * 2 - 4)
        right = min(frame_width - 8, x + text_w + padding * 2)
        bottom = top + text_h + padding * 2

        border_rgb = (int(color[2]), int(color[1]), int(color[0]), 255)
        draw.rounded_rectangle((x, top, right, bottom), radius=5, fill=(18, 24, 32, 188), outline=border_rgb, width=1)
        draw.text((x + padding, top + padding - 1), text, font=self._label_font, fill=(245, 248, 252, 255))

    def _draw_label_cv2(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: Tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        thickness = 1
        padding = 6
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)

        x = max(8, min(x, frame.shape[1] - text_w - padding * 2 - 8))
        top = max(8, y - text_h - padding * 2 - 4)
        right = min(frame.shape[1] - 8, x + text_w + padding * 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, top), (right, top + text_h + padding * 2), (18, 24, 32), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (x, top), (right, top + text_h + padding * 2), color, 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            text,
            (x + padding, top + text_h + 1),
            font,
            scale,
            (245, 248, 252),
            thickness,
            cv2.LINE_AA,
        )

    def _draw_perf_overlay(self, frame: np.ndarray, stats: PerformanceStats) -> None:
        lines = [
            f"FPS {stats.fps:.1f}",
            f"total {stats.total_ms:.1f}ms",
            f"det {stats.detect_ms:.1f} | trk {stats.track_ms:.1f} | rnd {stats.render_ms:.1f}",
        ]
        x = 12
        y = 22
        for line in lines:
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 255, 40), 2, cv2.LINE_AA)
            y += 22

    @staticmethod
    def _risk_color(level: str, zone_state: str) -> Tuple[int, int, int]:
        if level == DANGER:
            return (64, 84, 255)
        if level == CAUTION:
            return (0, 196, 255)
        if zone_state == ZONE_APPROACH:
            return (80, 186, 255)
        return (58, 214, 138) if zone_state == ZONE_ZEBRA else (232, 197, 67)

    @staticmethod
    def _zone_label(zone_state: str) -> str:
        mapping = {
            ZONE_OUTSIDE: "观察",
            ZONE_APPROACH: "接近",
            ZONE_ZEBRA: "斑马线",
            ZONE_CONFLICT: "冲突",
        }
        return mapping.get(zone_state, "观察")

    @staticmethod
    def _iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
        return inter_area / (area_a + area_b - inter_area)
