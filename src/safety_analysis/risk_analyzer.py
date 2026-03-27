"""Stateful traffic safety risk analysis for zebra-crossing pedestrians."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.config import (
    PEDESTRIAN_SPEED_THRESHOLD,
    RISK_APPROACH_SPEED_FACTOR,
    RISK_TTC_CAUTION,
    RISK_TTC_DANGER,
    RISK_ZEBRA_DWELL_FRAMES,
)
from src.pedestrian_tracking import TrackState
from src.safety_analysis.scene_zones import (
    ZONE_APPROACH,
    ZONE_CONFLICT,
    ZONE_OUTSIDE,
    ZONE_ZEBRA,
    SceneContext,
)
from src.zebra_crossing_detection.zebra_detector import ZebraDetectionResult


SAFE = "安全"
CAUTION = "注意"
DANGER = "危险"
RISK_PRIORITY = {SAFE: 0, CAUTION: 1, DANGER: 2}


@dataclass
class TrackRisk:
    track_id: int
    level: str
    reasons: List[str]


@dataclass
class _TrackLifecycle:
    zone_state: str = ZONE_OUTSIDE
    dwell_frames: int = 0


class RiskAnalyzer:
    """Heuristic risk analyzer combining scene zones, motion and predictions."""

    def __init__(
        self,
        speed_threshold: float = PEDESTRIAN_SPEED_THRESHOLD,
        prediction_steps: int = 10,
    ) -> None:
        self.speed_threshold = speed_threshold
        self.prediction_steps = prediction_steps
        self._track_states: Dict[int, _TrackLifecycle] = {}

    def evaluate(
        self,
        tracks: List[TrackState],
        zebra_result: ZebraDetectionResult,
        scene_context: SceneContext | None = None,
        vehicle_tracks: List[TrackState] | None = None,
    ) -> Tuple[str, List[TrackRisk]]:
        track_risks: List[TrackRisk] = []
        overall_level = SAFE
        active_ids = set()

        for track in tracks:
            active_ids.add(track.track_id)
            level, reasons = self._evaluate_track(track, zebra_result, scene_context, vehicle_tracks or [])
            track_risks.append(TrackRisk(track_id=track.track_id, level=level, reasons=reasons))
            if RISK_PRIORITY[level] > RISK_PRIORITY[overall_level]:
                overall_level = level

        stale_ids = [track_id for track_id in self._track_states if track_id not in active_ids]
        for track_id in stale_ids:
            self._track_states.pop(track_id, None)

        track_risks.sort(key=lambda item: (-RISK_PRIORITY[item.level], item.track_id))
        return overall_level, track_risks

    def _evaluate_track(
        self,
        track: TrackState,
        zebra_result: ZebraDetectionResult,
        scene_context: SceneContext | None,
        vehicle_tracks: List[TrackState],
    ) -> Tuple[str, List[str]]:
        lifecycle = self._track_states.get(track.track_id, _TrackLifecycle())
        current_zone = track.zone_state or ZONE_OUTSIDE
        if lifecycle.zone_state == current_zone:
            lifecycle.dwell_frames += 1
        else:
            lifecycle.zone_state = current_zone
            lifecycle.dwell_frames = 1
        self._track_states[track.track_id] = lifecycle

        predicted_zone = self._predicted_zone(track, zebra_result, scene_context)
        reasons: List[str] = []
        level = SAFE
        risk_score = 0.0

        if current_zone == ZONE_CONFLICT:
            level = DANGER
            risk_score = 0.95
            reasons.append("行人进入道路冲突区域")
        elif current_zone == ZONE_ZEBRA:
            level = CAUTION
            risk_score = 0.65
            reasons.append("行人位于斑马线区域")
            if lifecycle.dwell_frames >= RISK_ZEBRA_DWELL_FRAMES:
                level = DANGER
                risk_score = max(risk_score, 0.82)
                reasons.append("行人在斑马线停留时间较长")
        elif current_zone == ZONE_APPROACH:
            if predicted_zone in (ZONE_ZEBRA, ZONE_CONFLICT):
                level = CAUTION
                risk_score = 0.48
                reasons.append("行人正在靠近斑马线")
            elif track.speed > self.speed_threshold * RISK_APPROACH_SPEED_FACTOR:
                level = CAUTION
                risk_score = 0.42
                reasons.append("行人接近斑马线且移动较快")

        if predicted_zone == ZONE_CONFLICT and current_zone in (ZONE_APPROACH, ZONE_ZEBRA):
            level = DANGER
            risk_score = max(risk_score, 0.88)
            reasons.append("预测轨迹将进入道路冲突区域")
        elif predicted_zone == ZONE_ZEBRA and current_zone == ZONE_OUTSIDE:
            level = CAUTION
            risk_score = max(risk_score, 0.45)
            reasons.append("预测轨迹将进入斑马线")

        if track.speed > self.speed_threshold * 1.4 and current_zone != ZONE_OUTSIDE:
            level = DANGER
            risk_score = max(risk_score, 0.90)
            reasons.append("行人移动速度偏高")

        vehicle_level, vehicle_reason = self._evaluate_vehicle_conflict(track, vehicle_tracks)
        if vehicle_level and vehicle_reason:
            if RISK_PRIORITY[vehicle_level] > RISK_PRIORITY[level]:
                level = vehicle_level
            reasons.append(vehicle_reason)

        if not reasons:
            reasons.append("未发现明显风险")

        track.on_zebra = current_zone == ZONE_ZEBRA
        track.risk_score = risk_score
        return level, reasons

    def _predicted_zone(
        self,
        track: TrackState,
        zebra_result: ZebraDetectionResult,
        scene_context: SceneContext | None,
    ) -> str:
        if not track.predicted_positions:
            return ZONE_OUTSIDE
        for point in track.predicted_positions[: self.prediction_steps]:
            if scene_context is not None:
                zone = scene_context.classify_point(point)
                if zone != ZONE_OUTSIDE:
                    return zone
            elif self._point_in_zebra(point, zebra_result):
                return ZONE_ZEBRA
        return ZONE_OUTSIDE

    def _point_in_zebra(self, point: Tuple[float, float], zebra_result: ZebraDetectionResult) -> bool:
        x, y = map(int, point)
        if zebra_result.mask is not None:
            mask = zebra_result.mask
            h, w = mask.shape
            px = np.clip(x, 0, w - 1)
            py = np.clip(y, 0, h - 1)
            if mask[py, px] > 0:
                return True
        for zx1, zy1, zx2, zy2 in zebra_result.boxes:
            if zx1 <= x <= zx2 and zy1 <= y <= zy2:
                return True
        return False

    def _evaluate_vehicle_conflict(
        self,
        person_track: TrackState,
        vehicle_tracks: List[TrackState],
    ) -> Tuple[str | None, str | None]:
        if not vehicle_tracks or not person_track.history:
            return None, None

        px, py = person_track.history[-1]
        best_ttc = None

        for vehicle in vehicle_tracks:
            if not vehicle.history:
                continue
            vx, vy = vehicle.history[-1]
            rel_dist = float(np.hypot(px - vx, py - vy))
            if rel_dist <= 0:
                continue
            ttc = rel_dist / (vehicle.speed + 1e-6)
            if best_ttc is None or ttc < best_ttc:
                best_ttc = ttc

        if best_ttc is None:
            return None, None
        if best_ttc < RISK_TTC_DANGER:
            return DANGER, f"附近车辆接近过快，TTC={best_ttc:.1f}s"
        if best_ttc < RISK_TTC_CAUTION:
            return CAUTION, f"附近车辆正在接近，TTC={best_ttc:.1f}s"
        return None, None
