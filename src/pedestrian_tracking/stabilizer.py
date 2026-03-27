"""Track-level bbox stabilizer for smoother rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.config import (
    TRACK_BOX_KALMAN_Q,
    TRACK_BOX_KALMAN_R,
    TRACK_DRAW_HOLD_MS,
    TRACK_WH_EMA_ALPHA,
)
from src.pedestrian_tracking.tracker import BBox, TrackState


@dataclass
class StabilizerStats:
    active_tracks: int = 0
    held_tracks: int = 0
    visible_tracks: int = 0


@dataclass
class _TrackFilterState:
    kf: cv2.KalmanFilter
    width: float
    height: float
    label_anchor: Tuple[float, float]
    last_bbox: BBox
    last_track: TrackState
    last_seen_s: float


class TrackStabilizer:
    """Per-ID stabilization with center Kalman and width/height EMA."""

    def __init__(
        self,
        hold_ms: int = TRACK_DRAW_HOLD_MS,
        wh_alpha: float = TRACK_WH_EMA_ALPHA,
        kalman_q: float = TRACK_BOX_KALMAN_Q,
        kalman_r: float = TRACK_BOX_KALMAN_R,
    ) -> None:
        self.hold_seconds = max(0.0, hold_ms / 1000.0)
        self.wh_alpha = float(np.clip(wh_alpha, 0.0, 0.99))
        self.kalman_q = max(1e-6, float(kalman_q))
        self.kalman_r = max(1e-6, float(kalman_r))
        self._states: Dict[int, _TrackFilterState] = {}
        self._clock_s = 0.0

    def update(self, tracks: List[TrackState], frame_interval: float) -> tuple[List[TrackState], StabilizerStats]:
        dt = frame_interval if frame_interval > 0 else 1 / 30.0
        self._clock_s += dt

        visible_tracks: List[TrackState] = []
        seen_ids = set()

        for track in tracks:
            seen_ids.add(track.track_id)
            state = self._states.get(track.track_id)
            smoothed_bbox, label_anchor, state = self._update_track_state(track, state, dt)
            track.smoothed_bbox = smoothed_bbox
            track.label_anchor = label_anchor
            track.is_held = False

            state.last_bbox = smoothed_bbox
            state.last_track = self._clone_track(track)
            state.last_seen_s = self._clock_s
            self._states[track.track_id] = state

            visible_tracks.append(track)

        held_tracks = 0
        stale_track_ids: List[int] = []
        for track_id, state in self._states.items():
            if track_id in seen_ids:
                continue

            missing_for = self._clock_s - state.last_seen_s
            if missing_for > self.hold_seconds:
                stale_track_ids.append(track_id)
                continue

            held = self._clone_track(state.last_track)
            held.bbox = state.last_bbox
            held.smoothed_bbox = state.last_bbox
            held.label_anchor = state.label_anchor
            held.missed_frames += 1
            held.is_held = True
            visible_tracks.append(held)
            held_tracks += 1

        for track_id in stale_track_ids:
            self._states.pop(track_id, None)

        visible_tracks.sort(key=lambda item: item.track_id)
        stats = StabilizerStats(
            active_tracks=len(tracks),
            held_tracks=held_tracks,
            visible_tracks=len(visible_tracks),
        )
        return visible_tracks, stats

    def _update_track_state(
        self,
        track: TrackState,
        state: _TrackFilterState | None,
        dt: float,
    ) -> tuple[BBox, Tuple[float, float], _TrackFilterState]:
        x1, y1, x2, y2 = track.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        width = max(2.0, x2 - x1)
        height = max(2.0, y2 - y1)

        if state is None:
            kf = self._create_kalman(cx, cy, dt)
            filtered_cx, filtered_cy = cx, cy
            width_s = width
            height_s = height
            label_anchor = (x1, y1 - 10.0)
        else:
            self._update_transition(state.kf, dt)
            state.kf.predict()
            measurement = np.array([[cx], [cy]], dtype=np.float32)
            corrected = state.kf.correct(measurement)
            filtered_cx = float(corrected[0, 0])
            filtered_cy = float(corrected[1, 0])
            width_s = self.wh_alpha * state.width + (1.0 - self.wh_alpha) * width
            height_s = self.wh_alpha * state.height + (1.0 - self.wh_alpha) * height
            target_anchor = (filtered_cx - width_s / 2.0, filtered_cy - height_s / 2.0 - 10.0)
            label_anchor = (
                0.75 * state.label_anchor[0] + 0.25 * target_anchor[0],
                0.75 * state.label_anchor[1] + 0.25 * target_anchor[1],
            )
            kf = state.kf

        smoothed_bbox = (
            filtered_cx - width_s / 2.0,
            filtered_cy - height_s / 2.0,
            filtered_cx + width_s / 2.0,
            filtered_cy + height_s / 2.0,
        )

        next_state = _TrackFilterState(
            kf=kf,
            width=width_s,
            height=height_s,
            label_anchor=label_anchor,
            last_bbox=smoothed_bbox,
            last_track=self._clone_track(track),
            last_seen_s=self._clock_s,
        )
        return smoothed_bbox, label_anchor, next_state

    def _create_kalman(self, cx: float, cy: float, dt: float) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        self._update_transition(kf, dt)
        kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.kalman_q
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.kalman_r
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        return kf

    @staticmethod
    def _update_transition(kf: cv2.KalmanFilter, dt: float) -> None:
        kf.transitionMatrix = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _clone_track(track: TrackState) -> TrackState:
        return TrackState(
            track_id=track.track_id,
            bbox=track.bbox,
            score=track.score,
            smoothed_bbox=track.smoothed_bbox,
            label_anchor=track.label_anchor,
            foot_point=track.foot_point,
            zone_state=track.zone_state,
            on_zebra=track.on_zebra,
            history=list(track.history),
            missed_frames=track.missed_frames,
            is_held=track.is_held,
            velocity=track.velocity,
            speed=track.speed,
            risk_score=track.risk_score,
            predicted_positions=list(track.predicted_positions),
        )
