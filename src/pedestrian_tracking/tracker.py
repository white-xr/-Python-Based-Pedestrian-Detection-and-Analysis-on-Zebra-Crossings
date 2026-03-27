"""Lightweight IOU-based tracker for pedestrian detections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from src.config import TRACK_IOU_THRESHOLD, TRACK_MAX_HISTORY, TRACK_MAX_MISSED


BBox = Tuple[float, float, float, float]


@dataclass
class TrackState:
    track_id: int
    bbox: BBox
    score: float
    smoothed_bbox: BBox | None = None
    label_anchor: Tuple[float, float] | None = None
    foot_point: Tuple[float, float] | None = None
    zone_state: str = "outside"
    on_zebra: bool = False
    history: List[Tuple[float, float]] = field(default_factory=list)
    missed_frames: int = 0
    is_held: bool = False
    velocity: Tuple[float, float] = (0.0, 0.0)
    speed: float = 0.0
    risk_score: float = 0.0
    predicted_positions: List[Tuple[float, float]] = field(default_factory=list)


class SimpleTracker:
    """Greedy IOU matching tracker with optional motion extrapolation."""

    def __init__(
        self,
        iou_threshold: float = TRACK_IOU_THRESHOLD,
        max_history: int = TRACK_MAX_HISTORY,
        max_missed: int = TRACK_MAX_MISSED,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_history = max_history
        self.max_missed = max_missed
        self.tracks: Dict[int, TrackState] = {}
        self.next_id = 1

    def update(
        self,
        detections: List[Dict[str, object]] | None,
        frame_interval: float = 1 / 30.0,
    ) -> List[TrackState]:
        """Update tracker state with current detections or motion-only advance."""
        if detections is None:
            return self._advance_without_detections(frame_interval)

        det_bboxes = [det["bbox"] for det in detections]
        scores = [float(det.get("score", 0.0)) for det in detections]

        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid].bbox for tid in track_ids]

        matches, unmatched_tracks, unmatched_dets = self._match(track_bboxes, det_bboxes)

        for t_idx, d_idx in matches:
            track_id = track_ids[t_idx]
            track = self.tracks[track_id]
            track.bbox = det_bboxes[d_idx]
            track.score = scores[d_idx]
            track.missed_frames = 0
            self._append_history(track, det_bboxes[d_idx], frame_interval)

        for t_idx in unmatched_tracks:
            track_id = track_ids[t_idx]
            track = self.tracks[track_id]
            track.missed_frames += 1
            self._advance_track(track, frame_interval)

        for d_idx in unmatched_dets:
            bbox = det_bboxes[d_idx]
            score = scores[d_idx]
            new_track = TrackState(track_id=self.next_id, bbox=bbox, score=score)
            self._append_history(new_track, bbox, frame_interval)
            self.tracks[self.next_id] = new_track
            self.next_id += 1

        self._prune_tracks()
        return list(self.tracks.values())

    def _advance_without_detections(self, frame_interval: float) -> List[TrackState]:
        for track in self.tracks.values():
            track.missed_frames += 1
            self._advance_track(track, frame_interval)
        self._prune_tracks()
        return list(self.tracks.values())

    def _advance_track(self, track: TrackState, frame_interval: float) -> None:
        if frame_interval <= 0:
            return

        vx, vy = track.velocity
        if vx == 0.0 and vy == 0.0:
            return

        dx = vx * frame_interval
        dy = vy * frame_interval
        x1, y1, x2, y2 = track.bbox
        new_bbox = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        track.bbox = new_bbox
        self._append_history(track, new_bbox, frame_interval, update_velocity=False)

    def _append_history(
        self,
        track: TrackState,
        bbox: BBox,
        frame_interval: float,
        update_velocity: bool = True,
    ) -> None:
        alpha = 0.7
        if track.smoothed_bbox is None:
            track.smoothed_bbox = bbox
        else:
            sx1, sy1, sx2, sy2 = track.smoothed_bbox
            x1, y1, x2, y2 = bbox
            track.smoothed_bbox = (
                alpha * sx1 + (1 - alpha) * x1,
                alpha * sy1 + (1 - alpha) * y1,
                alpha * sx2 + (1 - alpha) * x2,
                alpha * sy2 + (1 - alpha) * y2,
            )

        cx = (bbox[0] + bbox[2]) / 2
        cy = bbox[3]
        prev_point = track.history[-1] if track.history else None
        track.history.append((cx, cy))
        if len(track.history) > self.max_history:
            track.history.pop(0)

        if prev_point is not None and frame_interval > 0 and update_velocity:
            vx = (cx - prev_point[0]) / frame_interval
            vy = (cy - prev_point[1]) / frame_interval
            track.velocity = (vx, vy)
            track.speed = float(np.hypot(vx, vy))
        elif prev_point is None:
            track.velocity = (0.0, 0.0)
            track.speed = 0.0

    def _prune_tracks(self) -> None:
        to_remove = [tid for tid, track in self.tracks.items() if track.missed_frames > self.max_missed]
        for tid in to_remove:
            self.tracks.pop(tid, None)

    def _match(
        self,
        tracks: List[BBox],
        detections: List[BBox],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        iou_matrix = self._iou_matrix(tracks, detections)
        matches: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        while True:
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_idx]
            if max_iou < self.iou_threshold:
                break
            t_idx, d_idx = max_idx
            matches.append((t_idx, d_idx))
            unmatched_tracks.discard(t_idx)
            unmatched_dets.discard(d_idx)
            iou_matrix[t_idx, :] = -1
            iou_matrix[:, d_idx] = -1

        return matches, list(unmatched_tracks), list(unmatched_dets)

    def _iou_matrix(self, tracks: List[BBox], dets: List[BBox]) -> np.ndarray:
        matrix = np.zeros((len(tracks), len(dets)), dtype=float)
        for i, track_box in enumerate(tracks):
            for j, det_box in enumerate(dets):
                matrix[i, j] = self._iou(track_box, det_box)
        return matrix

    @staticmethod
    def _iou(box_a: BBox, box_b: BBox) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union_area = area_a + area_b - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area
