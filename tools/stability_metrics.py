"""Compute tracking stability metrics for zebra-crossing safety pipeline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.safety_analysis.perception_fusion import PerceptionFusion


def p95(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, 95))


def second_diff(values: List[Tuple[int, float, float]]) -> List[float]:
    jitter: List[float] = []
    for idx in range(2, len(values)):
        f0, x0, y0 = values[idx - 2]
        f1, x1, y1 = values[idx - 1]
        f2, x2, y2 = values[idx]
        if f2 - f1 != 1 or f1 - f0 != 1:
            continue
        ddx = x2 - 2 * x1 + x0
        ddy = y2 - 2 * y1 + y0
        jitter.append(float(math.hypot(ddx, ddy)))
    return jitter


def trajectory_second_diff(points: List[Tuple[float, float]]) -> List[float]:
    jitter: List[float] = []
    for idx in range(2, len(points)):
        x0, y0 = points[idx - 2]
        x1, y1 = points[idx - 1]
        x2, y2 = points[idx]
        ddx = x2 - 2 * x1 + x0
        ddy = y2 - 2 * y1 + y0
        jitter.append(float(math.hypot(ddx, ddy)))
    return jitter


def analyze_video(video: Path, max_frames: int = 0) -> Dict[str, float | int | str]:
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps < 1 or source_fps > 120:
        source_fps = 30.0
    frame_interval = 1.0 / source_fps

    pipeline = PerceptionFusion()

    frames = 0
    seen_ids = set()
    id_new_count = 0
    prev_ids = set()
    lost_frame: Dict[int, int] = {}
    flicker_count = 0
    flicker_gap = int(max(6, round(source_fps * 0.4)))

    centers: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    bbox_jitter_values: List[float] = []
    traj_jitter_values: List[float] = []

    while True:
        if max_frames and frames >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        result = pipeline.process_frame(frame, frame_interval=frame_interval)
        current_ids = {track.track_id for track in result.tracks}

        new_ids = current_ids - seen_ids
        id_new_count += len(new_ids)
        seen_ids |= current_ids

        vanished = prev_ids - current_ids
        appeared = current_ids - prev_ids
        for tid in vanished:
            lost_frame[tid] = frames
        for tid in appeared:
            gone_frame = lost_frame.get(tid)
            if gone_frame is not None and (frames - gone_frame) <= flicker_gap:
                flicker_count += 1

        for track in result.tracks:
            bbox = track.smoothed_bbox or track.bbox
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            centers[track.track_id].append((frames, cx, cy))
            if track.predicted_positions:
                traj_jitter_values.extend(trajectory_second_diff(track.predicted_positions))

        prev_ids = current_ids
        frames += 1

    cap.release()

    for track_centers in centers.values():
        bbox_jitter_values.extend(second_diff(track_centers))

    windows = max(1.0, frames / 300.0)
    id_churn_per_300 = id_new_count / windows

    return {
        "video": str(video),
        "frames": frames,
        "source_fps": float(source_fps),
        "id_churn_per_300": float(id_churn_per_300),
        "bbox_jitter_p95": p95(bbox_jitter_values),
        "trajectory_jitter_p95": p95(traj_jitter_values),
        "flicker_count": int(flicker_count),
        "tracked_unique_ids": int(len(seen_ids)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="append", required=True, help="Path to a video file. Can be repeated.")
    parser.add_argument("--frames", type=int, default=0, help="Optional max frame count per video (0 means full).")
    args = parser.parse_args()

    reports = []
    for video_arg in args.video:
        video_path = Path(video_arg)
        reports.append(analyze_video(video_path, max_frames=args.frames))

    print(json.dumps({"reports": reports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
