"""Benchmark the perception pipeline with TEST.mp4."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.safety_analysis.perception_fusion import PerceptionFusion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="TEST.mp4")
    parser.add_argument("--frames", type=int, default=180)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    pipeline = PerceptionFusion()
    stats = {"preprocess": 0.0, "detect": 0.0, "track": 0.0, "predict": 0.0, "render": 0.0, "total": 0.0}
    frames = 0
    start = time.time()

    while frames < args.frames:
        success, frame = capture.read()
        if not success:
            break
        result = pipeline.process_frame(frame)
        stats["preprocess"] += result.stats.preprocess_ms
        stats["detect"] += result.stats.detect_ms
        stats["track"] += result.stats.track_ms
        stats["predict"] += result.stats.predict_ms
        stats["render"] += result.stats.render_ms
        stats["total"] += result.stats.total_ms
        frames += 1

    elapsed = time.time() - start
    capture.release()

    averages = {f"avg_{key}_ms": (value / frames if frames else 0.0) for key, value in stats.items()}
    report = {
        "video": str(video_path),
        "frames": frames,
        "wall_time_s": elapsed,
        "wall_fps": (frames / elapsed) if elapsed else 0.0,
        **averages,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
