"""Manual scene-zone annotation tool for zebra-crossing warning regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt


def load_frame(source: str):
    path = Path(source)
    if path.exists():
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".bmp"}:
            frame = cv2.imread(str(path))
            if frame is None:
                raise SystemExit(f"无法读取图像: {path}")
            return frame

        capture = cv2.VideoCapture(str(path))
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            raise SystemExit(f"无法读取视频首帧: {path}")
        return frame

    raise SystemExit(f"输入不存在: {source}")


def request_polygon(image_rgb, title: str, count: int):
    plt.figure(figsize=(12, 7))
    plt.imshow(image_rgb)
    if count > 0:
        prompt = f"{title}，请点击 {count} 个点，按 Enter 完成"
    else:
        prompt = f"{title}，请连续点击任意个点，按 Enter 完成"
    plt.title(prompt)
    plt.axis("off")
    points = plt.ginput(count if count > 0 else -1, timeout=0)
    plt.close()
    return [[x / image_rgb.shape[1], y / image_rgb.shape[0]] for x, y in points]


def request_rect(image_rgb, title: str):
    plt.figure(figsize=(12, 7))
    plt.imshow(image_rgb)
    plt.title(f"{title}，请点击左上和右下两个点，按 Enter 完成")
    plt.axis("off")
    points = plt.ginput(2, timeout=0)
    plt.close()
    if len(points) < 2:
        return []
    x1, y1 = points[0]
    x2, y2 = points[1]
    return [
        min(x1, x2) / image_rgb.shape[1],
        min(y1, y2) / image_rgb.shape[0],
        max(x1, x2) / image_rgb.shape[1],
        max(y1, y2) / image_rgb.shape[0],
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="图片或视频路径")
    parser.add_argument("--output", default="scene_zones.json", help="输出 JSON 路径")
    parser.add_argument("--zebra-points", type=int, default=4, help="斑马线多边形点数，-1 表示任意点")
    parser.add_argument("--approach-points", type=int, default=0, help="接近区点数，0 表示自动生成")
    parser.add_argument("--conflict-points", type=int, default=0, help="冲突区点数，0 表示自动生成")
    parser.add_argument("--with-focus-roi", action="store_true", help="是否额外手工标定关注检测区域")
    args = parser.parse_args()

    frame = load_frame(args.source)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    zebra_zone = request_polygon(image_rgb, "标定斑马线主体区域", args.zebra_points)
    approach_zone = request_polygon(image_rgb, "标定斑马线接近区", args.approach_points) if args.approach_points != 0 else []
    conflict_zone = request_polygon(image_rgb, "标定道路冲突区", args.conflict_points) if args.conflict_points != 0 else []
    focus_roi = request_rect(image_rgb, "标定关注检测区域") if args.with_focus_roi else []

    payload = {
        "normalized": True,
        "zebra_zone": zebra_zone,
    }
    if approach_zone:
        payload["approach_zone"] = approach_zone
    if conflict_zone:
        payload["conflict_zone"] = conflict_zone
    if focus_roi:
        payload["focus_roi"] = focus_roi
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"场景区域已保存到: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
