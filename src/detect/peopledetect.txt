from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from collections import deque
import os


class PeopleDetect:
    def __init__(self):
        # 加载 YOLOv8 预训练模型
        self.model = YOLO("yolov8n.pt")

        # 尝试加载中文字体
        font_path = "SimHei.ttf"
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"字体文件未找到: {font_path}")
        self.font = ImageFont.truetype(font_path, 20)

        # 初始化行人数量缓冲队列（用于平滑）
        self.person_queue = deque(maxlen=10)

        # 配色方案（统一管理）
        self.box_color = (148, 0, 211)       # 框色：暗紫色
        self.dot_color = (0, 255, 255)       # 圆点色：青色
        self.text_color = (221, 160, 221)    # 文本色：淡紫红

    def detectVideo(self, frame):
        # 检测当前帧
        results = self.model(frame)
        orig = frame.copy()

        # 转为 PIL 图像用于绘制中文
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_frame)

        person_count = 0  # 当前帧检测到的行人数

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if cls == 0 and conf > 0.5:
                    person_count += 1

                    # 画框
                    draw.rectangle([(x1, y1), (x2, y2)], outline=self.box_color, width=3)

                    # 画四角圆点
                    r = 7
                    corner_points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
                    for cx, cy in corner_points:
                        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=self.dot_color)

                    # 画顶部中间倒三角指示（使用亮红色）
                    mid_top_x = (x1 + x2) // 2
                    triangle_size = 20
                    triangle = [
                        (mid_top_x, y1 - 1),  # 顶点
                        (mid_top_x - triangle_size, y1 - 1 - triangle_size),
                        (mid_top_x + triangle_size, y1 - 1 - triangle_size),
                    ]
                    draw.polygon(triangle, fill=(255, 0, 0))  # 亮红色

                    # 画标签
                    draw.text((x1, y1 - 30), f"行人 {conf:.2f}", font=self.font, fill=self.text_color)

        # 更新队列并计算平均值
        self.person_queue.append(person_count)
        avg_person_count = round(sum(self.person_queue) / len(self.person_queue))

        # 转回 OpenCV 格式
        frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

        return orig, frame, avg_person_count