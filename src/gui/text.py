import cv2
import torch
from PIL import Image
import numpy as np
import os
import time
from PyQt5.QtCore import pyqtSignal, QThread
import subprocess


class PeopleDetect(QThread):
    update_frame = pyqtSignal(np.ndarray)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.model = torch.hub.load('ultralytics/yolov8', 'yolov8n.pt', pretrained=True)

    def draw_corner_lines(self, frame, box, color=(255, 0, 0)):
        """
        绘制四个角的点和线条
        :param frame: 图像帧
        :param box: 检测框 [x1, y1, x2, y2]
        :param color: 颜色，默认为红色 (BGR格式)
        """
        x1, y1, x2, y2 = box
        # 修改角点大小
        cv2.circle(frame, (x1, y1), 10, color, -1)  # 左上角
        cv2.circle(frame, (x1, y2), 10, color, -1)  # 左下角
        cv2.circle(frame, (x2, y1), 10, color, -1)  # 右上角
        cv2.circle(frame, (x2, y2), 10, color, -1)  # 右下角

        # 修改线条粗细
        cv2.line(frame, (x1, y1), (x1, y2), color, 6)  # 左边线
        cv2.line(frame, (x1, y1), (x2, y1), color, 6)  # 上边线
        cv2.line(frame, (x2, y1), (x2, y2), color, 3)  # 右边线
        cv2.line(frame, (x1, y2), (x2, y2), color, 3)  # 下边线

        # 绘制连接线条
        cv2.line(frame, (x1, y1), (x1, y2), color, 2)
        cv2.line(frame, (x1, y1), (x2, y1), color, 2)
        cv2.line(frame, (x2, y1), (x2, y2), color, 2)
        cv2.line(frame, (x1, y2), (x2, y2), color, 2)


    def detectVideo(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # 创建保存视频的对象
        output_path = os.path.splitext(video_path)[0] + '_output.mp4'
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)
            dets = results[0].boxes.cpu().numpy()

            for *xyxy, conf, cls in reversed(dets):
                box = list(map(int, xyxy))
                self.draw_corner_lines(frame, box, color=(0, 0, 255))  # 修改为蓝色

            out.write(frame)
            self.update_frame.emit(frame)
            time.sleep(1 / fps)

        cap.release()
        out.release()

    def run(self):
        self.detectVideo(self.video_path)
