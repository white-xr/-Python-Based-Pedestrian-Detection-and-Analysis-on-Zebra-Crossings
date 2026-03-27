import time

import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from src.config import DISPLAY_FPS_CAP
from src.safety_analysis.perception_fusion import PerceptionFusion


class VideoReadThread(QThread):
    signalFrame = pyqtSignal(object)
    signalFailed = pyqtSignal(str)

    def __init__(self, perception: PerceptionFusion | None = None, parent=None):
        super().__init__(parent)
        self.work = False
        self.pause = False
        self.video_path = ""
        self.capture = None
        self.perception = perception or PerceptionFusion()

    def threadStart(self, path):
        self.video_path = path
        self.work = True
        self.pause = False
        self.start()

    def threadStop(self):
        self.work = False
        self.quit()
        self.wait()

    def run(self):
        try:
            self.capture = cv2.VideoCapture(self.video_path)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.capture.isOpened():
                self.signalFailed.emit("无法打开视频或摄像头。")
                return

            source_fps = self.capture.get(cv2.CAP_PROP_FPS)
            if not source_fps or source_fps < 1 or source_fps > 120:
                source_fps = DISPLAY_FPS_CAP
            playback_fps = float(source_fps)
            frame_interval = 1.0 / playback_fps
            start_clock = time.perf_counter()
            timeline_index = 0

            while self.work:
                loop_start = time.perf_counter()

                if self.pause:
                    self.msleep(60)
                    continue

                success, frame = self.capture.read()
                if not success:
                    break

                result = self.perception.process_frame(frame, frame_interval=frame_interval)
                result.stats.fps = DISPLAY_FPS_CAP
                self.signalFrame.emit(result)
                timeline_index += 1

                now = time.perf_counter()
                expected_time = start_clock + timeline_index * frame_interval
                lag = now - expected_time
                if lag > frame_interval:
                    # Processing falls behind real-time, drop a few source frames to prevent slow-motion playback.
                    skip_frames = int(lag / frame_interval)
                    for _ in range(skip_frames):
                        if not self.capture.grab():
                            break
                        timeline_index += 1
                    expected_time = start_clock + timeline_index * frame_interval

                elapsed = now - loop_start
                remaining = min(frame_interval - elapsed, expected_time - now)
                if remaining > 0:
                    self.msleep(max(1, int(remaining * 1000)))

        except Exception as err:
            self.signalFailed.emit(str(err))
        finally:
            if self.capture is not None:
                self.capture.release()
