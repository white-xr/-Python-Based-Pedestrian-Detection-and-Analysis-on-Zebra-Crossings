import os
import time

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFont, QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.config import DISPLAY_FPS_CAP, UI_SUMMARY_UPDATE_HZ
from src.gui.constant import SYS_NAME, SYS_VERSION
from src.safety_analysis.perception_fusion import PerceptionFusion, PerceptionResult
from src.threads.videoreadthread import VideoReadThread


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "--"):
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        self.valueLabel = QLabel(value)
        self.valueLabel.setObjectName("metricValue")

        layout.addWidget(title_label)
        layout.addWidget(self.valueLabel)

    def setValue(self, value: str):
        self.valueLabel.setText(value)


class InfoLine(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("infoLine")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        self.titleLabel = QLabel(title)
        self.titleLabel.setObjectName("infoTitle")
        self.bodyLabel = QLabel("--")
        self.bodyLabel.setWordWrap(True)
        self.bodyLabel.setObjectName("infoBody")

        layout.addWidget(self.titleLabel)
        layout.addWidget(self.bodyLabel)

    def setText(self, text: str):
        self.bodyLabel.setText(text)


class VideoPanel(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("videoPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setObjectName("panelTitle")
        self.imageLabel = QLabel()
        self.imageLabel.setObjectName("imageLabel")
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setMinimumSize(320, 220)
        self.imageLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(title_label)
        layout.addWidget(self.imageLabel, 1)


class PDMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.perception = PerceptionFusion()
        self.videoReadThread = VideoReadThread(perception=self.perception)
        self.videoReadThread.signalFrame.connect(self.slotUpdateResult)
        self.videoReadThread.signalFailed.connect(self.slotShowError)
        self.cameraReadThread = VideoReadThread(perception=self.perception)
        self.cameraReadThread.signalFrame.connect(self.slotUpdateResult)
        self.cameraReadThread.signalFailed.connect(self.slotShowError)

        self.beginRecoding = False
        self.video_out = None
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._last_summary_update = 0.0
        self._summary_update_interval = 1.0 / UI_SUMMARY_UPDATE_HZ if UI_SUMMARY_UPDATE_HZ > 0 else 0.5

        self._build_ui()
        self._apply_styles()

    def _build_ui(self):
        self.setWindowTitle(f"{SYS_NAME} {SYS_VERSION}")
        self.setWindowIcon(QIcon("./icons/icon.jpg"))
        self.resize(1520, 920)

        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(18)

        left_column = QVBoxLayout()
        left_column.setSpacing(18)
        root.addLayout(left_column, 4)

        hero_row = QHBoxLayout()
        hero_row.setSpacing(12)
        left_column.addLayout(hero_row)

        title_wrap = QVBoxLayout()
        title_wrap.setSpacing(4)
        title_label = QLabel(SYS_NAME)
        title_label.setObjectName("heroTitle")
        subtitle_label = QLabel("YOLOv8行人检测 | 斑马线安全预警 | 显示层稳定30FPS")
        subtitle_label.setObjectName("heroSubtitle")
        title_wrap.addWidget(title_label)
        title_wrap.addWidget(subtitle_label)
        hero_row.addLayout(title_wrap, 1)

        self.openVideoButton = QPushButton("打开视频")
        self.openVideoButton.clicked.connect(self.slotOpenVideo)
        self.openCameraButton = QPushButton("打开摄像头")
        self.openCameraButton.clicked.connect(self.slotOpenCamera)
        self.pauseButton = QPushButton("暂停")
        self.pauseButton.clicked.connect(self.slotPauseVideo)
        self.recordButton = QPushButton("录制")
        self.recordButton.clicked.connect(self.slotToggleRecording)
        for button in [self.openVideoButton, self.openCameraButton, self.pauseButton, self.recordButton]:
            button.setMinimumHeight(44)
            hero_row.addWidget(button)

        self.mainVideoPanel = VideoPanel("预警画面")
        self.rawVideoPanel = VideoPanel("原始画面")
        left_column.addWidget(self.mainVideoPanel, 4)
        left_column.addWidget(self.rawVideoPanel, 2)

        right_column = QVBoxLayout()
        right_column.setSpacing(18)
        root.addLayout(right_column, 2)

        metrics_wrap = QWidget()
        metrics_layout = QGridLayout(metrics_wrap)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setHorizontalSpacing(12)
        metrics_layout.setVerticalSpacing(12)

        self.personCard = MetricCard("稳定跟踪ID数")
        self.zebraCard = MetricCard("斑马线内人数")
        self.riskCard = MetricCard("整体风险")
        self.fpsCard = MetricCard("显示FPS", f"{DISPLAY_FPS_CAP:.0f}")

        metrics_layout.addWidget(self.personCard, 0, 0)
        metrics_layout.addWidget(self.zebraCard, 0, 1)
        metrics_layout.addWidget(self.riskCard, 1, 0)
        metrics_layout.addWidget(self.fpsCard, 1, 1)
        right_column.addWidget(metrics_wrap)

        summary_panel = QFrame()
        summary_panel.setObjectName("summaryPanel")
        summary_layout = QVBoxLayout(summary_panel)
        summary_layout.setContentsMargins(16, 16, 16, 16)
        summary_layout.setSpacing(10)
        summary_title = QLabel("当前摘要")
        summary_title.setObjectName("panelTitle")
        self.summaryLabel = QLabel("等待视频输入")
        self.summaryLabel.setWordWrap(True)
        self.summaryLabel.setObjectName("summaryText")
        self.statusBadge = QLabel("待机")
        self.statusBadge.setObjectName("statusBadge")
        summary_layout.addWidget(summary_title)
        summary_layout.addWidget(self.summaryLabel)
        summary_layout.addWidget(self.statusBadge, 0, Qt.AlignLeft)
        right_column.addWidget(summary_panel)

        details_panel = QFrame()
        details_panel.setObjectName("summaryPanel")
        details_layout = QVBoxLayout(details_panel)
        details_layout.setContentsMargins(16, 16, 16, 16)
        details_layout.setSpacing(10)
        details_title = QLabel("关键信息")
        details_title.setObjectName("panelTitle")
        self.primaryEvent = InfoLine("主要预警")
        self.secondaryEvent = InfoLine("次要预警")
        self.systemInfo = InfoLine("系统状态")
        details_layout.addWidget(details_title)
        details_layout.addWidget(self.primaryEvent)
        details_layout.addWidget(self.secondaryEvent)
        details_layout.addWidget(self.systemInfo)
        details_layout.addStretch(1)
        right_column.addWidget(details_panel, 1)

        self.statusBar().showMessage("系统已就绪")

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QWidget {
                background: #f3f5f7;
                color: #152033;
                font-family: "Microsoft YaHei UI", "Segoe UI";
            }
            #heroTitle {
                font-size: 26px;
                font-weight: 700;
                color: #142033;
            }
            #heroSubtitle {
                font-size: 13px;
                color: #5b6678;
            }
            QPushButton {
                border: 1px solid #d6dde8;
                border-radius: 12px;
                background: #ffffff;
                color: #172235;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #eef4ff;
                border-color: #9fbef5;
            }
            #videoPanel, #summaryPanel, #metricCard, #infoLine {
                background: #ffffff;
                border: 1px solid #dde4ee;
                border-radius: 18px;
            }
            #panelTitle, #metricTitle, #infoTitle {
                font-size: 13px;
                font-weight: 700;
                color: #5c6a7e;
                letter-spacing: 0.5px;
            }
            #metricValue {
                font-size: 28px;
                font-weight: 700;
                color: #142033;
            }
            #imageLabel {
                background: #0e1725;
                border-radius: 14px;
                border: 1px solid #223149;
            }
            #summaryText, #infoBody {
                font-size: 15px;
                line-height: 1.45;
                color: #1b2a41;
            }
            #statusBadge {
                border-radius: 12px;
                padding: 6px 12px;
                background: #edf8f1;
                color: #238a53;
                font-size: 13px;
                font-weight: 700;
            }
            """
        )

    @pyqtSlot(object)
    def slotUpdateResult(self, perception_result: PerceptionResult):
        self._set_frame(self.mainVideoPanel.imageLabel, perception_result.annotated_frame)
        self._set_frame(self.rawVideoPanel.imageLabel, perception_result.original_frame)

        now = time.time()
        if now - self._last_summary_update >= self._summary_update_interval:
            self.personCard.setValue(str(perception_result.stable_track_count or perception_result.person_count))
            self.zebraCard.setValue(str(perception_result.on_zebra_count))
            self.riskCard.setValue(perception_result.overall_risk)
            self.fpsCard.setValue(f"{DISPLAY_FPS_CAP:.0f}")

            self.summaryLabel.setText(perception_result.status_summary)
            self._update_status_badge(perception_result.overall_risk)
            self._update_info_lines(perception_result)
            self.statusBar().showMessage(
                f"{perception_result.status_summary} | 显示层锁定 {DISPLAY_FPS_CAP:.0f} FPS"
            )
            self._last_summary_update = now

        if self.beginRecoding and self.video_out:
            self.video_out.write(perception_result.annotated_frame)

    @pyqtSlot(str)
    def slotShowError(self, message: str):
        QMessageBox.warning(self, "运行错误", message)

    def _set_frame(self, label: QLabel, frame: np.ndarray):
        if frame is None:
            return

        target_width = max(320, label.width())
        target_height = max(220, label.height())
        h, w = frame.shape[:2]
        scale = min(target_width / float(w), target_height / float(h))
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(image))

    def _update_status_badge(self, level: str):
        if level == "危险":
            style = "background:#fff0f0;color:#c93434;"
            text = "危险"
        elif level == "注意":
            style = "background:#fff8e8;color:#bf7b12;"
            text = "注意"
        else:
            style = "background:#edf8f1;color:#238a53;"
            text = "安全"
        self.statusBadge.setText(text)
        self.statusBadge.setStyleSheet(
            f"border-radius: 12px; padding: 6px 12px; font-size: 13px; font-weight: 700; {style}"
        )

    def _update_info_lines(self, result: PerceptionResult):
        top_risks = result.track_risks[:2]
        if top_risks:
            first = top_risks[0]
            first_reason = first.reasons[0] if first.reasons else "无"
            self.primaryEvent.setText(f"目标 {first.track_id} | {first.level} | {first_reason}")
        else:
            self.primaryEvent.setText("暂无主要预警")

        if len(top_risks) > 1:
            second = top_risks[1]
            second_reason = second.reasons[0] if second.reasons else "无"
            self.secondaryEvent.setText(f"目标 {second.track_id} | {second.level} | {second_reason}")
        else:
            self.secondaryEvent.setText("暂无次要预警")

        high_risk_ids = ",".join(str(track_id) for track_id in result.high_risk_track_ids) or "无"
        approach_count = result.zone_counts.get("approach", 0)
        scene_source = {"manual": "手工标定", "auto": "自动生成"}.get(result.scene_source, "未配置")
        self.systemInfo.setText(
            f"稳定ID: {result.stable_track_count or result.person_count} | 接近区: {approach_count} | "
            f"高风险ID: {high_risk_ids} | 区域: {scene_source} | "
            f"耗时: {result.stats.total_ms:.1f} ms | 显示FPS: {DISPLAY_FPS_CAP:.0f}"
        )

    def slotOpenVideo(self):
        if self.cameraReadThread.isRunning():
            QMessageBox.information(self, "提示", "请先关闭摄像头。")
            return
        if self.videoReadThread.isRunning():
            QMessageBox.information(self, "提示", "视频已在播放。")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频",
            os.path.expanduser("~/Videos"),
            "视频文件 (*.mp4 *.avi *.mov *.mkv)",
        )
        if file_path and os.path.exists(file_path):
            self.videoReadThread.threadStart(file_path)

    def slotOpenCamera(self):
        if self.videoReadThread.isRunning():
            QMessageBox.information(self, "提示", "请先关闭当前视频。")
            return
        if self.cameraReadThread.isRunning():
            QMessageBox.information(self, "提示", "摄像头已打开。")
            return
        self.cameraReadThread.threadStart(0)

    def slotPauseVideo(self):
        active_thread = self.videoReadThread if self.videoReadThread.isRunning() else self.cameraReadThread
        if active_thread.isRunning():
            active_thread.pause = not active_thread.pause
            self.pauseButton.setText("继续" if active_thread.pause else "暂停")

    def slotToggleRecording(self):
        if not self.beginRecoding:
            self.slotBeginRecoding()
        else:
            self.slotEndRecoding()

    def slotBeginRecoding(self):
        active_thread = self.videoReadThread if self.videoReadThread.isRunning() else self.cameraReadThread
        if not active_thread.isRunning():
            QMessageBox.information(self, "提示", "请先打开视频或摄像头。")
            return

        success, frame = active_thread.capture.read()
        if not success or frame is None:
            QMessageBox.warning(self, "错误", "无法获取录制帧。")
            return

        self.beginRecoding = True
        filename = time.strftime("%Y-%m-%d_%H-%M-%S") + ".avi"
        h, w = frame.shape[:2]
        self.video_out = cv2.VideoWriter(filename, self.fourcc, DISPLAY_FPS_CAP, (w, h))
        self.recordButton.setText("结束录制")
        self.statusBar().showMessage(f"正在录制: {filename}", 3000)

    def slotEndRecoding(self):
        if self.video_out:
            self.video_out.release()
            self.video_out = None
        self.beginRecoding = False
        self.recordButton.setText("录制")
        self.statusBar().showMessage("录制已结束", 3000)

    def closeEvent(self, event):
        if self.videoReadThread.isRunning():
            self.videoReadThread.threadStop()
        if self.cameraReadThread.isRunning():
            self.cameraReadThread.threadStop()
        if self.video_out:
            self.video_out.release()
        event.accept()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    win = PDMainWindow()
    win.show()
    sys.exit(app.exec_())
