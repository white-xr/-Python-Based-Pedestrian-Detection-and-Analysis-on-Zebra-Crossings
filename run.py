import sys

import ultralytics
from PyQt5.QtWidgets import QApplication

from src.gui.mainwindow import PDMainWindow


def main():
    print("=" * 40)
    print(f"YOLO 鐗堟湰: {ultralytics.__version__}")
    print("姝ｅ湪鍚姩鏂戦┈绾胯浜烘娴嬩笌瀹夊叏棰勮绯荤粺...")
    print("=" * 40)

    try:
        app = QApplication(sys.argv)
        win = PDMainWindow()
        win.show()
        exit_code = app.exec_()
        print(f"搴旂敤宸查€€鍑猴紝閫€鍑虹爜: {exit_code}")
        sys.exit(exit_code)
    except Exception as exc:
        print("鍚姩澶辫触:")
        print(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
