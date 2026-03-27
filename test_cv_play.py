import cv2
import time

video_path = r"D:\ZebraWatch\720p.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("打不开视频")
    exit()

frame_count = 0
t0 = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    cv2.imshow("raw", frame)
    # 这里只用 1ms 的 waitKey，不做人为减速
    if cv2.waitKey(1) & 0xFF == 27:
        break

t1 = time.time()
print("总帧数:", frame_count, "总时间:", t1 - t0, "平均FPS:", frame_count / (t1 - t0))
cap.release()
cv2.destroyAllWindows()