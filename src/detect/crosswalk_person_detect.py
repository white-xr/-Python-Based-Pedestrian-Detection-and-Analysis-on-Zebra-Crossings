from ultralytics import YOLO
import cv2
import numpy as np

class CrosswalkPersonDetector:
    def __init__(self, person_model='yolov8n.pt', zebra_model='zebra.pt', device='cpu'):
        self.person_model = YOLO(person_model).to(device)  # 支持GPU
        self.zebra_model = YOLO(zebra_model).to(device)
        self.PERSON_CLASS_ID = 0  # COCO数据集中的'person'类别ID
        self.ZEBRA_DETECTION_INTERVAL = 10  # Added for the new detectVideo method
        self.skip_zebra_counter = 0  # Added for the new detectVideo method
        self.last_zebra_boxes = None  # Added for the new detectVideo method

    def calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        # 计算交集区域
        x_inter1 = max(x1, x3)
        y_inter1 = max(y1, y3)
        x_inter2 = min(x2, x4)
        y_inter2 = min(y2, y4)

        # 检查是否有交集
        if x_inter2 <= x_inter1 or y_inter2 <= y_inter1:
            return 0.0

        # 计算交集面积
        area_inter = (x_inter2 - x_inter1) * (y_inter2 - y_inter1)

        # 计算并集面积
        area_box1 = (x2 - x1) * (y2 - y1)
        area_box2 = (x4 - x3) * (y4 - y3)
        area_union = area_box1 + area_box2 - area_inter

        # 计算IoU
        iou = area_inter / area_union if area_union > 0 else 0.0
        return iou

    def calculate_iou_batch(self, boxes1, boxes2):
        """批量计算IoU"""
        ious = []
        for box1 in boxes1:
            ious.append([self.calculate_iou(box1, box2) for box2 in boxes2])
        return np.array(ious)

    def detectVideo(self, frame):
        try:
            # 行人和斑马线检测
            person_results = self.person_model.predict(source=frame, conf=0.3, verbose=False)[0]
            zebra_results = self.zebra_model.predict(source=frame, conf=0.3, verbose=False)[0]

            person_boxes = []
            if person_results.boxes:
                person_boxes = [box.xyxy.cpu().numpy()[0] for box in person_results.boxes if int(box.cls) == self.PERSON_CLASS_ID]

            zebra_boxes = self.last_zebra_boxes if self.last_zebra_boxes is not None else []
            if self.skip_zebra_counter >= self.ZEBRA_DETECTION_INTERVAL:
                zebra_boxes = zebra_results.boxes.xyxy.cpu().numpy() if zebra_results.boxes else []
                self.last_zebra_boxes = zebra_boxes
                self.skip_zebra_counter = 0
            self.skip_zebra_counter += 1

            combined_frame = frame.copy()

            # 绘制斑马线框
            for box in zebra_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(combined_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(combined_frame, "Zebra", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # 批量计算IoU
            if person_boxes and len(zebra_boxes) > 0:
                ious = self.calculate_iou_batch(person_boxes, zebra_boxes)
                max_ious = np.max(ious, axis=1)
            else:
                max_ious = np.zeros(len(person_boxes))

            # 绘制行人框
            person_count = len(person_boxes)
            for i, box in enumerate(person_boxes):
                x1, y1, x2, y2 = map(int, box)
                # 使用行人底部中心点和底部区域来判断
                foot_point_x = (x1 + x2) // 2
                foot_point_y = y2  # 行人底部y坐标
                
                on_zebra = False
                for zbox in zebra_boxes:
                    zx1, zy1, zx2, zy2 = map(int, zbox)
                    # 放宽判定条件：
                    # 1. 检查脚部位置是否在斑马线区域内（水平方向放宽一些）
                    # 2. 或者检查是否有足够大的重叠区域
                    foot_in_zebra = (
                        foot_point_x >= (zx1 - 20) and  # 左右各放宽20像素
                        foot_point_x <= (zx2 + 20) and
                        foot_point_y >= (zy1 - 10) and  # 上下各放宽10像素
                        foot_point_y <= (zy2 + 10)
                    )
                    
                    if foot_in_zebra or max_ious[i] > 0.2:  # 提高IoU阈值，但作为备选判断条件
                        on_zebra = True
                        break

                color = (0, 255, 0) if on_zebra else (0, 0, 255)
                label = "OnZebra" if on_zebra else "OffZebra"
                
                # 绘制行人框和底部中心点
                cv2.rectangle(combined_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(combined_frame, (foot_point_x, foot_point_y), 3, color, -1)  # 绘制底部中心点
                cv2.putText(combined_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            return frame, combined_frame, person_count
        except Exception as e:
            print(f"检测错误: {e}")
            return frame, frame, 0

# 示例用法
if __name__ == "__main__":
    detector = CrosswalkPersonDetector(device='cpu')  # 改为'cuda'如果有GPU
    cap = cv2.VideoCapture(0)  # 打开摄像头或替换为视频文件路径（如'video.mp4'）
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        original_frame, processed_frame, count = detector.detectVideo(frame)
        cv2.imshow('Crosswalk Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()