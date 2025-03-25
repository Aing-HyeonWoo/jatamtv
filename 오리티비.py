import cv2
import torch
import os
import time
import numpy as np
from picamera2 import Picamera2

# YOLOv5 모델 로드
MODEL_PATH = "/home/pi/best.pt"  # 🔴 `best.pt` 경로 확인 후 수정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
model.to(device)
model.eval()

# 클래스 이름 딕셔너리
CLASS_NAMES = {
    0: "Car",
    1: "Human",
    2: "what"
}

# 라즈베리파이 카메라 초기화
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)  # 🔵 해상도 조정 가능
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# 실시간 감지 루프
try:
    while True:
        # 📷 카메라에서 이미지 캡처
        frame = picam2.capture_array()

        # 모델 추론
        results = model(frame)  # YOLOv5 추론
        detections = results.pred[0].cpu().numpy()  # 결과 가져오기

        for detection in detections:
            x_min, y_min, x_max, y_max, conf, class_id = detection[:6]
            class_id = int(class_id)

            if conf > 0.05:  # 신뢰도 5% 이상 필터링
                class_name = CLASS_NAMES.get(class_id, f"Unknown({class_id})")
                print(f"{class_name}: x={x_min:.2f}, y={y_min:.2f}, conf={conf:.2f}")

                # 바운딩 박스 그리기
                color = (0, 255, 0)  # 초록색 박스
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)

                # 클래스 이름 및 신뢰도 표시
                label = f"{class_name} ({conf:.2f})"
                cv2.putText(frame, label, (int(x_min), int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 📺 화면에 출력
        cv2.imshow("YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # 'q' 누르면 종료

except KeyboardInterrupt:
    print("종료됨")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
