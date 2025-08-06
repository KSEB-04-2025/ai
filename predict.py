from ultralytics import YOLO
from PIL import Image
import cv2
import os

# 모델 로드
model = YOLO(r"C:\Users\happy\OneDrive\Desktop\yolov8n\model\v4\best.pt")

# 이미지가 들어 있는 폴더 경로
image_dir = r"C:\Users\happy\OneDrive\Desktop\yolov8n\test"  # 폴더 경로로 수정
image_extensions = [".jpg", ".jpeg", ".png"]

# 폴더 내 이미지 파일 순회
for filename in os.listdir(image_dir):
    if os.path.splitext(filename)[1].lower() in image_extensions:
        image_path = os.path.join(image_dir, filename)
        img = Image.open(image_path).convert("RGB")  # RGB로 변환 (필요한 경우)

        # 예측
        results = model(img, iou=0.3, conf=0.44)

        # 결과 출력
        print(f"{filename} 결과:")
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"→ 클래스: {model.names[cls]}, 신뢰도: {conf:.2f}")

        # 시각화
        annotated_img = results[0].plot()
        cv2.imshow("YOLOv8 예측 결과", annotated_img)
        cv2.waitKey(0)

cv2.destroyAllWindows()
