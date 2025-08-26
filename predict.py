from ultralytics import YOLO
from PIL import Image
import cv2
import os


model = YOLO(r"C:\Users\happy\OneDrive\Desktop\yolov8n\model\v4\best.pt")


image_dir = r"C:\Users\happy\OneDrive\Desktop\yolov8n\test"  
image_extensions = [".jpg", ".jpeg", ".png"]


for filename in os.listdir(image_dir):
    if os.path.splitext(filename)[1].lower() in image_extensions:
        image_path = os.path.join(image_dir, filename)
        img = Image.open(image_path).convert("RGB")  

       
        results = model(img, iou=0.3, conf=0.44)

       
        print(f"{filename} 결과:")
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"→ 클래스: {model.names[cls]}, 신뢰도: {conf:.2f}")

        
        annotated_img = results[0].plot()
        cv2.imshow("YOLOv8 예측 결과", annotated_img)
        cv2.waitKey(0)

cv2.destroyAllWindows()
