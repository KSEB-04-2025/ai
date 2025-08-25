from ultralytics import YOLO
import time


pt_path = r"C:\Users\happy\OneDrive\Desktop\yolov8n\model\v4\best.pt"

print("[1/2] 모델 로드 중...")
model = YOLO(pt_path)

print("[2/2] ONNX 변환 시작... (수 초 ~ 수십 초 소요)")
start_time = time.time()

onnx_path = model.export(
    format="onnx",   
    opset=12,        
    simplify=True,   
    dynamic=True     
)

elapsed = time.time() - start_time
print(f" ONNX 변환 완료! 파일: {onnx_path} (소요 시간: {elapsed:.2f}초)")
