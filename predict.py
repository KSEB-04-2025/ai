from ultralytics import YOLO
import os

#  모델 로드
model = YOLO("model/best.pt")

#  예측할 이미지 경로 (단일 파일)
image_path = "이미지경로"
image_name = os.path.basename(image_path)

#  결과 저장 폴더
save_dir = os.path.join('model', 'result')
os.makedirs(save_dir, exist_ok=True)

#  결과 이미지 저장 경로
result_img_dir = os.path.join(save_dir, 'predict_result')
result_txt_path = os.path.join(save_dir, f"{image_name}.txt")

#  예측 수행
results = model.predict(
    source=image_path,
    conf=0.3,
    device='cuda',
    save=True,
    project='model',
    name='result',  # 저장 폴더: model/result
    exist_ok=True,
    show=False
)

#  결과 텍스트 저장
detected = results[0].boxes
with open(result_txt_path, 'w') as f:
    if detected is None or len(detected) == 0:
        f.write("NO_DEFECT\n")
        print(f"✅ {image_name} : 결함 없음")
    else:
        f.write("DEFECT_DETECTED\n")
        for box in detected.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            f.write(f"{x1},{y1},{x2},{y2}\n")
        print(f"⚠️ {image_name} : 결함 탐지됨 ({len(detected)})")

print(f"\n📝 텍스트 저장 위치: {result_txt_path}")
print(f"🖼️ 결과 이미지 저장 위치: model/result/{image_name} (자동 저장됨)")
