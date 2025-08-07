from ultralytics import YOLO
from pathlib import Path
import shutil

# 경로 설정
INCOMING_DIR = Path("incoming_data")  
TRAIN_IMAGE_DIR = Path("retrain_dataset/images/train")
LABEL_SAVE_DIR = Path("autolabeled_data/labels")
DATA_YAML_PATH = Path("retrain_dataset/data.yaml")

# 1️ 이전 학습 이미지 제거 후 폴더 생성
if TRAIN_IMAGE_DIR.exists():
    shutil.rmtree(TRAIN_IMAGE_DIR)
TRAIN_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# 2️ 이미지 복사 (원본 이미지 → 학습용 폴더)
for img_path in INCOMING_DIR.glob("*.jpg"):
    shutil.copy(img_path, TRAIN_IMAGE_DIR / img_path.name)

# 3️ 자동 라벨링
model = YOLO("model/v4/best.pt")

model.predict(
    source=str(TRAIN_IMAGE_DIR),
    save_txt=True,
    save_conf=True,
    project="autolabeled_data",
    name="labels",
    conf=0.44,
    iou=0.3,
    imgsz=896
)

# 4️ data.yaml 생성
DATA_YAML_PATH.write_text(f"""\  
path: retrain_dataset
train: images/train
val: images/train
names:
  0: defect
""")

# 5️ 재학습 실행
model.train(
    data=str(DATA_YAML_PATH),
    epochs=50,
    imgsz=896,
    project="model_retrain_runs",
    name="retrained_model",
    device=0
)

# 6️ 모델 파일 best.pt → 기존 위치로 덮어쓰기
trained_model_path = Path("model_retrain_runs/retrained_model/weights/best.pt")
final_model_path = Path("model/v4/best.pt")
final_model_path.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(trained_model_path, final_model_path)
print("✅ 모델 업데이트 완료:", final_model_path)
