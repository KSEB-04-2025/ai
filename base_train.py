import os
import shutil
from ultralytics import YOLO

BASE_MODEL = "yolov8n.pt"
DATA_YAML = "dataset/data.yaml"
PROJECT = "runs/baseline"
EXPERIMENT_NAME = "baseline_y8n"
DEVICE = 0

EPOCHS = 80
BATCH = 8
IMG_SIZE = 640

def train_baseline():
    model = YOLO(BASE_MODEL)
    
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        val=True,
        project=PROJECT,
        name=EXPERIMENT_NAME,
        seed=42
    )

    best_model_path = os.path.join(PROJECT, EXPERIMENT_NAME, "weights", "best.pt")
    baseline_dir = "baseline"
    baseline_model_path = os.path.join(baseline_dir, "baseline.pt")

    os.makedirs(baseline_dir, exist_ok=True)
    shutil.copy(best_model_path, baseline_model_path)

    print(f"\n YOLO best.pt 저장 위치: {best_model_path}")
    print(f" baseline 기준 모델 복사 위치: {baseline_model_path}")

if __name__ == "__main__":
    train_baseline()
