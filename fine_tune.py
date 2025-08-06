import os
from ultralytics import YOLO

BASE_WEIGHT = r"C:\Users\happy\OneDrive\Desktop\yolov8n\runs\baseline\baseline_y8n7\weights\best.pt"
DATA_YAML   = r"new_data/data.yaml"

PROJECT     = "runs/fine_tune"
STAGE1_NAME = "stage1_freeze10"
STAGE2_NAME = "stage2_freeze0"

IMG_SIZE    = 640
DEVICE      = 0
BATCH       = 4
EPOCHS_S1   = 30
EPOCHS_S2   = 80


def train_stage1():
    model = YOLO(BASE_WEIGHT)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS_S1,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        freeze=10,           # Backbone freeze
        val=True,
        project=PROJECT,
        name=STAGE1_NAME,
        seed=0
    )
    return os.path.join(PROJECT, STAGE1_NAME, "weights", "best.pt")


def train_stage2(stage1_best):
    model = YOLO(stage1_best)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS_S2,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        lr0=3e-4,             # 낮은 학습률로 정밀 튜닝
        freeze=0,             # 전체 학습
        val=True,
        mosaic=0.8,           # 약한 augmentation
        mixup=0.1,            # 너무 높으면 과적합 방지 못함
        patience=20,          # early stopping 빠르게
        cos_lr=True,          # smooth하게 decay
        project=PROJECT,
        name=STAGE2_NAME,
        seed=0
    )
    return os.path.join(PROJECT, STAGE2_NAME, "weights", "best.pt")


if __name__ == "__main__":
    print("📌 Stage 1 시작 (freeze=10)")
    stage1_best = train_stage1()

    if not os.path.exists(stage1_best):
        raise FileNotFoundError("❌ Stage 1 best.pt 없음")

    print("📌 Stage 2 시작 (freeze=0)")
    stage2_best = train_stage2(stage1_best)

    if not os.path.exists(stage2_best):
        raise FileNotFoundError("❌ Stage 2 best.pt 없음")

    print(f"\n✅ 파인튜닝 완료! 최종 모델 경로: {stage2_best}")
