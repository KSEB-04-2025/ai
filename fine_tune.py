import os
from ultralytics import YOLO


BASE_WEIGHT = r"C:\Users\happy\OneDrive\Desktop\yolov8n\runs\baseline\baseline_y8n3\weights\best.pt"
DATA_YAML   = r"new_data/data.yaml"

PROJECT     = "runs/fine_tune"
STAGE1_NAME = "stage1_freeze10"
STAGE2_NAME = "stage2_freeze0"

IMG_SIZE    = 896
DEVICE      = 0         
BATCH       = 4
EPOCHS_S1   = 30
EPOCHS_S2   = 100


def train_stage1():
    model = YOLO(BASE_WEIGHT)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS_S1,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        lr0=1e-3,
        freeze=10,
        val=True,
        cos_lr=True,
        mixup=0.1,
        mosaic=1.0,
        patience=20,
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
        lr0=3e-4,
        freeze=0,
        val=True,
        cos_lr=True,
        mixup=0.2,
        mosaic=1.0,
        patience=30,
        project=PROJECT,
        name=STAGE2_NAME,
        seed=0
    )
    return os.path.join(PROJECT, STAGE2_NAME, "weights", "best.pt")


if __name__ == "__main__":
    print(" Stage 1 시작 (freeze=10)")
    stage1_best = train_stage1()

    if not os.path.exists(stage1_best):
        raise FileNotFoundError(" Stage 1 best.pt 없음")

    print(" Stage 2 시작 (freeze=0)")
    stage2_best = train_stage2(stage1_best)

    if not os.path.exists(stage2_best):
        raise FileNotFoundError("Stage 2 best.pt 없음")

    print(f"\n✅ 파인튜닝 완료! 최종 모델 경로: {stage2_best}")
