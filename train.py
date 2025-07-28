import os
import glob
import pandas as pd
from ultralytics import YOLO


BASE_WEIGHT = r"C:\Users\USER\Downloads\ai-feat-2-ai-yolotestmodel\model\weights\best.pt"  
DATA_YAML   = r"C:\Users\USER\Downloads\ai-feat-2-ai-yolotestmodel\dataset\data.yaml"

PROJECT     = "runs/film_ft"
STAGE1_NAME = "stage1_freeze10"
STAGE2_NAME = "stage2_freeze0"

EVAL_CONF   = 0.05   # Recall 측정시 낮게
IMG_SZ_TR   = 896    # 작은 스크래치 탐지 향상 목적
EPOCHS_S1   = 20     # Stage1: 빠르게 헤드만 적응
EPOCHS_S2   = 100    # Stage2: 전체 미세조정
BATCH       = 16
DEVICE      = 0      


# 유틸

def latest_best_weight(project_dir: str) -> str | None:
    paths = glob.glob(os.path.join(project_dir, "*", "weights", "best.pt"))
    return max(paths, key=os.path.getctime) if paths else None

def evaluate_model(model_path: str, split="test", conf=EVAL_CONF):
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    model = YOLO(model_path)
    m = model.val(
        data=DATA_YAML,
        split=split,
        imgsz=IMG_SZ_TR,
        device=DEVICE,
        conf=conf
    )
    return {
        "model": model_path,
        "split": split,
        "mAP50": m.box.map50,
        "mAP50-95": m.box.map,
        "precision": m.box.mp,
        "recall": m.box.mr
    }


# 1) Stage 1: freeze=10 (백본 대부분 동결, 헤드만 적응)

def finetune_stage1():
    model = YOLO(BASE_WEIGHT)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS_S1,
        imgsz=IMG_SZ_TR,
        batch=BATCH,
        device=DEVICE,
        lr0=1e-3,
        freeze=10,
        cos_lr=True,
        mosaic=1.0,
        mixup=0.1,
        project=PROJECT,
        name=STAGE1_NAME,
        val=True,
        patience=20,
        seed=0
    )
    return os.path.join(PROJECT, STAGE1_NAME, "weights", "best.pt")


# 2) Stage 2: freeze=0 (전체 레이어 미세조정)

def finetune_stage2(stage1_best: str):
    model = YOLO(stage1_best)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS_S2,
        imgsz=IMG_SZ_TR,
        batch=BATCH,
        device=DEVICE,
        lr0=3e-4,        # 전체 학습이므로 더 낮게
        freeze=0,
        cos_lr=True,
        mosaic=1.0,
        mixup=0.2,
        project=PROJECT,
        name=STAGE2_NAME,
        val=True,
        patience=30,
        seed=0
    )
    return os.path.join(PROJECT, STAGE2_NAME, "weights", "best.pt")


# main

if __name__ == "__main__":
    print("[STEP 0] Evaluate baseline on TEST ...")
    baseline_res = evaluate_model(BASE_WEIGHT, split="test", conf=EVAL_CONF)

    print("[STEP 1] Finetune Stage1 (freeze=10) ...")
    s1_best = finetune_stage1()
    if s1_best is None or not os.path.exists(s1_best):
        raise FileNotFoundError("Stage1 best.pt 를 찾을 수 없습니다.")

    print("[STEP 2] Finetune Stage2 (freeze=0) ...")
    s2_best = finetune_stage2(s1_best)
    if s2_best is None or not os.path.exists(s2_best):
        raise FileNotFoundError("Stage2 best.pt 를 찾을 수 없습니다.")

    print("[STEP 3] Evaluate finetuned model on TEST ...")
    finetuned_res = evaluate_model(s2_best, split="test", conf=EVAL_CONF)

    df = pd.DataFrame([baseline_res, finetuned_res])
    print(df)
    out_csv = "model_comparison_film_only.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[DONE] 결과 저장: {out_csv}")
    print(f"Stage2 best weight: {s2_best}")
