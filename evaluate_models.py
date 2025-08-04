import os
from ultralytics import YOLO

# 경로 설정
DATA_YAML = r"new_data/data.yaml"
IMG_SIZE = 896
DEVICE = 0

BASELINE_MODEL = r"C:\Users\happy\OneDrive\Desktop\yolov8n\runs\baseline\baseline_y8n3\weights\best.pt"
FINETUNED_MODEL = r"C:\Users\happy\OneDrive\Desktop\yolov8n\runs\fine_tune\stage2_freeze03\weights\best.pt"

def evaluate_model(model_path, model_name):
    print(f"\n📊 [{model_name}] 평가 중...")

    model = YOLO(model_path)
    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        device=DEVICE,
        split='val'
    )

   
    recall = metrics.box.r.mean()
    print(f"✅ [{model_name}] Recall(R): {recall:.4f}")
    return recall

if __name__ == "__main__":
    baseline_recall = evaluate_model(BASELINE_MODEL, "Baseline")
    finetuned_recall = evaluate_model(FINETUNED_MODEL, "Fine-Tuned")

    print("\n📈 모델 리콜(Recall) 비교")
    print(f"📌 Baseline    : {baseline_recall:.4f}")
    print(f"📌 Fine-Tuned  : {finetuned_recall:.4f}")
    print("✅ 완료!")
