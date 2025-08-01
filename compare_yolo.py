import csv
from ultralytics import YOLO

def compare_models():
    models = [
        ("Pretrained (액정)", "model/weights/best.pt"),
        ("Finetuned (필름)", "runs/detect/train4/weights/best.pt"),
    ]
    data = "film_scratch/data.yaml"
    results = []

    for label, weight_path in models:
        model = YOLO(weight_path)
        r = model.val(data=data, split="test")
        results.append({
            "Model": label,
            "Precision": float(r.box.mp),
            "Recall": float(r.box.mr),
            "mAP50": float(r.box.map50),
            "mAP50-95": float(r.box.map),
        })

    # CSV 저장
    with open('model_comparison.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("비교 결과가 'model_comparison.csv'로 저장되었습니다!")

if __name__ == "__main__":
    compare_models()
