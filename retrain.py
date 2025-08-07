import os
import glob
from ultralytics import YOLO
from PIL import Image

BASE_MODEL = "model/v4/best.pt"
NEW_IMAGES_DIR = "new_data/images/collected"
NEW_LABELS_DIR = "new_data/labels/auto"
RETRAIN_DATASET_DIR = "retrain_dataset"
RETRAIN_MODEL_SAVE_DIR = "model/v4_retrained"


MIN_IMAGES = 100


def auto_label_images(model_path, image_dir, label_save_dir):
    model = YOLO(model_path)
    os.makedirs(label_save_dir, exist_ok=True)

    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    for img_path in image_paths:
        img = Image.open(img_path)
        results = model(img)

        for r in results:
            label_txt = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            label_path = os.path.join(label_save_dir, label_txt)
            with open(label_path, "w") as f:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    xywhn = box.xywhn[0].tolist()  # normalized
                    line = f"{cls} {' '.join([f'{x:.6f}' for x in xywhn])}\n"
                    f.write(line)

    print(f"✅ 라벨 자동 생성 완료: {len(image_paths)}장")


def run_retraining():
    # retrain dataset 디렉토리 준비
    os.makedirs(RETRAIN_DATASET_DIR, exist_ok=True)
    os.makedirs(RETRAIN_MODEL_SAVE_DIR, exist_ok=True)

    retrain_yaml_path = "new_data/data.yaml"  

    model = YOLO(BASE_MODEL)
    model.train(
        data=retrain_yaml_path,
        epochs=50,
        imgsz=896,
        batch=8,
        device=0,
        project=RETRAIN_MODEL_SAVE_DIR,
        name="retrain_result"
    )

    print("✅ 재학습 완료!")

# 3. 평가 및 리콜 비교
def evaluate_models(old_model_path, new_model_path):
    model_old = YOLO(old_model_path)
    model_new = YOLO(new_model_path)

    print("\n[기존 모델 평가]")
    old_metrics = model_old.val(data="new_data/data.yaml")
    print("\n[재학습 모델 평가]")
    new_metrics = model_new.val(data="new_data/data.yaml")

    old_recall = old_metrics.results_dict["recall"]
    new_recall = new_metrics.results_dict["recall"]

    print(f"\n📊 리콜 비교")
    print(f"📌 기존 모델: {old_recall:.4f}")
    print(f"📌 신규 모델: {new_recall:.4f}")

#  4. 메인 로직
def main():
    image_paths = glob.glob(os.path.join(NEW_IMAGES_DIR, "*.jpg"))
    if len(image_paths) >= MIN_IMAGES:
        print(f"🚀 신규 이미지 {len(image_paths)}장 → 재학습 시작")

        auto_label_images(BASE_MODEL, NEW_IMAGES_DIR, NEW_LABELS_DIR)
        run_retraining()
        evaluate_models(BASE_MODEL, os.path.join(RETRAIN_MODEL_SAVE_DIR, "retrain_result", "weights", "best.pt"))
    else:
        print(f"📌 현재 {len(image_paths)}장 수집됨. {MIN_IMAGES}장 이상일 때 재학습 진행됨.")

if __name__ == "__main__":
    main()
