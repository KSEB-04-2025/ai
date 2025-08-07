import os
import glob
import shutil
from PIL import Image
from ultralytics import YOLO



BASE_MODEL_PATH = r"model/v4/best.pt"
INCOMING_IMAGE_DIR = r"incoming_data"
RETRAIN_IMAGE_DIR = r"retrain_dataset/images/train"
RETRAIN_LABEL_DIR = r"retrain_dataset/labels/train"
DATA_YAML_PATH = r"retrain_dataset/data.yaml"
RETRAIN_PROJECT = "runs/retrain"
RETRAIN_NAME = "retrained_model"
MIN_IMAGES_FOR_RETRAIN = 100
DEVICE = 0
IMG_SIZE = 896
EPOCHS = 50
BATCH = 8


def auto_label_images(model_path, image_dir, label_save_dir):
    print("🔁 자동 라벨링 시작...")
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
                    xywhn = box.xywhn[0].tolist()
                    line = f"{cls} {' '.join([f'{x:.6f}' for x in xywhn])}\n"
                    f.write(line)

    print(f" 라벨 자동 생성 완료: {len(image_paths)}장")


def move_images_to_retrain_dataset(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for img_path in glob.glob(os.path.join(src_dir, "*.jpg")):
        shutil.copy(img_path, dst_dir)
    print(" 이미지 복사 완료")


def retrain_model(model_path):
    print(" 재학습 시작...")
    model = YOLO(model_path)
    model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        val=True,
        name=RETRAIN_NAME,
        project=RETRAIN_PROJECT
    )

    new_best = os.path.join(RETRAIN_PROJECT, RETRAIN_NAME, "weights", "best.pt")
    print(f" 재학습 완료: {new_best}")
    return new_best


def replace_best_model(new_model_path):
    dst = BASE_MODEL_PATH
    shutil.copy(new_model_path, dst)
    print(f" 기존 best.pt 갱신됨: {dst}")


def main():
    incoming_images = glob.glob(os.path.join(INCOMING_IMAGE_DIR, "*.jpg"))
    if len(incoming_images) < MIN_IMAGES_FOR_RETRAIN:
        print(f" 현재 {len(incoming_images)}장 - 100장 이상일 때만 재학습 진행")
        return

    # 1. 자동 라벨링
    auto_label_images(BASE_MODEL_PATH, INCOMING_IMAGE_DIR, RETRAIN_LABEL_DIR)

    # 2. 이미지 복사
    move_images_to_retrain_dataset(INCOMING_IMAGE_DIR, RETRAIN_IMAGE_DIR)

    # 3. 재학습 수행
    new_model = retrain_model(BASE_MODEL_PATH)

    # 4. 기존 best.pt 교체
    replace_best_model(new_model)

    print(" 파이프라인 완료!")


if __name__ == "__main__":
    main()
