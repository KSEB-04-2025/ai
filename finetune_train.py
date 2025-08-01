from ultralytics import YOLO
import os

def main():
    print("==== Start finetune_train.py ====")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("base_dir:", base_dir)
    baseline_ckpt = os.path.join(base_dir, 'model', 'weights_1', 'best.pt')
    print("baseline_ckpt:", baseline_ckpt)
    print("Check if best.pt exists:", os.path.exists(baseline_ckpt))
    model = YOLO(baseline_ckpt)
    print("YOLO model loaded!")

    data_path = os.path.join(base_dir, 'film_scratch', 'data.yaml')
    print("data_path:", data_path)
    print("Check if data.yaml exists:", os.path.exists(data_path))

    print("Start training...")
    model.train(
        data=data_path,
        epochs=100,
        batch=8,
        imgsz=768,
        lr0=0.0008,
        lrf=0.001,
        cos_lr=True,
        mosaic=0.1,
        copy_paste=0.02,
        erasing=0.15,
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.7,
        fliplr=0.2,
        flipud=0.08,
        translate=0.07,
        degrees=5,
        scale=0.08,
        perspective=0.01,
        iou=0.6,
        weight_decay=0.0008,
        dropout=0.13,
        patience=20,
        project=os.path.join(base_dir, 'runs', 'finetune'),
        name='film_scratch_1'
    )
    print("Training called.")

if __name__ == '__main__':
    print("Script started by __main__")
    main()
    print("Script finished")
