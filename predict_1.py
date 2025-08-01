from ultralytics import YOLO
import os

def predict_images():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = YOLO(os.path.join(base_dir, 'runs', 'finetune', 'film_scratch_18', 'weights', 'best.pt'))
    # 추론 대상 이미지 폴더
    test_dir = os.path.join(base_dir, 'normaldata')

    results = model.predict(
        source=test_dir,
        conf=0.25,      # 기본 0.25 → 낮출수록 recall↑ precision↓, 0.1~0.2 추천
        iou=0.5,        # 필요시 0.4~0.6 조정
        save=True,      # 결과 이미지 저장
        save_txt=True   # 결과 라벨 txt도 저장
    )
    print("Prediction finished!")
    return results

if __name__ == "__main__":
    predict_images()
