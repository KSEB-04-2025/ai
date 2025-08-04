# ai

# 개요

일반 결함 데이터(398장)와 난반사 필름 데이터(102장)를 결합하여 학습한 YOLO 기반 baseline 모델의 성능을 정리합니다.

# 데이터 구성

- 일반 결함 이미지: 398장

- 난반사 필름 이미지: 102장

# Inference 설정

    confidence threshold (conf): 0.25
    IoU threshold (iou): 0.45

- `conf`와 `iou` 값은 추론 시 기본 설정이며, 필요에 따라 조정하여 성능을 최적화할 수 있습니다.

# 결과

    Validation Recall: 0.89

# 사용방법

    yolo detect predict model=runs/train/baseline_full_finetune/weights/best.pt source=film_infer/images conf=0.25 iou=0.45

결과는 `runs/detect/predict` 폴더에 저장됩니다.
