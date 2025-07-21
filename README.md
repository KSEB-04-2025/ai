# ai

스마트폰 액정 스크래치 데이터(398 장)로 학습한 **baseline 모델** 가중치(`best.pt`) 입니다.
따로 Finetune 학습을 하지 않았습니다.

---

## 💾 파일 구조

AI/
├─ model/
│ └─ weights/
│ └─ best.pt # ← 이 리포지토리의 핵심 파일
├─ infer/
│ └─ images/ # 예시 필름 스크래치 이미지

> 모델만 필요하시면은 `model/weights/best.pt`만 받아가셔도 됩니다.

## 🚀 한줄추론 코드

yolo detect predict \
model=model/weights/best.pt \
source=film_infer/images \ # ← 추론할 이미지/폴더 경로
save save_txt save_conf save_crop \
project=runs_film \
name=baseline_pred

## 웹캠

yolo detect predict model=model/weights/best.pt source=0 show=True

## 🛠 환경 (tested)

- **Python 3.11.3**
- **Ultralytics YOLOv8 v8.3.162** (install: `pip install ultralytics==8.3.162`)
