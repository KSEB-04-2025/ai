# 🔍 결함 탐지 AI 시스템 (YOLOv8n 기반)

본 프로젝트는 **YOLOv8n 모델**을 기반으로 결함(예: 스크래치)을 탐지하는 시스템입니다.  
단일 이미지 예측, 실시간 웹캠 감지, 그리고 GCS에서 100장 이상의 이미지가 수집되면 자동 라벨링 및 재학습까지 수행하는 **자동화된 재학습 파이프라인**을 포함합니다.

---

## 📁 폴더 구조

| 경로 | 설명 |
|------|------|
| `model/best.pt` | 최신 YOLOv8 결함 탐지 모델 |
| `model/result/` | 예측 결과 이미지 및 `.txt` 저장 경로 |
| `incoming_data/` | GCS에서 복사된 원본 이미지 저장 폴더 |
| `autolabeled_data/labels/` | 자동 생성된 라벨 (`.txt`) 저장 경로 |
| `autolabeled_data/visualized/` | 시각화된 결과 이미지 저장 경로 |
| `retrain_dataset/` | 재학습을 위한 데이터셋 (images/train, labels/train) |
| `auto_label_and_retrain.py` | 자동 라벨링 + 재학습 통합 파이프라인 |
| `server.py` or `main.py` | FastAPI 기반 서버, 예측 및 GCS 업로드 처리 |
| `Cloud Function` | GCS 트리거 기반 이미지 수 감지 및 서버 `/retrain` 호출 |

---

## ⚙️ 설치

```bash
pip install ultralytics opencv-python fastapi python-multipart python-dotenv pillow google-cloud-storage pymongo
