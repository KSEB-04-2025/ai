# Python FastAPI Defect Classification API

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-green.svg)](https://ultralytics.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-4.0%2B-4EA94B.svg)](https://www.mongodb.com/)
[![Google Cloud Storage](https://img.shields.io/badge/Google_Cloud_Storage-blue.svg)](https://cloud.google.com/storage)
[![Docker](https://img.shields.io/badge/Docker-v3.8-blue.svg)](https://www.docker.com/)

---

## 프로젝트 개요

본 프로젝트는 Python과 FastAPI를 기반으로 구축된 이미지 결함 분류 API 서버입니다. 업로드된 이미지에 대해 YOLO 모델을 사용하여 결함을 감지하고, 그 결과를 MongoDB에 저장하며, 원본 및 주석이 달린 이미지를 Google Cloud Storage(GCS)에 업로드합니다. 또한, GCS에서 최신 모델을 자동으로 감지하고 핫 리로드하는 기능을 포함하고 있습니다.

## ✨ 주요 기능

-   **이미지 업로드 및 결함 예측**: 이미지를 업로드하면 YOLO 모델을 통해 결함을 감지하고 예측 결과를 반환합니다.
-   **자동 모델 관리**: Google Cloud Storage에 저장된 최신 YOLO 모델을 자동으로 다운로드하고 애플리케이션 재시작 없이 핫 리로드합니다.
-   **결과 저장**: 결함 감지 시 예측 결과(결함 여부, 바운딩 박스, 신뢰도 등)를 MongoDB에 저장합니다.
-   **이미지 저장**: 원본 이미지(재학습용)와 결함이 표시된 주석 이미지(annotated image)를 Google Cloud Storage에 업로드합니다.
-   **조건부 저장**: 결함이 감지된 경우에만 결과 및 이미지를 저장합니다.
-   **상태 확인 엔드포인트**: `/health` 엔드포인트를 통해 애플리케이션의 상태를 확인할 수 있습니다.

## ️ 기술 스택

| 구분 | 기술 | 버전/설명 |
| --- | --- | --- |
| **언어** | Python | 3.9+ |
| **웹 프레임워크** | FastAPI | |
| **머신러닝** | Ultralytics YOLO | |
| **데이터베이스** | MongoDB | |
| **클라우드 스토리지** | Google Cloud Storage (GCS) | |
| **이미지 처리** | Pillow, OpenCV | |
| **환경 변수 관리** | python-dotenv | |
| **컨테이너** | Docker | |

## API 엔드포인트

애플리케이션이 실행되면, 아래 엔드포인트를 통해 API를 사용할 수 있습니다.

-   **Health Check**: `/health` (GET)
-   **Defect Classification**: `/defect/` (POST)
    -   이미지 파일을 `multipart/form-data` 형태로 업로드하여 결함 예측을 수행합니다.

## ⚙️ 환경 설정

`.env` 파일을 생성하여 다음 환경 변수를 설정해야 합니다:

```
MONGO_URI="mongodb://localhost:27017/"
GCS_BUCKET="your-gcs-bucket-name"
GCS_FOLDER="defect"
GCS_RETRAIN_FOLDER="retrain"
GCS_MODEL_FOLDER="model"
MODEL_CHECK_INTERVAL="300" # seconds (default 5 minutes)
```

-   `MONGO_URI`: MongoDB 연결 URI.
-   `GCS_BUCKET`: Google Cloud Storage 버킷 이름.
-   `GCS_FOLDER`: 결함 감지된 이미지가 저장될 GCS 폴더.
-   `GCS_RETRAIN_FOLDER`: 재학습용 원본 이미지가 저장될 GCS 폴더.
-   `GCS_MODEL_FOLDER`: YOLO 모델 파일이 저장된 GCS 폴더.
-   `MODEL_CHECK_INTERVAL`: GCS에서 모델 업데이트를 확인할 주기 (초 단위).

Google Cloud Storage 연동을 위해 `service-account.json` 파일이 필요하며, 이 파일은 컨테이너 내 `/app/service-account.json` 경로에 마운트되어야 합니다.
