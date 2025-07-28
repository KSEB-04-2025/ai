
## 🚀 프로젝트 실행 가이드 (YOLOv8 기반 결함 탐지)

### 📦 1. 필수 설치 환경

#### ✅ Python 버전

* Python 3.8 \~ 3.11 권장
* 가상환경 권장

```bash
# 가상환경 생성 및 실행 (선택)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

---

#### ✅ 필수 패키지 설치

YOLOv8 및 기타 의존성 설치:

```bash
pip install --upgrade pip
pip install ultralytics opencv-python
```

필요한 경우:

```bash
pip install numpy pandas matplotlib
```

> 또는 `requirements.txt`가 있을 경우:

```bash
pip install -r requirements.txt
```

---

### 📁 2. 모델 파일 설명

| 역할        | 경로                      | 설명                        |
| --------- | ----------------------- | ------------------------- |
| 파인튜닝 모델 | `model/v2/best.pt`      | 학습된 스크래치 결함 탐지 YOLOv8n 모델 |
| 베이스라인  | `model/weights/best.pt` | 초기 YOLOv8n 베이스 모델         |



---

###  3. 예측 실행 방법

####  (1) 폴더 전체 예측

```bash
python predict.py --model model/v2/best.pt --img_dir dataset/images/test
```

예측 결과는 다음 경로에 저장됩니다:

```
runs/predict_results/test_predictions/
├── image.jpg
└── labels/image.txt
```

---

####  (2) 단일 이미지 예측

```bash
python predict.py --model model/v2/best.pt --img_dir test_img/film_1.jpg
```

> `--img_dir` 인자에 **단일 이미지 경로를 넣으면 자동 처리**됩니다.

---

####  (선택) 예측 세부 설정 옵션

| 옵션          | 설명                                    |
| ----------- | ------------------------------------- |
| `--conf`    | confidence threshold (default: `0.3`) |
| `--imgsz`   | 입력 이미지 크기 (default: `896`)            |
| `--out_dir` | 결과 저장 루트 경로 (default: `runs/...`)     |
| `--name`    | 결과 저장 하위 폴더명                          |
| `--device`  | GPU 번호 또는 `cpu`                       |

예시:

```bash
python predict.py --model model/v2/best.pt --img_dir dataset/images/test --name scratch_test --conf 0.3 --device 0
```

---

### 📝 4. 프로젝트 구조 예시

```
project-root/
├── model/
│   ├── v2/
│   │   └── best.pt
│   └── weights/
│       └── best.pt
├── dataset/
│   └── images/test/
├── predict.py
├── train.py
└── ...
```

