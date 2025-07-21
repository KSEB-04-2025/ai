# 🔍 결함 탐지 AI (YOLOv8s 기반)

이 프로젝트는 학습된 YOLOv8s 모델(`best.pt`)을 사용하여  
단일 이미지 또는 이미지 폴더에서 **결함(예: 스크래치)**을 탐지합니다.  
탐지 결과는 시각화된 이미지와 함께 텍스트 파일로 저장됩니다.

---

## 📁 폴더 구조

| 경로 | 설명 |
|------|------|
| `model/best.pt` | 학습된 YOLOv8s 결함 탐지 모델 |
| `predict.py` | 단일 이미지 또는 폴더 예측 코드 |
| `predict_cam.py` | 실시간 웹캠 결함 감지 |
| `model/result/` | 예측된 이미지와 텍스트 저장 경로 |
| `test_images/` | 테스트 이미지 폴더 (선택) |


---

## ⚙️ 설치 방법

```bash
pip install ultralytics opencv-python
