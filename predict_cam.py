from ultralytics import YOLO
import cv2

# ✅ 모델 로드
model = YOLO("model/best.pt")

# ✅ 웹캠 열기 (0 = 기본 카메라)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break

    # ✅ 추론 수행 (BGR → RGB는 내부에서 처리됨)
    results = model.predict(
        source=frame,
        conf=0.3,
        imgsz=640,
        device='cuda',
        verbose=False
    )

    # ✅ 결과 시각화 (bounding box 포함)
    annotated_frame = results[0].plot()

    # ✅ 화면에 표시
    cv2.imshow("YOLOv8 - 실시간 결함 감지", annotated_frame)

    # ❌ 'q' 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ 정리
cap.release()
cv2.destroyAllWindows()
