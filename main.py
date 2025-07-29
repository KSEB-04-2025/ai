from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
from pymongo import MongoClient
from dotenv import load_dotenv
from uuid import uuid4
from google.cloud import storage
import os, io
import cv2

# ───────── 환경 변수 로딩 ─────────
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
GCS_BUCKET = os.getenv("GCS_BUCKET")  # GCS 버킷 이름
GCS_FOLDER = os.getenv("GCS_FOLDER")
GCS_KEY_PATH = "service_account.json"  # GCS 서비스 계정 키 파일 경로

if not MONGO_URI:
    raise RuntimeError("MONGO_URI 환경변수가 없습니다.")
if not GCS_BUCKET:
    raise RuntimeError("GCS_BUCKET 환경변수가 없습니다.")

# ───────── 기본 설정 ─────────
app = FastAPI()
model = YOLO("model/best.pt")  # YOLOv8 모델 로드
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ───────── MongoDB 연결 ─────────
client = MongoClient(MONGO_URI)
mongo_col = client["yolo_results"]["detections"]

# ───────── GCS 클라이언트 ─────────
storage_client = storage.Client.from_service_account_json(GCS_KEY_PATH)
bucket = storage_client.bucket(GCS_BUCKET)

# ───────── 이미지 업로드 + 예측 ─────────
@app.post("/defect/")
async def upload_and_predict(request: Request, file: UploadFile = File(...)):
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp_str}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # 이미지 저장
    image_bytes = await file.read()
    with open(file_path, "wb") as buffer:
        buffer.write(image_bytes)

    # 이미지 열기
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"이미지 열기 실패: {e}"})

    # YOLO 예측
    try:
        results = model(img)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"YOLO 예측 실패: {e}"})

    # 예측 결과 파싱
    predictions = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            predictions.append({
                "class_id": cls,
                "class_name": model.names[cls],
                "confidence": conf,
                "bbox": xyxy
            })

        # 결함 여부 판단 ("defect" 포함 시 X, 아니면 O)
    class_names = [pred["class_name"] for pred in predictions]
    is_defective = any("defect" in name.lower() for name in class_names)

    # 결함이 없는 경우: 저장 없이 응답만
    if not is_defective:
        return JSONResponse(content={
            "message": "정상: 결함 없음 (저장하지 않음)",
            "label": "O",
            "n_spots": len(predictions),
            "predictions": predictions
        })

    # 결함이 있는 경우만 아래 실행
    label = "X"

    # Annotated 이미지 저장 및 업로드 to GCS
    annotated_path = None
    gcs_url = None
    try:
        annotated_img = results[0].plot()
        annotated_path = file_path.replace("uploaded_images", "uploaded_images/annotated")
        os.makedirs(os.path.dirname(annotated_path), exist_ok=True)
        cv2.imwrite(annotated_path, annotated_img)

        # GCS 업로드
        blob = bucket.blob(f"{GCS_FOLDER}/{filename}")
        blob.upload_from_filename(annotated_path)
        blob.make_public()
        gcs_url = f"https://storage.googleapis.com/{GCS_BUCKET}/{blob.name}"
        if gcs_url is None:
            return JSONResponse(status_code=500, content={"message": "GCS 업로드 실패로 URL 생성 불가"})
    except Exception as e:
        print("[경고] Annotated 이미지 저장 또는 GCS 업로드 실패:", e)

    # MongoDB 문서 구성
    mongo_doc = {
        "label": label,
        "n_spots": len(predictions),
        "img_file_id": filename,
        "img_url": gcs_url,
        "uploadDate": now.isoformat(),
        "date_time": now.isoformat(),
        "client_ip": request.client.host
    }

    inserted_id = mongo_col.insert_one(mongo_doc).inserted_id

    # 응답
    return JSONResponse(content={
        "message": "결함 감지: DB 및 GCS 저장 완료",
        "saved_filename": filename,
        "document_id": str(inserted_id),
        "label": label,
        "n_spots": len(predictions),
        "gcs_url": gcs_url,
        "predictions": predictions
    })
