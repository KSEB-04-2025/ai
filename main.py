from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from datetime import datetime, timezone, timedelta
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
GCS_RETRAIN_FOLDER = os.getenv("GCS_RETRAIN_FOLDER")  # GCS 재학습 폴더 이름
GCS_KEY_PATH = "service-account.json"  # GCS 서비스 계정 키 파일 경로

if not MONGO_URI:
    raise RuntimeError("MONGO_URI 환경변수가 없습니다.")
if not GCS_BUCKET:
    raise RuntimeError("GCS_BUCKET 환경변수가 없습니다.")

# ───────── 기본 설정 ─────────
app = FastAPI()
model = YOLO("/app/model/best.pt")  # YOLOv8 모델 로드
print("🔥🔥🔥 This is the NEW main.py 🔥🔥🔥")  # <-- 여기에 삽입
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ───────── MongoDB 연결 ─────────
client = MongoClient(MONGO_URI)
mongo_col = client["zezeone"]["results"]

# ───────── GCS 클라이언트 ─────────
storage_client = storage.Client.from_service_account_json(GCS_KEY_PATH)
bucket = storage_client.bucket(GCS_BUCKET)

# ───────── Health ─────────
@app.head("/health")
def health():
    return {"status": "ok"}
    
# ───────── 이미지 업로드 + 예측 ─────────
@app.post("/defect/", summary = "Defect Classification3232")
async def upload_and_predict(request: Request, file: UploadFile = File(...)):
    now = datetime.now()
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex}.png"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # 이미지 저장
    image_bytes = await file.read()
    with open(file_path, "wb") as buffer:
        buffer.write(image_bytes)

    # 이미지 열기
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        print(f"[이미지 열기 실패] {e}")
        return JSONResponse(status_code=400, content={"message": f"이미지 열기 실패: {e}"})

    # YOLO 예측
    try:
        results = model(img, conf=0.44, iou=0.3)
    except Exception as e:
        print(f"[YOLO 예측 실패] {e}")
        return JSONResponse(status_code=500, content={"message": f"YOLO 예측 실패: {e}"})

    # 예측 결과 파싱
    predictions = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        print(f"[INFO] YOLO 결과 → box 수: {len(boxes)}")  # ← 여기가 핵심!
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
        
    print("DEBUG: class_names =", class_names)
    print("DEBUG: is_defective =", is_defective)
    print("DEBUG: predictions =", predictions)
    
    # 결함이 있는 경우만 아래 실행
    label = "X"

     # Annotated 이미지 저장 및 업로드 to GCS
    annotated_path = None
    gcs_url = None
    try:
        if hasattr(results[0], "plot"):
            annotated_img = results[0].plot()
        elif hasattr(results[0], "plot_result"):
            annotated_img = results[0].plot_result()
        else:
            print("[경고] plot() 사용 불가: ", type(results[0]))
            annotated_img = None

        if annotated_img is not None:
            annotated_path = file_path.replace("uploaded_images", "uploaded_images/annotated")
            os.makedirs(os.path.dirname(annotated_path), exist_ok=True)
            cv2.imwrite(annotated_path, annotated_img)

            # GCS 업로드
            blob = bucket.blob(f"{GCS_FOLDER}/{filename}")
            blob.upload_from_filename(annotated_path)
            gcs_url = f"https://storage.googleapis.com/{GCS_BUCKET}/{blob.name}"
    except Exception as e:
        print("[경고] Annotated 이미지 저장 또는 GCS 업로드 실패:", e)

    # 원본 이미지 GCS 업로드
    try:
        retrain_blob = bucket.blob(f"{GCS_RETRAIN_FOLDER}/{filename}")
        retrain_blob.upload_from_filename(file_path)
        print(f"[INFO] 원본 이미지 {GCS_RETRAIN_FOLDER}에 업로드 완료")
    except Exception as e:
        print(f"[ERROR] 원본 이미지 업로드 실패: {e}")
        
    # 한국시간 KST 생성
    KST = timezone(timedelta(hours=9))
    now_kst = datetime.now(KST)
    ts_ms   = int(now_kst.timestamp() * 1000)
    
    # 아이디 임의로 생성
    count = mongo_col.count_documents({})
    mongo_doc_id = f"defect_{count + 1}"
    
    # MongoDB 문서 구성
    mongo_doc = {
        "_id": mongo_doc_id,
        "label": label,
        "img_file_id": "defect/" + filename,
        "img_url": gcs_url,
        "uploadDate": now_kst,
        "date_time": now_kst,
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


