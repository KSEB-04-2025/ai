from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone, timedelta
from ultralytics import YOLO
from PIL import Image
from pymongo import MongoClient
from dotenv import load_dotenv
from uuid import uuid4
from google.cloud import storage
from pathlib import Path
import os, io, threading, time
import cv2

# ───────── ENV ─────────
load_dotenv()
MONGO_URI            = os.getenv("MONGO_URI")
GCS_BUCKET           = os.getenv("GCS_BUCKET")                 # e.g., "zezeone_images"
GCS_FOLDER           = os.getenv("GCS_FOLDER")                 # e.g., "defect"
GCS_RETRAIN_FOLDER   = os.getenv("GCS_RETRAIN_FOLDER")         # e.g., "retrain"
GCS_MODEL_FOLDER     = os.getenv("GCS_MODEL_FOLDER", "model")  # e.g., "model"
GCS_KEY_PATH         = "service-account.json"                  # 컨테이너 내 마운트 경로
MODEL_CHECK_INTERVAL = int(os.getenv("MODEL_CHECK_INTERVAL", "300"))  # 초(기본 5분)

if not MONGO_URI:
    raise RuntimeError("MONGO_URI 환경변수가 없습니다.")
if not GCS_BUCKET:
    raise RuntimeError("GCS_BUCKET 환경변수가 없습니다.")

# ───────── 경로/상수 ─────────
MODEL_LOCAL_PATH = "/app/model/best.pt"
MODEL_LOCAL = Path(MODEL_LOCAL_PATH)
MODEL_LOCAL.parent.mkdir(parents=True, exist_ok=True)
GCS_BLOB_PATH = f"{GCS_MODEL_FOLDER.rstrip('/')}/best.pt"

# ───────── 앱 & 클라이언트 ─────────
app = FastAPI()
print("🔥 main.py — auto-download (GCS) + auto-reload (mtime)")

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

mongo_client = MongoClient(MONGO_URI)
mongo_col = mongo_client["zezeone"]["results"]

storage_client = storage.Client.from_service_account_json(GCS_KEY_PATH)
bucket = storage_client.bucket(GCS_BUCKET)

# ───────── 전역 모델 상태 (요청 시 mtime 자동 감지/핫리로드) ─────────
try:
    model = YOLO(MODEL_LOCAL_PATH)
    model_mtime = os.path.getmtime(MODEL_LOCAL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"[MODEL] 파일이 없습니다: {MODEL_LOCAL_PATH}")

model_lock = threading.Lock()

def get_model():
    """요청 시마다 로컬 모델 파일 mtime 확인 → 바뀌면 안전하게 핫리로드"""
    global model, model_mtime
    try:
        mtime = os.path.getmtime(MODEL_LOCAL_PATH)
    except FileNotFoundError:
        # 교체 중(.part)일 수 있음 → 현재 모델 유지
        return model
    if mtime != model_mtime:
        with model_lock:
            try:
                m2 = os.path.getmtime(MODEL_LOCAL_PATH)
            except FileNotFoundError:
                return model
            if m2 != model_mtime:
                model = YOLO(MODEL_LOCAL_PATH)
                model_mtime = m2
                print(f"[MODEL] auto-reloaded at {time.strftime('%F %T')}")
    return model

# ───────── GCS 최신 감지 + 자동 다운로드(원자 교체) 백그라운드 워커 ─────────
def get_local_mtime_utc():
    try:
        ts = MODEL_LOCAL.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except FileNotFoundError:
        return None

def download_best_pt_atomic():
    blob = bucket.blob(GCS_BLOB_PATH)
    tmp = MODEL_LOCAL.with_suffix(".part")
    if tmp.exists():
        tmp.unlink()
    blob.download_to_filename(tmp.as_posix())
    tmp.replace(MODEL_LOCAL)  # 원자 교체
    print(f"[MODEL] downloaded and replaced: {MODEL_LOCAL}")

def watch_gcs_model_loop():
    print(f"[MODEL] watcher started (interval={MODEL_CHECK_INTERVAL}s)")
    while True:
        try:
            blob = bucket.blob(GCS_BLOB_PATH)
            if not blob.exists(storage_client):
                print(f"[MODEL] not found: gs://{GCS_BUCKET}/{GCS_BLOB_PATH}")
            else:
                blob.reload()  # 최신 메타데이터
                remote_updated = blob.updated  # tz-aware UTC
                local_mtime = get_local_mtime_utc()

                need = (local_mtime is None) or (remote_updated and local_mtime < remote_updated)
                if need:
                    print(f"[MODEL] remote newer → downloading (local={local_mtime}, remote={remote_updated})")
                    download_best_pt_atomic()
                # else: up-to-date
        except Exception as e:
            # 네트워크/권한 이슈는 다음 사이클에 재시도
            print(f"[MODEL] watcher error: {e}")
        time.sleep(MODEL_CHECK_INTERVAL)

# 앱 기동 후 워커 시작 (데몬 스레드)
threading.Thread(target=watch_gcs_model_loop, daemon=True).start()

# ───────── Health ─────────
@app.head("/health")
def health():
    return {"status": "ok"}

# ───────── 이미지 업로드 + 예측 ─────────
@app.post("/defect/", summary="Defect Classification")
async def upload_and_predict(request: Request, file: UploadFile = File(...)):
    now = datetime.now()
    filename = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex}.png"
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

    # YOLO 예측 (현재 모델 핸들 확보)
    try:
        mdl = get_model()
        results = mdl(img, conf=0.44, iou=0.3)
    except Exception as e:
        print(f"[YOLO 예측 실패] {e}")
        return JSONResponse(status_code=500, content={"message": f"YOLO 예측 실패: {e}"})

    # 예측 결과 파싱
    predictions = []
    for result in results:
        if result.boxes is None:
            continue
        boxes = result.boxes.cpu().numpy()
        print(f"[INFO] YOLO 결과 → box 수: {len(boxes)}")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            predictions.append({
                "class_id": cls,
                "class_name": mdl.names[cls],
                "confidence": conf,
                "bbox": xyxy
            })

    # 결함 여부 판단
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

    # Annotated 이미지 저장 및 업로드
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

            # GCS 업로드 (annotated)
            blob = bucket.blob(f"{GCS_FOLDER}/{filename}")
            blob.upload_from_filename(annotated_path)
            gcs_url = f"https://storage.googleapis.com/{GCS_BUCKET}/{blob.name}"
    except Exception as e:
        print("[경고] Annotated 이미지 저장 또는 GCS 업로드 실패:", e)

    # 원본 이미지 GCS 업로드 (재학습용)
    try:
        retrain_blob = bucket.blob(f"{GCS_RETRAIN_FOLDER}/{filename}")
        retrain_blob.upload_from_filename(file_path)
        print(f"[INFO] 원본 이미지 {GCS_RETRAIN_FOLDER}에 업로드 완료")
    except Exception as e:
        print(f"[ERROR] 원본 이미지 업로드 실패: {e}")

    # 한국시간 KST
    KST = timezone(timedelta(hours=9))
    now_kst = datetime.now(KST)

    # MongoDB 저장
    count = mongo_col.count_documents({})
    mongo_doc_id = f"defect_{count + 1}"
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

    return JSONResponse(content={
        "message": "결함 감지: DB 및 GCS 저장 완료",
        "saved_filename": filename,
        "document_id": str(inserted_id),
        "label": label,
        "n_spots": len(predictions),
        "gcs_url": gcs_url,
        "predictions": predictions
    })
