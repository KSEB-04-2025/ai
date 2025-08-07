from fastapi import FastAPI
from fastapi.responses import JSONResponse
from google.cloud import storage
from dotenv import load_dotenv
import os
import subprocess

# ✅ 환경변수 로드 (.env에서)
load_dotenv()

GCS_BUCKET = os.getenv("GCS_BUCKET")                     # 예: "zezeone_images"
GCS_RETRAIN_FOLDER = os.getenv("GCS_RETRAIN_FOLDER")     # 예: "retrain"
LOCAL_DATA = "incoming_data"                             # 로컬 다운로드 경로


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"


os.makedirs(LOCAL_DATA, exist_ok=True)


app = FastAPI()


@app.post("/retrain")
async def retrain_model():
    try:
        #  GCS 클라이언트
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET)

        #  GCS에서 retrain 폴더 내 전체 파일 가져오기
        blobs = list(bucket.list_blobs(prefix=f"{GCS_RETRAIN_FOLDER}/"))
        image_blobs = [b for b in blobs if b.name.lower().endswith((".png", ".jpg", ".jpeg"))]

        #  조건: 100장 이상일 때만 진행
        if len(image_blobs) < 100:
            return JSONResponse(content={
                "message": f"🟡 이미지 수 부족: {len(image_blobs)}장 (최소 100장 필요)"
            }, status_code=200)

        #  이미지 다운로드 → incoming_data/
        for blob in image_blobs:
            filename = os.path.basename(blob.name)
            dest_path = os.path.join(LOCAL_DATA, filename)
            blob.download_to_filename(dest_path)

        #  자동 라벨링 + 재학습 스크립트 실행
        subprocess.run(["python", "auto_label_and_retrain.py"], check=True)

        return JSONResponse(content={
            "message": f"✅ 총 {len(image_blobs)}장 재학습 완료"
        }, status_code=200)

    except subprocess.CalledProcessError as e:
        return JSONResponse(content={
            "error": "🚨 재학습 스크립트 실행 실패",
            "detail": str(e)
        }, status_code=500)

    except Exception as e:
        return JSONResponse(content={
            "error": "❌ 예외 발생",
            "detail": str(e)
        }, status_code=500)
