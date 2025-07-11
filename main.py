from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from datetime import datetime
import os

app = FastAPI()

UPLOAD_DIR = "uploaded_images"  # 이미지를 저장할 폴더명
os.makedirs(UPLOAD_DIR, exist_ok=True)  # 폴더 없으면 생성

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # 업로드 파일명에 현재 시각을 추가 (중복 방지)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{now}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    # 이미지 파일 저장
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # 저장 완료 메시지 반환
    return JSONResponse(content={
        "message": "이미지 저장 완료",
        "saved_filename": filename,
        "result": "OK"
    })
