from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from datetime import datetime
import os

app = FastAPI()

UPLOAD_DIR = "uploaded_images"  
os.makedirs(UPLOAD_DIR, exist_ok=True)  

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{now}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    
    return JSONResponse(content={
        "message": "이미지 저장 완료",
        "saved_filename": filename,
        "result": "OK"
    })
