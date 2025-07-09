from fastapi import FastAPI
from pymongo import MongoClient
import os

app = FastAPI()

mongo_url = os.getenv("MONGODB_URL")
client = MongoClient(mongo_url)

@app.get("/dbstatus")
def db_status():
    try:
        # ping 명령으로 상태 확인
        client.admin.command('ping')
        return {"db_connected": True}
    except Exception as e:
        return {"db_connected": False, "error": str(e)}
