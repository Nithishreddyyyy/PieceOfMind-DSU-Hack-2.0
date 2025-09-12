from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
from config import MONGO_URI

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 🔹 Connect to MongoDB once when the app starts
try:
    client = MongoClient(MONGO_URI)
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB!")
except Exception as e:
    print("❌ Could not connect to MongoDB:", e)

# ---- Routes ----

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/testdb")
async def testdb():
    try:
        client.admin.command('ping')
        return {"message": "✅ MongoDB connection successful!"}
    except Exception as e:
        return {"message": f"❌ MongoDB connection failed: {e}"}
