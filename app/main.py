from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard")
async def Dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html",{"request": request})


# monitoring
@app.get("/monitoring")
async def Monitoring(request: Request):
    return templates.TemplateResponse("monitoring.html",{"request" : request})

