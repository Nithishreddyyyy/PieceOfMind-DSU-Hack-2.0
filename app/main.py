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

@app.get("/focus")
async def focus(request: Request):
    return templates.TemplateResponse("focus.html", {"request": request})

@app.get("/settings")
async def settings(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})

@app.get("/onboarding")
async def onboarding(request: Request):
    return templates.TemplateResponse("onboarding.html", {"request": request})

@app.get("/privacy")
async def privacy(request: Request):
    return templates.TemplateResponse("privacy.html", {"request": request})

@app.get("/terms")
async def terms(request: Request):
    return templates.TemplateResponse("terms.html", {"request": request})

@app.get("/testLogin")
async def test_login(request: Request):
    return templates.TemplateResponse("testLogin.html", {"request": request})

