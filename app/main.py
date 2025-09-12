from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root endpoint
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Main navigation endpoints
@app.get("/dashboard")
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/monitoring")
async def monitoring(request: Request):
    return templates.TemplateResponse("monitoring.html", {"request": request})

@app.get("/focus")
async def focus(request: Request):
    return templates.TemplateResponse("focus.html", {"request": request})

@app.get("/settings")
async def settings(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})

@app.get("/onboarding")
async def onboarding(request: Request):
    return templates.TemplateResponse("onboarding.html", {"request": request})

# Footer/legal endpoints
@app.get("/privacy")
async def privacy(request: Request):
    return templates.TemplateResponse("privacy.html", {"request": request})

@app.get("/terms")
async def terms(request: Request):
    return templates.TemplateResponse("terms.html", {"request": request})

# Test/development endpoints
@app.get("/testLogin")
async def test_login(request: Request):
    return templates.TemplateResponse("testLogin.html", {"request": request})

# Alternative routes that match the HTML template paths
@app.get("/settings.html")
async def settings_html_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/settings", status_code=302)
@app.get("/app/templates/index.html")
async def index_alt():
    return RedirectResponse(url="/", status_code=302)

@app.get("/app/templates/dashboard.html")
async def dashboard_alt():
    return RedirectResponse(url="/dashboard", status_code=302)

@app.get("/app/templates/monitoring.html")
async def monitoring_alt():
    return RedirectResponse(url="/monitoring", status_code=302)

@app.get("/app/templates/focus.html")
async def focus_alt():
    return RedirectResponse(url="/focus", status_code=302)

@app.get("/app/templates/settings.html")
async def settings_alt():
    return RedirectResponse(url="/settings", status_code=302)

@app.get("/app/templates/onboarding.html")
async def onboarding_alt():
    return RedirectResponse(url="/onboarding", status_code=302)

@app.get("/app/templates/privacy.html")
async def privacy_alt():
    return RedirectResponse(url="/privacy", status_code=302)

@app.get("/app/templates/terms.html")
async def terms_alt():
    return RedirectResponse(url="/terms", status_code=302)

@app.get("/app/templates/testLogin.html")
async def test_login_alt():
    return RedirectResponse(url="/testLogin", status_code=302)

