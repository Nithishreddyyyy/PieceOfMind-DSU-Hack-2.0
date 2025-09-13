from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from fastapi.responses import RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
import pymysql
from werkzeug.security import check_password_hash
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()
SESSION_SECRET = os.getenv("SESSION_SECRET", "supersecretkey")


# MySQL setup
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "test1234")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "DSUHack")

def get_mysql_connection():
    return pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        cursorclass=pymysql.cursors.DictCursor
    )

# FastAPI app setup
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")



# ------------------------
# Main Routes
# ------------------------

@app.get("/")
async def index(request: Request):
    user = request.session.get("user")
    return templates.TemplateResponse("index.html", {"request": request, "user": user})



# Login page (GET)
@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

# Login form submission (POST)
@app.post("/login")
async def login_post(request: Request, email: str = Form(...), password: str = Form(...)):
    conn = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT Name, Email, Password FROM Users WHERE Email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    error = None
    if user:
        db_password = user['Password']
        if db_password == password or check_password_hash(db_password, password):
            request.session['user'] = {"email": user['Email'], "name": user['Name']}
            return RedirectResponse(url="/dashboard", status_code=302)
        else:
            error = "Wrong credentials."
    else:
        error = "Wrong credentials."
    return templates.TemplateResponse("login.html", {"request": request, "error": error, "email": email})


def login_required(request: Request):
    print("Session:", request.session)
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return True

@app.get("/dashboard")
async def dashboard(request: Request):
    login_required(request)
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/monitoring")
async def monitoring(request: Request):
    login_required(request)
    return templates.TemplateResponse("monitoring.html", {"request": request})


@app.get("/focus")
async def focus(request: Request):
    login_required(request)
    return templates.TemplateResponse("focus.html", {"request": request})


@app.get("/settings")
async def settings(request: Request):
    login_required(request)
    return templates.TemplateResponse("settings.html", {"request": request})


@app.get("/onboarding")
async def onboarding(request: Request):
    login_required(request)
    return templates.TemplateResponse("onboarding.html", {"request": request})


@app.get("/privacy")
async def privacy(request: Request):
    return templates.TemplateResponse("privacy.html", {"request": request})


@app.get("/terms")
async def terms(request: Request):
    return templates.TemplateResponse("terms.html", {"request": request})


@app.get("/testLogin")
async def test_login(request: Request):
    login_required(request)
    return templates.TemplateResponse("testLogin.html", {"request": request})

# ------------------------
# Redirects for HTML paths
# ------------------------

@app.get("/settings.html")
async def settings_html_redirect():
    return RedirectResponse(url="/settings", status_code=302)

@app.get("/app/templates/{page_name}.html")
async def template_redirect(page_name: str):
    # Map directly to main route names
    redirect_map = {
        "index": "/",
        "dashboard": "/dashboard",
        "monitoring": "/monitoring",
        "focus": "/focus",
        "settings": "/settings",
        "onboarding": "/onboarding",
        "privacy": "/privacy",
        "terms": "/terms",
        "testLogin": "/testLogin",
    }
    target = redirect_map.get(page_name, "/")
    return RedirectResponse(url=target, status_code=302)
