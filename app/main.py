from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUTH0_API_AUDIENCE = os.getenv("AUTH0_API_AUDIENCE")
SESSION_SECRET = os.getenv("SESSION_SECRET", "supersecretkey")

# Configure OAuth with Auth0
oauth = OAuth()
oauth.register(
    name="auth0",
    client_id=AUTH0_CLIENT_ID,
    client_secret=AUTH0_CLIENT_SECRET,
    server_metadata_url=f"https://{AUTH0_DOMAIN}/.well-known/openid-configuration",
    client_kwargs={"scope": "openid profile email"},
)

# MongoDB setup
mongo_uri = os.getenv(
    "MONGO_URI",
    "mongodb+srv://admin:test1234@devhack.6gqwt4w.mongodb.net/?retryWrites=true&w=majority&appName=DevHack",
)
mongo_client = MongoClient(mongo_uri, server_api=ServerApi("1"))
mongo_db = mongo_client["pieceofmind"]
users_collection = mongo_db["users"]

# FastAPI app setup
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------------
# Auth Routes
# ------------------------

@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for("callback")
    return await oauth.auth0.authorize_redirect(request, redirect_uri)


@app.get("/callback")
async def callback(request: Request):
    token = await oauth.auth0.authorize_access_token(request)
    userinfo = token.get("userinfo") or await oauth.auth0.parse_id_token(request, token)

    # Store user in MongoDB if not exists
    auth0_id = userinfo.get("sub")
    email = userinfo.get("email")
    name = userinfo.get("name")

    if not users_collection.find_one({"auth0_id": auth0_id}):
        users_collection.insert_one(
            {
                "auth0_id": auth0_id,
                "email": email,
                "full_name": name,
                "profile": userinfo,
            }
        )

    # Save user in session
    request.session["user"] = userinfo
    return RedirectResponse(url="/")


@app.get("/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    return_to = str(request.url_for("index"))
    return RedirectResponse(
        url=f"https://{AUTH0_DOMAIN}/v2/logout?client_id={AUTH0_CLIENT_ID}&returnTo={return_to}",
        status_code=302,
    )

# ------------------------
# Main Routes
# ------------------------

@app.get("/")
async def index(request: Request):
    user = request.session.get("user")
    return templates.TemplateResponse("index.html", {"request": request, "user": user})


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


@app.get("/privacy")
async def privacy(request: Request):
    return templates.TemplateResponse("privacy.html", {"request": request})


@app.get("/terms")
async def terms(request: Request):
    return templates.TemplateResponse("terms.html", {"request": request})


@app.get("/testLogin")
async def test_login(request: Request):
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
