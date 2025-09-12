from fastapi import FastAPI, Request, Depends
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your-random-secret-key")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# OAuth Config
oauth = OAuth()
oauth.register(
    "auth0",
    client_id=AUTH0_CLIENT_ID,
    client_secret=AUTH0_CLIENT_SECRET,
    client_kwargs={"scope": "openid profile email"},
    server_metadata_url=f"https://{AUTH0_DOMAIN}/.well-known/openid-configuration",
)

# ---- Routes ----
@app.get("/")
async def index(request: Request):
    user = request.session.get("user")
    return templates.TemplateResponse("index.html", {"request": request, "user": user})

@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for("callback")
    print(f"Redirect URI sent to Auth0: {redirect_uri}")
    return await oauth.auth0.authorize_redirect(request, redirect_uri)

# @app.get("/callback")
# async def callback(request: Request):
#     token = await oauth.auth0.authorize_access_token(request)
#     user = token.get("userinfo")
#     request.session["user"] = dict(user)
#     return RedirectResponse(url="/")

@app.get("/callback")
async def callback(request: Request):
    try:
        token = await oauth.auth0.authorize_access_token(request)
        user = token.get("userinfo")
        request.session["user"] = dict(user)
        return RedirectResponse(url="/testLogin")
    except Exception as e:
        import traceback
        traceback_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return HTMLResponse(
            f"<h2>❌ Callback error</h2><pre>{traceback_str}</pre>",
            status_code=500
        )




@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(
        url=f"https://{AUTH0_DOMAIN}/v2/logout?client_id={AUTH0_CLIENT_ID}&returnTo=http://localhost:8000"
    )
