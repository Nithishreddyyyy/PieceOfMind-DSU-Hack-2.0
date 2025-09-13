import os
from flask import Flask, render_template, redirect, url_for, session, request
from authlib.integrations.flask_client import OAuth
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SESSION_SECRET", "supersecretkey")

# Auth0 config
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUTH0_API_AUDIENCE = os.getenv("AUTH0_API_AUDIENCE")
AUTH0_CALLBACK_URL = os.getenv("AUTH0_CALLBACK_URL", "https://localhost:5000/callback")

# MongoDB config
mongo_uri = os.getenv("MONGO_URI", "mongodb+srv://admin:test1234@devhack.6gqwt4w.mongodb.net/?retryWrites=true&w=majority&appName=DevHack")
mongo_client = MongoClient(mongo_uri)
mongo_db = mongo_client["pieceofmind"]
users_collection = mongo_db["users"]

# Auth0 OAuth setup
oauth = OAuth(app)
auth0 = oauth.register(
    'auth0',
    client_id=AUTH0_CLIENT_ID,
    client_secret=AUTH0_CLIENT_SECRET,
    api_base_url=f'https://{AUTH0_DOMAIN}',
    access_token_url=f'https://{AUTH0_DOMAIN}/oauth/token',
    authorize_url=f'https://{AUTH0_DOMAIN}/authorize',
    client_kwargs={
        'scope': 'openid profile email',
    },
)

@app.route('/')
def index():
    user = session.get('user')
    return render_template('index.html', user=user)

@app.route('/login')
def login():
    return auth0.authorize_redirect(redirect_uri=AUTH0_CALLBACK_URL)

@app.route('/callback')
def callback():
    token = auth0.authorize_access_token()
    userinfo = auth0.parse_id_token(token)
    auth0_id = userinfo.get('sub')
    email = userinfo.get('email')
    name = userinfo.get('name')
    # Store user in MongoDB if not exists
    if not users_collection.find_one({"auth0_id": auth0_id}):
        users_collection.insert_one({
            "auth0_id": auth0_id,
            "email": email,
            "full_name": name,
            "profile": userinfo
        })
    session['user'] = userinfo
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(
        f"https://{AUTH0_DOMAIN}/v2/logout?client_id={AUTH0_CLIENT_ID}&returnTo=https://localhost:5000"
    )

# Main navigation endpoints
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', user=session.get('user'))

@app.route('/monitoring')
def monitoring():
    return render_template('monitoring.html', user=session.get('user'))

@app.route('/focus')
def focus():
    return render_template('focus.html', user=session.get('user'))

@app.route('/settings')
def settings():
    return render_template('settings.html', user=session.get('user'))

@app.route('/onboarding')
def onboarding():
    return render_template('onboarding.html', user=session.get('user'))

# Footer/legal endpoints
@app.route('/privacy')
def privacy():
    return render_template('privacy.html', user=session.get('user'))

@app.route('/terms')
def terms():
    return render_template('terms.html', user=session.get('user'))

# Test/development endpoints
@app.route('/testLogin')
def test_login():
    return render_template('testLogin.html', user=session.get('user'))

if __name__ == '__main__':
    app.run(debug=True,host = '0.0.0.0' , port = 3000)
