import os
import json
import re
import uuid
import shutil
import asyncio
import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, Request, status
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from PIL import Image
import bcrypt
import google.generativeai as genai
from dotenv import load_dotenv

# --- CONFIGURATION & ENV ---
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_pocket_smart_ai_2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("No Google API Key found in environment variables. Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# FastAPI App Initialization
app = FastAPI(title="PocketSmart: AI Budget Planner")

# CORS Middleware (Epic 3 - CORS & Static Routing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password Hashing & Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Static & Template Directories
templates = Jinja2Templates(directory="templates")
os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- IN-MEMORY DATA STORAGE ---
users_db: Dict[str, dict] = {}
blacklisted_tokens = set()
user_recommendations: Dict[str, list] = {}  # username -> list of RecommendationItem

class UserSession:
    def __init__(self, username: str, token: str):
        self.username = username
        self.token = token
        self.login_time = datetime.now(timezone.utc)
        self.last_activity = datetime.now(timezone.utc)
        self.user_data: Dict[str, Any] = {}

class RecommendationItem:
    def __init__(self, rec_id: str, rec_type: str, input_summary: Any, result_summary: Any, full_result: Any):
        self.id = rec_id
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.recommendation_type = rec_type
        self.input_summary = input_summary
        self.result_summary = result_summary
        self.full_result = full_result

active_sessions: Dict[str, UserSession] = {}

# --- PYDANTIC SCHEMAS ---
class RegisterUser(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserInDB(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    hashed_password: str

class HomeBudgetInput(BaseModel):
    total_budget: float
    num_lights: int = 0
    num_fans: int = 0
    num_furniture: int = 0
    num_dining_tables: int = 0
    has_living_room: bool = True
    has_kitchen: bool = False
    has_bedroom: bool = False
    additional_requirements: Optional[str] = "None"

class PartyBudgetInput(BaseModel):
    total_budget: float
    party_type: str
    num_guests: int
    venue_type: Optional[str] = "Not specified"
    needs_catering: bool = True
    needs_decoration: bool = True
    needs_entertainment: bool = True
    additional_requirements: Optional[str] = "None"

class JewelryBudgetInput(BaseModel):
    total_budget: float
    occasion: str
    preferences: Optional[str] = "Not specified"

# --- AUTH & HELPER FUNCTIONS ---
def verify_password(plain_password, hashed_password):
    try:
        if isinstance(hashed_password, str):
            hashed_password = hashed_password.encode('utf-8')
        return bcrypt.checkpw(plain_password.encode('utf-8')[:72], hashed_password)
    except Exception:
        return False

def get_password_hash(password):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8')[:72], salt).decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_token(request: Request) -> Optional[str]:
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ")[1]
    return request.cookies.get("access_token")

async def get_current_user(request: Request) -> Optional[UserInDB]:
    token = await get_token(request)
    if not token or token in blacklisted_tokens:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in users_db:
            return None
        return UserInDB(**users_db[username])
    except JWTError:
        return None

async def get_current_active_user(request: Request) -> UserInDB:
    user = await get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

def save_upload_file(image: UploadFile) -> str:
    filename = f"{uuid.uuid4()}_{image.filename}"
    saved_path = os.path.join("static", "uploads", filename)
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    return saved_path

def save_to_history(username: str, recommendation_type: str, input_data: dict, result: dict):
    rec_id = str(uuid.uuid4())[:8]
    summary = result.get("budget_breakdown") or result.get("jewelry_recommendations") or result
    item = RecommendationItem(
        rec_id=rec_id,
        rec_type=recommendation_type,
        input_summary=input_data,
        result_summary=summary,
        full_result=result
    )
    if username not in user_recommendations:
        user_recommendations[username] = []
    user_recommendations[username].append(item)

def extract_json_from_response(text: str) -> dict:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    clean_text = text.strip()
    if clean_text.startswith("{") and clean_text.endswith("}"):
        return json.loads(clean_text)
    return {"raw_text": text}

# --- AI ENGINES ---
def get_home_recommendations(budget_input: HomeBudgetInput) -> dict:
    prompt = f"""
    Interior design product recommendations for an Indian home with budget ₹{budget_input.total_budget:.2f}.
    Items: {budget_input.num_lights} lights, {budget_input.num_fans} fans, {budget_input.num_furniture} furniture, {budget_input.num_dining_tables} dining tables.
    Rooms: Living Room={budget_input.has_living_room}, Kitchen={budget_input.has_kitchen}, Bedroom={budget_input.has_bedroom}.
    Notes: {budget_input.additional_requirements}
    Return strictly JSON:
    {{
        "total_budget": {budget_input.total_budget:.2f},
        "budget_breakdown": [
            {{
                "category": "Lighting",
                "allocation": 0.0,
                "items": [
                    {{"name": "", "description": "", "estimated_price": 0.0, "quantity": 0, "search_term": ""}}
                ]
            }}
        ],
        "remaining_budget": 0.0,
        "additional_suggestions": []
    }}
    """
    try:
        response = model.generate_content(prompt)
        result = extract_json_from_response(response.text)
        for cat in result.get("budget_breakdown", []):
            for itm in cat.get("items", []):
                term = urllib.parse.quote_plus(itm.get("search_term") or itm.get("name", ""))
                itm["shopping_links"] = {
                    "amazon": f"https://www.amazon.in/s?k={term}",
                    "flipkart": f"https://www.flipkart.com/search?q={term}",
                    "ikea": f"https://www.ikea.com/in/en/search/?q={term}"
                }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

def get_party_recommendations(budget_input: PartyBudgetInput) -> dict:
    prompt = f"""
    Party planning recommendations in India with budget ₹{budget_input.total_budget:.2f}.
    Event: {budget_input.party_type}, Guests: {budget_input.num_guests}, Venue: {budget_input.venue_type}.
    Catering: {budget_input.needs_catering}, Decor: {budget_input.needs_decoration}, Entertainment: {budget_input.needs_entertainment}.
    Notes: {budget_input.additional_requirements}
    Return strictly JSON:
    {{
        "total_budget": {budget_input.total_budget:.2f},
        "budget_breakdown": [
            {{
                "category": "catering",
                "allocation": 0.0,
                "items": [
                    {{"name": "", "description": "", "estimated_price": 0.0, "quantity": 0, "search_term": ""}}
                ]
            }}
        ],
        "venue_suggestions": [
            {{"name": "", "type": "", "capacity": 0, "estimated_cost": 0.0, "search_terms": ""}}
        ],
        "remaining_budget": 0.0,
        "additional_suggestions": []
    }}
    """
    try:
        response = model.generate_content(prompt)
        result = extract_json_from_response(response.text)
        for cat in result.get("budget_breakdown", []):
            for itm in cat.get("items", []):
                term = urllib.parse.quote_plus(itm.get("search_term") or itm.get("name", ""))
                itm["shopping_links"] = {
                    "amazon": f"https://www.amazon.in/s?k={term}",
                    "swiggy": f"https://www.swiggy.com/search?query={term}",
                    "zomato": f"https://www.zomato.com/search?q={term}"
                }
        for ven in result.get("venue_suggestions", []):
            v_term = urllib.parse.quote_plus(ven.get("search_terms") or ven.get("name", ""))
            ven["search_links"] = {
                "google": f"https://www.google.com/search?q={v_term}",
                "booking": f"https://www.booking.com/search.html?ss={v_term}",
                "makemytrip": f"https://www.makemytrip.com/hotels/hotel-listing/?searchtext={v_term}",
                "oyorooms": f"https://www.oyorooms.com/search?location={v_term}"
            }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

def get_jewelry_recommendations(budget_input: JewelryBudgetInput, image_path: Optional[str] = None) -> dict:
    prompt = f"""
    Jewelry styling recommendations in India for budget ₹{budget_input.total_budget:.2f}.
    Occasion: {budget_input.occasion}, Style Preferences: {budget_input.preferences}.
    Return strictly JSON:
    {{
        "total_budget": {budget_input.total_budget:.2f},
        "outfit_analysis": {{"colors": [], "style": "", "formality": ""}},
        "jewelry_recommendations": [
            {{"item_type": "", "description": "", "style": "", "estimated_price": 0.0, "search_term": ""}}
        ],
        "remaining_budget": 0.0,
        "styling_tips": []
    }}
    """
    try:
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path)
            response = model.generate_content([prompt, img])
        else:
            response = model.generate_content(prompt)
        result = extract_json_from_response(response.text)
        for itm in result.get("jewelry_recommendations", []):
            term = urllib.parse.quote_plus(itm.get("search_term") or itm.get("item_type", ""))
            itm["shopping_links"] = {
                "amazon": f"https://www.amazon.in/s?k={term}",
                "flipkart": f"https://www.flipkart.com/search?q={term}",
                "tanishq": f"https://www.tanishq.co.in/search?q={term}",
                "caratlane": f"https://www.caratlane.com/search?q={term}",
                "meesho": f"https://www.meesho.com/search?q={term}"
            }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# --- STARTUP EVENT (Story 4: Startup and main function) ---
@app.on_event("startup")
async def setup_session_cleanup():
    """Background task to clean up expired sessions"""
    async def cleanup_expired_sessions():
        while True:
            current_time = datetime.now(timezone.utc)
            expired_sessions = [
                username for username, session in active_sessions.items()
                if (current_time - session.last_activity).total_seconds() > 1800  # 30 minutes
            ]
            for username in expired_sessions:
                print(f"Removing expired session for {username}")
                del active_sessions[username]
            await asyncio.sleep(300)
    asyncio.create_task(cleanup_expired_sessions())

# --- USER AUTHENTICATION & SESSION ENDPOINTS ---
@app.post("/register")
async def register_submit(user: RegisterUser):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    users_db[user.username] = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": get_password_hash(user.password),
    }
    return JSONResponse(status_code=201, content={"message": "User registered successfully"})

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user_record = users_db.get(form_data.username)
    if not user_record or not verify_password(form_data.password, user_record["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": form_data.username}, expires_delta=access_token_expires)
    
    session = UserSession(username=form_data.username, token=access_token)
    active_sessions[form_data.username] = session
    
    response = JSONResponse(content={"access_token": access_token, "token_type": "bearer"})
    response.set_cookie(key="access_token", value=access_token, httponly=True, max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60, samesite="lax")
    return response

@app.post("/logout")
async def logout(request: Request):
    token = await get_token(request)
    if token:
        blacklisted_tokens.add(token)
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username in active_sessions:
                del active_sessions[username]
        except JWTError:
            pass
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="access_token")
    return response

@app.get("/session-info")
async def get_session_info(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    session = active_sessions.get(current_user.username)
    if session:
        return {
            "username": session.username,
            "login_time": session.login_time.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "user_data": session.user_data
        }
    raise HTTPException(status_code=404, detail="No active session found")

# --- HTML TEMPLATE ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    user = await get_current_user(request)
    return templates.TemplateResponse(request=request, name="index.html", context={"user": user})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = await get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse(request=request, name="login.html")

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    user = await get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse(request=request, name="register.html")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    return templates.TemplateResponse(request=request, name="dashboard.html", context={"user": current_user})

@app.get("/home-planner", response_class=HTMLResponse)
async def home_planner_page(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    return templates.TemplateResponse(request=request, name="home_planner.html", context={"user": current_user})

@app.get("/party-planner", response_class=HTMLResponse)
async def party_planner_page(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    return templates.TemplateResponse(request=request, name="party_planner.html", context={"user": current_user})

@app.get("/jewelry-planner", response_class=HTMLResponse)
async def jewelry_planner_page(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    return templates.TemplateResponse(request=request, name="jewelry_planner.html", context={"user": current_user})

@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request, current_user: UserInDB = Depends(get_current_active_user)):
    """History page to view past recommendations (Story 3: CORS & Static Routing)"""
    return templates.TemplateResponse(request=request, name="history.html", context={"user": current_user})

# --- EPIC 3: CORE FASTAPI ROUTES & BUDGET INTEGRATIONS ---

@app.post("/home-budget")
@app.post("/api/home-planner")
async def plan_home_budget(
    budget_input: HomeBudgetInput,
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Generate home budget recommendations (Story 1)"""
    if current_user.username in active_sessions:
        active_sessions[current_user.username].last_activity = datetime.now(timezone.utc)
        active_sessions[current_user.username].user_data["last_home_budget"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "budget": budget_input.total_budget,
            "requirements": {
                "lights": budget_input.num_lights,
                "fans": budget_input.num_fans,
                "furniture": budget_input.num_furniture,
                "dining_tables": budget_input.num_dining_tables
            }
        }
    
    result = get_home_recommendations(budget_input)
    save_to_history(
        username=current_user.username,
        recommendation_type="home",
        input_data=budget_input.dict(),
        result=result
    )
    return result

@app.post("/party-budget")
@app.post("/api/party-planner")
async def plan_party_budget(
    budget_input: PartyBudgetInput,
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Generate party planning recommendations (Story 1)"""
    if current_user.username in active_sessions:
        active_sessions[current_user.username].last_activity = datetime.now(timezone.utc)
        active_sessions[current_user.username].user_data["last_party_budget"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "budget": budget_input.total_budget,
            "party_type": budget_input.party_type,
            "num_guests": budget_input.num_guests
        }
    
    result = get_party_recommendations(budget_input)
    save_to_history(
        username=current_user.username,
        recommendation_type="party",
        input_data=budget_input.dict(),
        result=result
    )
    return result

@app.post("/jewelry-budget")
@app.post("/api/jewelry-planner")
async def plan_jewelry_budget(
    total_budget: float = Form(...),
    occasion: str = Form(...),
    preferences: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    request: Request = None,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Generate jewelry recommendations with optional outfit image (Story 1)"""
    budget_input = JewelryBudgetInput(
        total_budget=total_budget,
        occasion=occasion,
        preferences=preferences or "Not specified"
    )
    
    image_path = None
    if image and image.filename:
        image_path = save_upload_file(image)
        
    if current_user.username in active_sessions:
        active_sessions[current_user.username].last_activity = datetime.now(timezone.utc)
        active_sessions[current_user.username].user_data["last_jewelry_budget"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "budget": budget_input.total_budget,
            "occasion": budget_input.occasion,
            "has_image": image is not None
        }
        
    result = get_jewelry_recommendations(budget_input, image_path=image_path)
    
    input_data = budget_input.dict()
    if image and image.filename:
        input_data["image"] = image.filename
        
    save_to_history(
        username=current_user.username,
        recommendation_type="jewelry",
        input_data=input_data,
        result=result
    )
    return result

@app.get("/recommendation-history")
async def get_recommendation_history(
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Get the user's recommendation history list (Story 1)"""
    if current_user.username not in user_recommendations:
        return {"history": []}
        
    history = sorted(
        user_recommendations[current_user.username],
        key=lambda x: x.timestamp,
        reverse=True
    )
    
    history_data = []
    for item in history:
        history_data.append({
            "id": item.id,
            "timestamp": item.timestamp,
            "type": item.recommendation_type,
            "input": item.input_summary,
            "summary": item.result_summary
        })
    return {"history": history_data}

@app.get("/recommendation-details/{recommendation_id}")
async def get_recommendation_details(
    recommendation_id: str,
    request: Request,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Get full details of a specific recommendation (Story 2: Modular Architecture Setup)"""
    if current_user.username not in user_recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found")
        
    for item in user_recommendations[current_user.username]:
        if item.id == recommendation_id:
            return {
                "id": item.id,
                "timestamp": item.timestamp,
                "type": item.recommendation_type,
                "input": item.input_summary,
                "full_result": item.full_result
            }
            
    raise HTTPException(status_code=404, detail="Recommendation not found")

# --- MAIN ENTRY POINT (Story 4) ---
if __name__ == "__main__":
    import uvicorn
    print("Starting PocketSmart: AI Budget Planner...")
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)