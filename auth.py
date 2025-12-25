"""
Système d'authentification JWT pour Paris Ciné Clone
"""
import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import jwt

# Configuration JWT
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 jours

# Fichier de stockage des utilisateurs (en production, utiliser une vraie DB)
USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")
security = HTTPBearer()

# ==================== MODELS ====================

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    username: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

class LetterboxdData(BaseModel):
    username: str = ''
    watchlist: list[str] = []
    watched: list[str] = []
    lastUpdated: Optional[str] = None

class UserPreferences(BaseModel):
    selcard: list[str] = []
    seladdr: list[str] = ['*']
    sellang: list[str] = []
    selcine: list[str] = []
    seenFilterMode: str = 'shade'

class UserProfile(BaseModel):
    email: str
    username: str
    preferences: Optional[UserPreferences] = None
    letterboxd: Optional[LetterboxdData] = None
    created_at: str

# ==================== DATABASE (JSON FILE) ====================

def load_users() -> dict:
    """Charge les utilisateurs depuis le fichier JSON."""
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def save_users(users: dict) -> None:
    """Sauvegarde les utilisateurs dans le fichier JSON."""
    try:
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving users: {e}")

# ==================== PASSWORD HASHING ====================

def hash_password(password: str) -> str:
    """Hash un mot de passe avec SHA256 + salt."""
    salt = secrets.token_hex(32)
    pwd_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}${pwd_hash}"

def verify_password(password: str, hashed: str) -> bool:
    """Vérifie un mot de passe contre son hash."""
    try:
        salt, pwd_hash = hashed.split('$')
        return hashlib.sha256((password + salt).encode()).hexdigest() == pwd_hash
    except Exception:
        return False

# ==================== JWT TOKENS ====================

def create_access_token(email: str, username: str) -> str:
    """Crée un token JWT."""
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": email,
        "username": username,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    """Décode et vérifie un token JWT."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# ==================== AUTH DEPENDENCIES ====================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Récupère l'utilisateur courant depuis le token."""
    token = credentials.credentials
    payload = decode_token(token)
    email = payload.get("sub")

    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

    users = load_users()
    user_data = users.get(email)

    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return {
        "email": email,
        "username": user_data.get("username"),
        "preferences": user_data.get("preferences", {}),
        "created_at": user_data.get("created_at")
    }

# ==================== USER MANAGEMENT ====================

def create_user(email: str, username: str, password: str) -> dict:
    """Crée un nouvel utilisateur."""
    users = load_users()

    # Vérifier si l'email existe déjà
    if email in users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Vérifier si le username existe déjà
    for user_data in users.values():
        if user_data.get("username") == username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )

    # Créer l'utilisateur
    user = {
        "email": email,
        "username": username,
        "password": hash_password(password),
        "preferences": UserPreferences().dict(),
        "created_at": datetime.utcnow().isoformat()
    }

    users[email] = user
    save_users(users)

    # Retourner l'utilisateur sans le mot de passe
    user_without_password = user.copy()
    del user_without_password["password"]
    return user_without_password

def authenticate_user(email: str, password: str) -> Optional[dict]:
    """Authentifie un utilisateur."""
    users = load_users()
    user_data = users.get(email)

    if not user_data:
        return None

    if not verify_password(password, user_data.get("password", "")):
        return None

    # Retourner l'utilisateur sans le mot de passe
    user_without_password = user_data.copy()
    del user_without_password["password"]
    return user_without_password

def update_user_preferences(email: str, preferences: dict) -> dict:
    """Met à jour les préférences d'un utilisateur."""
    users = load_users()

    if email not in users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    users[email]["preferences"] = preferences
    save_users(users)

    return users[email]["preferences"]

def update_user_letterboxd(email: str, letterboxd_data: dict) -> dict:
    """Met à jour les données Letterboxd d'un utilisateur."""
    users = load_users()

    if email not in users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    users[email]["letterboxd"] = letterboxd_data
    save_users(users)

    return users[email]["letterboxd"]

def get_user_letterboxd(email: str) -> Optional[dict]:
    """Récupère les données Letterboxd d'un utilisateur."""
    users = load_users()

    if email not in users:
        return None

    return users[email].get("letterboxd", None)
