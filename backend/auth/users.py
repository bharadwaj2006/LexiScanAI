"""User store backed by a local JSON file."""
import json
import os
from datetime import datetime, timezone
from passlib.context import CryptContext

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _load() -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(users: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def seed_default_admin() -> None:
    """Ensure a default admin account exists on first run."""
    users = _load()
    if "admin" not in users:
        users["admin"] = {
            "username": "admin",
            "hashed_password": pwd_ctx.hash("admin123"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _save(users)


def get_user(username: str) -> dict | None:
    users = _load()
    return users.get(username)


def create_user(username: str, password: str) -> dict:
    users = _load()
    if username in users:
        raise ValueError(f"Username '{username}' is already taken.")
    now = datetime.now(timezone.utc).isoformat()
    users[username] = {
        "username": username,
        "hashed_password": pwd_ctx.hash(password),
        "created_at": now,
    }
    _save(users)
    return {"username": username, "created_at": now}


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def authenticate_user(username: str, password: str) -> dict | None:
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user
