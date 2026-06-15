"""User store backed by a local JSON file."""
import json
import os
from datetime import datetime, timezone
import bcrypt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")


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
        # Hash "admin123"
        hashed = bcrypt.hashpw("admin123".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        users["admin"] = {
            "username": "admin",
            "email": "admin@lexiscan.local",
            "hashed_password": hashed,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _save(users)


def get_user(username: str) -> dict | None:
    users = _load()
    return users.get(username)


def create_user(username: str, email: str, password: str) -> dict:
    users = _load()
    if username in users:
        raise ValueError(f"Username '{username}' is already taken.")
    
    # Check if email is already taken
    for u in users.values():
        if u.get("email") == email:
            raise ValueError(f"Email '{email}' is already registered.")

    now = datetime.now(timezone.utc).isoformat()
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    users[username] = {
        "username": username,
        "email": email,
        "hashed_password": hashed,
        "created_at": now,
    }
    _save(users)
    return {"username": username, "email": email, "created_at": now}


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def authenticate_user(username: str, password: str) -> dict | None:
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user
