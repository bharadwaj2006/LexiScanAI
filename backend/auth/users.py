"""User store backed by PostgreSQL via SQLAlchemy."""
from datetime import datetime, timezone

import bcrypt
from sqlalchemy.orm import Session

from db import SessionLocal, engine
from models import Base, User


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _user_to_dict(user: User) -> dict:
    """Convert a User ORM row to the dict shape the rest of the app expects."""
    return {
        "username": user.username,
        "email": user.email,
        "hashed_password": user.password_hash,
        "created_at": user.created_at.isoformat(),
    }


def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def seed_default_admin() -> None:
    """Create the users table (if needed) and seed default + Bharadwaj accounts."""
    Base.metadata.create_all(bind=engine)

    db: Session = SessionLocal()
    try:
        _ensure_user(
            db,
            username="admin",
            email="admin@lexiscan.local",
            password="admin123",
        )
        _ensure_user(
            db,
            username="Bharadwaj",
            email="bharadwaj2701@gmail.com",
            password="Bharadwaj@1234",
        )
        db.commit()
    finally:
        db.close()


def _ensure_user(db: Session, username: str, email: str, password: str) -> None:
    """Insert a user only if neither the username nor email already exists."""
    exists = (
        db.query(User)
        .filter((User.username == username) | (User.email == email))
        .first()
    )
    if exists:
        return
    db.add(
        User(
            username=username,
            email=email,
            password_hash=_hash_password(password),
            created_at=datetime.now(timezone.utc),
        )
    )


def get_user(username: str) -> dict | None:
    db: Session = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        return _user_to_dict(user) if user else None
    finally:
        db.close()


def create_user(username: str, email: str, password: str) -> dict:
    db: Session = SessionLocal()
    try:
        if db.query(User).filter(User.username == username).first():
            raise ValueError(f"Username '{username}' is already taken.")
        if db.query(User).filter(User.email == email).first():
            raise ValueError(f"Email '{email}' is already registered.")

        now = datetime.now(timezone.utc)
        user = User(
            username=username,
            email=email,
            password_hash=_hash_password(password),
            created_at=now,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return {"username": user.username, "email": user.email, "created_at": user.created_at.isoformat()}
    finally:
        db.close()


def authenticate_user(username: str, password: str) -> dict | None:
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user
