"""Authentication API router — /api/auth/*"""
from datetime import timedelta

from fastapi import APIRouter, HTTPException, status

from auth.models import Token, UserCreate, UserLogin, UserOut
from auth.users import authenticate_user, create_user
from auth.jwt_utils import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, get_current_user
from fastapi import Depends

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=201)
async def register(body: UserCreate):
    """Create a new user account."""
    try:
        user = create_user(body.username, body.email, body.password)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return user


@router.post("/login", response_model=Token)
async def login(body: UserLogin):
    """Authenticate and return a JWT access token."""
    user = authenticate_user(body.username, body.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": token, "token_type": "bearer"}


@router.get("/me", response_model=UserOut)
async def me(current_user: dict = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return {
        "username": current_user["username"],
        "email": current_user.get("email", ""),
        "created_at": current_user.get("created_at", ""),
    }

