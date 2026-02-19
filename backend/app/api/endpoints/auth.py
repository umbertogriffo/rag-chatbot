from fastapi import APIRouter

from backend.app.core.security import create_access_token
from backend.app.schemas.auth import Token, TokenRequest

router = APIRouter()


@router.post("/token", response_model=Token)
async def login(request: TokenRequest):
    """Simple token generation - in production, validate credentials against a user store."""
    access_token = create_access_token(subject=request.username)
    return Token(access_token=access_token)
