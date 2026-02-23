from fastapi import APIRouter

from api.endpoints import chat, health

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(chat.router, prefix="", tags=["chat"])
