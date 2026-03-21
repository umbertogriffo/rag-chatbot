from fastapi import APIRouter

from api.endpoints import admin, chat, chat_stream, documents, health

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(chat.router, prefix="", tags=["chat"])
api_router.include_router(documents.router, prefix="", tags=["documents"])
api_router.include_router(chat_stream.router, prefix="", tags=["chat-stream"])
api_router.include_router(admin.router, prefix="", tags=["admin"])
