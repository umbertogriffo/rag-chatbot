from datetime import datetime, timezone
from uuid import uuid4

from sqlmodel import Field, SQLModel


class ChatSession(SQLModel, table=True):
    __tablename__ = "chat_sessions"

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    user_id: str = Field(index=True)
    title: str = Field(default="New Chat")
    model_name: str = Field(default="llama-3.1")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatMessage(SQLModel, table=True):
    __tablename__ = "chat_messages"

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    session_id: str = Field(index=True, foreign_key="chat_sessions.id")
    role: str  # "user" or "assistant"
    content: str
    sources: str | None = Field(default=None)  # JSON-serialized source documents
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
