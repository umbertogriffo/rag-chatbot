from datetime import datetime

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    model_name: str | None = None
    max_new_tokens: int = 512
    k: int = 2
    use_rag: bool = True
    synthesis_strategy: str = "async-tree-summarization"


class ChatResponse(BaseModel):
    message: str
    session_id: str
    sources: list[dict] | None = None


class ChatSessionResponse(BaseModel):
    id: str
    title: str
    model_name: str
    created_at: datetime
    updated_at: datetime


class ChatMessageResponse(BaseModel):
    id: str
    role: str
    content: str
    sources: list[dict] | None = None
    created_at: datetime
