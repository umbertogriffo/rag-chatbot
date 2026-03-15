from pydantic import BaseModel


class ChatRequest(BaseModel):
    text: str
    rag: bool = False
    reasoning: bool = False
    web_search: bool = False
