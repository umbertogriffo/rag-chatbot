from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenRequest(BaseModel):
    username: str
    password: str | None = None
