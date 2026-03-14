import secrets
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_PATH = Path(__file__).parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    PROJECT_NAME: str = "RAG Chatbot API"
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Logging Configuration
    LOG_LEVEL: str = "INFO"

    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    API_KEY_HEADER: str = "X-API-Key"
    API_KEYS: list[str] = []  # Optional API keys for simple auth

    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]

    DATABASE_URL: str = "sqlite:///./rag_chatbot.db"

    MODEL_FOLDER: Path = ROOT_PATH / "models"
    VECTOR_STORE_PATH: Path = ROOT_PATH / Path("vector_store")
    DOCS_PATH: Path = ROOT_PATH / Path("docs")

    DEFAULT_MODEL: str = "llama-3.2:1b"
    DEFAULT_K: int = 2
    DEFAULT_MAX_NEW_TOKENS: int = 512
    DEFAULT_CHUNK_SIZE: int = 1000
    DEFAULT_CHUNK_OVERLAP: int = 50
    DEFAULT_SYNTHESIS_STRATEGY: str = "async-tree-summarization"
    DEFAULT_CHAT_HISTORY_LENGTH: int = 2

    WEBSOCKET_MAX_SIZE: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_UPLOAD_EXTENSIONS: list[str] = [".md"]


settings = Settings()
