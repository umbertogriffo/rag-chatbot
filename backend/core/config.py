from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_PATH = Path(__file__).parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_PATH / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    PROJECT_NAME: str = "Chatbot API"
    VERSION: str = "0.1.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Logging Configuration
    LOG_LEVEL: str = "INFO"

    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]

    MODEL_FOLDER: Path = ROOT_PATH / "models"
    VECTOR_STORE_PATH: Path = ROOT_PATH / "vector_store" / "docs_index"
    DOC_REGISTRY_DB_PATH: Path = ROOT_PATH / "vector_store" / "document_registry.db"
    DOC_REGISTRY_DB_URL: str = f"sqlite:///{ROOT_PATH / 'vector_store' / 'document_registry.db'}"
    DOCS_PATH: Path = ROOT_PATH / Path("docs")

    MODEL: str = "llama-3.2:1b"
    K: int = 2
    MAX_NEW_TOKENS: int = 512
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 50
    SYNTHESIS_STRATEGY: str = "tree-summarization"
    CHAT_HISTORY_LENGTH: int = 2
    NUM_RETRIEVALS: int = 2

    WEBSOCKET_MAX_SIZE: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_UPLOAD_EXTENSIONS: list[str] = [".md"]


settings = Settings()
