from pathlib import Path

from bot.client.openai_client import OpenAIClient
from bot.model.model_registry import get_model_settings
from core.config import settings


def create_llm_client(model_folder: Path) -> OpenAIClient:
    settings.MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    model_settings = get_model_settings(settings.MODEL)

    return OpenAIClient(
        base_url=settings.LLAMA_SERVER_BASE_URL,
        model_name=settings.MODEL,
        model_settings=model_settings,
        timeout=settings.LLAMA_SERVER_TIMEOUT,
    )
