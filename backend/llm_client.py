from pathlib import Path

from bot.client.openai_client import OpenAIClient
from core.config import settings
from core.model import ModelSettings


def create_llm_client(model_folder: Path) -> OpenAIClient:
    settings.MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

    model_settings = ModelSettings(
        url=settings.MODEL_URL,
        name=settings.MODEL,
        file_name=f"{settings.MODEL}.gguf",
        reasoning_start_tag=settings.REASONING_START_TAG,
        reasoning_stop_tag=settings.REASONING_STOP_TAG,
        system_template=settings.SYSTEM_TEMPLATE,
        reasoning=settings.REASONING,
    )

    return OpenAIClient(
        base_url=settings.LLAMA_SERVER_BASE_URL,
        model_folder=model_folder,
        model_settings=model_settings,
        timeout=300,
    )
