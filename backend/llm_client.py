from pathlib import Path
from urllib.parse import urlparse

from config import settings
from llm_providers.llamacpp_client import LlamaCppClient
from schemas.model import ModelSettings


def init_llm_client(model_folder: Path) -> LlamaCppClient:
    settings.MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

    model_url_path = urlparse(settings.MODEL_URL).path
    model_file_name = Path(model_url_path).name or f"{settings.MODEL}.gguf"

    model_settings = ModelSettings(
        url=settings.MODEL_URL,
        name=settings.MODEL,
        file_name=model_file_name,
        reasoning_start_tag=settings.MODEL_REASONING_START_TAG,
        reasoning_stop_tag=settings.MODEL_REASONING_STOP_TAG,
        system_template=settings.MODEL_SYSTEM_TEMPLATE,
        reasoning=settings.MODEL_REASONING,
    )

    return LlamaCppClient(
        base_url=settings.LLAMA_SERVER_BASE_URL,
        model_folder=model_folder,
        model_settings=model_settings,
        timeout=settings.LLAMA_SERVER_TIMEOUT,
    )
