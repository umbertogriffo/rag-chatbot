import dataclasses
import os
from abc import ABC
from dataclasses import dataclass
from typing import Optional

import requests
from tqdm import tqdm

# Check https://github.com/nomic-ai/gpt4all for the latest models.


@dataclass
class GenerateConfig:
    max_tokens: int = 200
    temp: float = 0.7
    top_k: int = 40
    top_p: float = 0.4
    repeat_penalty: float = 1.18
    repeat_last_n: int = 64
    n_batch: int = 8
    n_predict: Optional[int] = None
    streaming: bool = False


def get_generate_config():
    return dataclasses.asdict(GenerateConfig())


class ModelSettings(ABC):
    url: str
    name: str
    system_template: str
    prompt_template: str
    device = "cpu"


class WizardSettings(ModelSettings):
    url = "https://huggingface.co/TheBloke/WizardLM-13B-V1.2-GGUF/resolve/main/wizardlm-13b-v1.2.Q4_0.gguf"
    name = "wizardlm-13b-v1.2.Q4_0.gguf"
    system_template = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    prompt_template = "USER: {0}\nASSISTANT: "


class MistralSettings(ModelSettings):
    url = "https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q4_0.gguf"
    name = "mistral-7b-openorca.Q4_0.gguf"
    system_template = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    # https://github.com/nomic-ai/gpt4all/issues/1614
    prompt_template = """### Human:\n{0}\n### Assistant:"""
    # Nomic Vulkan support for Q4_0, Q6 quantizations in GGUF.
    device = "gpu"


SUPPORTED_MODELS = {"wizard": WizardSettings, "mistral": MistralSettings}


def get_models():
    return list(SUPPORTED_MODELS.keys())


def get_model_setting(model_name: str):
    model_settings = SUPPORTED_MODELS.get(model_name)

    # validate input
    if model_settings is None:
        raise KeyError(model_name + " is a not supported model")

    return model_settings


def auto_download(model_settings: ModelSettings, download_path: str) -> None:
    """
    Downloads a model file based on the provided name and saves it to the specified path.

    Args:
        model_settings (ModelSettings): The settings of the model to download.
        download_path (str): The path where the downloaded model file will be saved.

    Returns:
        None

    Raises:
        Any exceptions raised during the download process will be caught and printed, but not re-raised.

    This function fetches model settings using the provided name, including the model's URL, and then downloads
    the model file from the URL. The download is done in chunks, and a progress bar is displayed to visualize
    the download process.

    """
    model_name = model_settings.name
    url = model_settings.url

    if not os.path.exists(download_path):
        # send a GET request to the URL to download the file.
        # Stream it while downloading, since the file is large

        try:
            response = requests.get(url, stream=True)
            # open the file in binary mode and write the contents of the response
            # in chunks.
            with open(download_path, "wb") as f:
                for chunk in tqdm(response.iter_content(chunk_size=8912)):
                    if chunk:
                        f.write(chunk)

        except Exception as e:
            print(f"=> Download Failed. Error: {e}")
            return

        print(f"=> Model: {model_name} downloaded successfully 🥳")
