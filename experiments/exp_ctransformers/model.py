import os
from abc import ABC

import requests
from ctransformers import Config
from tqdm import tqdm

"""
top_k="The top-k value to use for sampling."
top_p="The top-p value to use for sampling."
temperature="The temperature to use for sampling."
repetition_penalty="The repetition penalty to use for sampling."
last_n_tokens="The number of last tokens to use for repetition penalty."
seed="The seed value to use for sampling tokens."
max_new_tokens="The maximum number of new tokens to generate."
stop="A list of sequences to stop generation when encountered."
stream="Whether to stream the generated text."
reset="Whether to reset the model state before generating text."
batch_size="The batch size to use for evaluating tokens in a single prompt."
threads="The number of threads to use for evaluating tokens."
context_length="The maximum context length to use."
gpu_layers="The number of layers to run on GPU."
"""

# Set gpu_layers to the number of layers to offload to GPU.
# Set to 0 if no GPU acceleration is available on your system.

config = Config(
    top_k=40,
    top_p=0.95,
    temperature=0.8,
    repetition_penalty=1.1,
    last_n_tokens=64,
    seed=-1,
    batch_size=8,
    threads=-1,
    max_new_tokens=1024,
    stop=None,
    stream=False,
    reset=True,
    context_length=2048,
    gpu_layers=50,
    mmap=True,
    mlock=False,
)


class ModelSettings(ABC):
    url: str
    name: str
    system_template: str
    prompt_template: str


class ZephyrSettings(ModelSettings):
    url = "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf"
    name = "zephyr-7b-beta.Q4_K_M.gguf"
    system_template = (
        "You are a helpful, respectful and honest assistant. "
        "Answer exactly in few words from the context."
    )
    prompt_template = """<|system|> {system}
Answer the question below from context below:
</s>
<|user|>
{context}
{question}</s>
<|assistant|>
"""


class MistralSettings(ModelSettings):
    url = "https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q4_K_M.gguf"
    name = "mistral-7b-openorca.Q4_K_M.gguf"
    system_template = "You are a helpful, respectful and honest assistant."
    prompt_template = """<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
<|im_start|>system
"""


SUPPORTED_MODELS = {"zephyr": ZephyrSettings, "mistral": MistralSettings}


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
