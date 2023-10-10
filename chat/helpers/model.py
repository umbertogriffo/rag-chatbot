import os
from abc import ABC
from typing import List

import requests
from langchain.callbacks import (
    FinalStreamingStdOutCallbackHandler,
    StreamingStdOutCallbackHandler,
)
from langchain.llms import GPT4All
from tqdm import tqdm

# Check https://github.com/nomic-ai/gpt4all for the latest models.


class ModelSettings(ABC):
    url: str
    name: str
    template: str
    n_ctx: int = 512
    n_predict: int = 256
    temperature: float = 0.8
    repeat_penalty: float = 1.3
    answer_prefix_tokens: List[str]


class WizardSettings(ModelSettings):
    url = "http://gpt4all.io/models/ggml-wizardLM-7B.q4_2.bin"
    name = "ggml-wizardLM-7B.q4_2.bin"
    template = """
    You are an exceptional Senior Software Engineer that gently answer technical questions.
    ---
    Question: {question}
    Answer:"""
    n_ctx = 2048
    n_predict = 256
    temperature: float = 0.7
    repeat_penalty: float = 1.1
    answer_prefix_tokens = ["The", "answer", ":"]


class Llama2Settings(ModelSettings):
    url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin"
    name = "llama-2-7b-chat.ggmlv3.q4_0.bin"
    template = """
    <s>[INST] <<SYS>>
    You are an exceptional Senior Software Engineer that gently answer technical questions.
    <</SYS>>
    {question} [/INST]</s>"""
    n_ctx = 2048
    n_predict = 256
    temperature: float = 0.7
    repeat_penalty: float = 1.18
    answer_prefix_tokens = [" "]


SUPPORTED_MODELS = {
    "wizard": WizardSettings,
    "llama2": Llama2Settings,
}


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


def load_gpt4all(
    model_path: str,
    answer_prefix_tokens: List[str],
    n_ctx: int,
    n_predict: int,
    temperature: float,
    repeat_penalty: float,
    n_threads: int = 4,
    streaming: bool = True,
    verbose: bool = True,
) -> GPT4All:
    """
    Loads the GPT4All model using the LangChain library.

    The LangChain library utilizes the `PyLLaMAcpp` module to load the converted `GPT4All` weights.

    Parameters:
    ----------
    model_path : str
        The path to the GPT4All model.

    answer_prefix_tokens: List[str]
        Token sequence that prefixes the anwer.

    n_ctx: int
        Token context window.

    n_predict: int
        The maximum number of tokens to generate.

    temperature : float
        The temperature to use for sampling.

    repeat_penalty : float
        The penalty to apply to repeated tokens.

    n_threads : int, optional
        The number of threads to use (default is 4).

    streaming : bool
        Whether to stream the results or not. (default is True).
    verbose : bool
        Whether be verbose or not. (default is True).

    Returns:
    -------
    GPT4All
        The loaded GPT4All model.

    """
    callbacks = [StreamingStdOutCallbackHandler()]
    # callbacks = [
    #     FinalStreamingStdOutCallbackHandler(answer_prefix_tokens=answer_prefix_tokens)
    # ]
    llm = GPT4All(
        model=model_path,
        streaming=streaming,
        backend="llama",
        f16_kv=True,
        callbacks=callbacks,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=1,
        n_predict=n_predict,
        temp=temperature,
        repeat_penalty=repeat_penalty,
        verbose=verbose,
    )
    return llm
