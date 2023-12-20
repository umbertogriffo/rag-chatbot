import os
from abc import ABC
from pathlib import Path
from typing import Dict

import requests
from exp_lama_cpp.prompts import generate_prompt, generate_summarization_prompt
from llama_cpp import Llama
from tqdm import tqdm


class ModelSettings(ABC):
    url: str
    file_name: str
    system_template: str
    prompt_template: str
    summarization_template: str
    config: Dict


class StableLMZephyrSettings(ModelSettings):
    url = "https://huggingface.co/TheBloke/stablelm-zephyr-3b-GGUF/resolve/main/stablelm-zephyr-3b.Q5_K_M.gguf"
    file_name = "stablelm-zephyr-3b.Q5_K_M.gguf"
    config = {
        "n_ctx": 4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
        "n_threads": 8,  # The number of CPU threads to use, tailor to your system and the resulting performance
        "n_gpu_layers": 35,  # The number of layers to offload to GPU, if you have GPU acceleration available
    }

    system_template = "You are a helpful, respectful and honest assistant. "
    prompt_template = """<|user|>Answer the question below:
{question}<|endoftext|>
<|assistant|>
"""
    summarization_template = """<|user|>Create a concise and comprehensive summary of the provided text, 
ensuring that key information, concepts, and code snippets are retained. 
Do not omit or shorten the code snippets, as they are crucial for a comprehensive understanding of the content. 
"{text}"
CONCISE SUMMARY:<|endoftext|>
<|assistant|>
"""


SUPPORTED_MODELS = {"stablelm-zephyr": StableLMZephyrSettings}


def get_models():
    return list(SUPPORTED_MODELS.keys())


def get_model_setting(model_name: str):
    model_settings = SUPPORTED_MODELS.get(model_name)

    # validate input
    if model_settings is None:
        raise KeyError(model_name + " is a not supported model")

    return model_settings


class Model:
    """
    This Model class encapsulates the initialization of the language model, as well as the generation of
    prompts and outputs.
    You can create an instance of this class and use its methods to handle the specific tasks you need.
    """

    def __init__(self, model_folder: Path, model_settings: ModelSettings):
        self.model_settings = model_settings
        self.model_path = model_folder / self.model_settings.file_name
        self.prompt_template = self.model_settings.prompt_template
        self.summarization_template = self.model_settings.summarization_template
        self.system_template = self.model_settings.system_template

        self._auto_download()

        self.llm = Llama(model_path=str(self.model_path), **self.model_settings.config)

    def _auto_download(self) -> None:
        """
        Downloads a model file based on the provided name and saves it to the specified path.

        Returns:
            None

        Raises:
            Any exceptions raised during the download process will be caught and printed, but not re-raised.

        This function fetches model settings using the provided name, including the model's URL, and then downloads
        the model file from the URL. The download is done in chunks, and a progress bar is displayed to visualize
        the download process.

        """
        file_name = self.model_settings.file_name
        url = self.model_settings.url

        if not os.path.exists(self.model_path):
            # send a GET request to the URL to download the file.
            # Stream it while downloading, since the file is large

            try:
                response = requests.get(url, stream=True)
                # open the file in binary mode and write the contents of the response
                # in chunks.
                with open(self.model_path, "wb") as f:
                    for chunk in tqdm(response.iter_content(chunk_size=8912)):
                        if chunk:
                            f.write(chunk)

            except Exception as e:
                print(f"=> Download Failed. Error: {e}")
                return

            print(f"=> Model: {file_name} downloaded successfully 🥳")

    def generate_prompt(self, question):
        return generate_prompt(
            template=self.prompt_template,
            system=self.system_template,
            question=question,
        )

    def generate_summarization_prompt(self, text):
        return generate_summarization_prompt(
            template=self.summarization_template, text=text
        )

    def generate_answer(self, prompt: str, max_new_tokens: int = 1024) -> str:
        output = self.llm(prompt, max_tokens=max_new_tokens, echo=True)
        return output["choices"][0]["text"].split("<|assistant|>")[-1]

    def start_answer_iterator_streamer(self, prompt: str, max_new_tokens: int = 1024):
        stream = self.llm.create_completion(
            prompt, max_tokens=max_new_tokens, temperature=0.8, stream=True
        )
        return stream
