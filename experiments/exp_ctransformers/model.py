import os
from abc import ABC
from pathlib import Path

import requests
from ctransformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Config
from exp_ctransformers.prompts import generate_prompt
from tqdm import tqdm
from transformers import TextStreamer


class ModelSettings(ABC):
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
        Set gpu_layers to the number of layers to offload to GPU.
        Set to 0 if no GPU acceleration is available on your system.
    """

    url: str
    file_name: str
    model_type: str
    system_template: str
    prompt_template: str
    config: Config


class ZephyrSettings(ModelSettings):
    url = "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf"
    file_name = "zephyr-7b-beta.Q4_K_M.gguf"
    model_type = "mistral"
    config = Config(
        top_k=40,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.1,
        last_n_tokens=64,
        seed=-1,
        batch_size=512,
        threads=-1,
        max_new_tokens=1024,
        stop=None,
        stream=False,
        reset=True,
        context_length=3048,
        gpu_layers=50,
        mmap=True,
        mlock=False,
    )
    system_template = "You are a helpful, respectful and honest assistant. "
    prompt_template = """<|system|> {system}
Answer the question below:
</s>
<|user|>
{question}</s>
<|assistant|>
"""


class MistralSettings(ModelSettings):
    url = "https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q4_K_M.gguf"
    file_name = "mistral-7b-openorca.Q4_K_M.gguf"
    model_type = "mistral"
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


class Model:
    """
    This Model class encapsulates the initialization of the language model and tokenizer, as well as the generation of
    prompts and outputs.
    You can create an instance of this class and use its methods to handle the specific tasks you need.
    """

    def __init__(self, model_folder: Path, model_settings: ModelSettings):
        self.model_settings = model_settings
        self.model_path = model_folder / self.model_settings.file_name
        self.prompt_template = self.model_settings.prompt_template
        self.system_template = self.model_settings.system_template

        self._auto_download()

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id=str(model_folder),
            model_file=self.model_settings.file_name,
            model_type=self.model_settings.model_type,
            config=AutoConfig(config=self.model_settings.config),
            hf=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm)

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

    def generate_output(self, prompt: str, max_new_tokens: int = 1000):
        inputs = self.tokenizer(text=prompt, return_tensors="pt").input_ids
        streamer = TextStreamer(tokenizer=self.tokenizer, skip_prompt=True)
        output = self.llm.generate(
            inputs, streamer=streamer, max_new_tokens=max_new_tokens
        )

        return output
