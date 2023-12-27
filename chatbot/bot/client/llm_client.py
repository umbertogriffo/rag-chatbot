import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from bot.client.prompt import (
    generate_conversation_awareness_prompt,
    generate_ctx_prompt,
    generate_qa_prompt,
    generate_refined_ctx_prompt,
)
from bot.model.model import Model


class LlmClientType(Enum):
    CTRANSFORMERS = "ctransformers"
    LAMA_CPP = "lama_cpp"


class LlmClient(ABC):
    """
    Abstract base class for implementing language model clients.

    Args:
        model_folder (Path): The folder where the model is stored.
        model_settings (ModelSettings): Settings for the language model.

    Attributes:
        model_settings (ModelSettings): Settings for the language model.
        model_folder (Path): The folder where the model is stored.
        model_path (Path): The full path to the language model file.
        llm: Loaded language model.
        tokenizer: Loaded tokenizer.

    Raises:
        NotImplementedError: If any abstract method is not implemented by the subclass.
    """

    def __init__(self, model_folder: Path, model_settings: Model):
        self.model_settings = model_settings
        self.model_folder = model_folder
        self.model_path = self.model_folder / self.model_settings.file_name

        self._auto_download()

        self.llm = self._load_llm()
        self.tokenizer = self._load_tokenizer()

    @abstractmethod
    def _load_llm(self) -> Any:
        """
        Abstract method to load the language model.
        """
        raise NotImplementedError

    @abstractmethod
    def _load_tokenizer(self) -> Any:
        """
        Abstract method to load the tokenizer.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Abstract method to generate an answer for a given prompt.

        Args:
            prompt (str): The input prompt for generating the answer.
            max_new_tokens (int): The maximum number of new tokens to generate (default is 1000).

        Returns:
            str: The generated answer.
        """
        raise NotImplementedError

    @abstractmethod
    async def async_generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        This method should be implemented to asynchronously generate an answer based on the given prompt.

        Args:
            prompt (str): The input prompt for generating the answer.
            max_new_tokens (int): The maximum number of new tokens to generate (default is 1000).

        Returns:
            str: The generated answer.

        """

        raise NotImplementedError

    @abstractmethod
    def stream_answer(self, prompt: str, skip_prompt: bool = True, max_new_tokens: int = 512) -> str:
        """
        Abstract method to stream the generation of an answer for a given prompt.

        Args:
            prompt (str): The input prompt for generating the answer.
            skip_prompt (bool): Whether to skip the prompt tokens during streaming (default is True).
            max_new_tokens (int): The maximum number of new tokens to generate (default is 1000).

        Returns:
            str: The generated answer.
        """
        raise NotImplementedError

    @abstractmethod
    def start_answer_iterator_streamer(self, prompt: str, skip_prompt: bool = True, max_new_tokens: int = 512) -> Any:
        """
        Abstract method to start an answer iterator streamer for a given prompt.

        Args:
            prompt (str): The input prompt for generating the answer.
            skip_prompt (bool): Whether to skip the prompt tokens during streaming (default is True).
            max_new_tokens (int): The maximum number of new tokens to generate (default is 1000).

        """
        raise NotImplementedError

    @abstractmethod
    async def async_start_answer_iterator_streamer(
        self, prompt: str, skip_prompt: bool = True, max_new_tokens: int = 512
    ) -> Any:
        """
        This abstract method should be implemented to asynchronously start an answer iterator streamer,
        providing a flexible way to generate answers in a streaming fashion based on the given prompt.

        Args:
            prompt (str): The input prompt for generating the answer.
            skip_prompt (bool): Whether to skip the prompt tokens during streaming (default is True).
            max_new_tokens (int): The maximum number of new tokens to generate (default is 1000).

        """

        raise NotImplementedError

    @abstractmethod
    def parse_token(self, token):
        raise NotImplementedError

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

    def generate_qa_prompt(self, question: str) -> str:
        """
        Generates a question-answering (QA) prompt using predefined templates.

        Args:
            question (str): The question for which the prompt is generated.

        Returns:
            str: The generated QA prompt.
        """
        return generate_qa_prompt(
            template=self.model_settings.qa_prompt_template,
            system=self.model_settings.system_template,
            question=question,
        )

    def generate_ctx_prompt(self, question: str, context: str) -> str:
        """
        Generates a context-based prompt using predefined templates.

        Args:
            question (str): The question for which the prompt is generated.
            context (str): The context information for the prompt.

        Returns:
            str: The generated context-based prompt.
        """
        return generate_ctx_prompt(
            template=self.model_settings.ctx_prompt_template,
            system=self.model_settings.system_template,
            question=question,
            context=context,
        )

    def generate_refined_ctx_prompt(self, question: str, context: str, existing_answer: str) -> str:
        """
        Generates a refined prompt for question-answering with existing answer.

        Args:
            question (str): The question for which the prompt is generated.
            context (str): The context information for the prompt.
            existing_answer (str): The existing answer to be refined.

        Returns:
            str: The generated refined prompt.
        """
        return generate_refined_ctx_prompt(
            template=self.model_settings.refined_ctx_prompt_template,
            system=self.model_settings.system_template,
            question=question,
            context=context,
            existing_answer=existing_answer,
        )

    def generate_refined_question_conversation_awareness_prompt(self, question: str, chat_history: str) -> str:
        return generate_conversation_awareness_prompt(
            template=self.model_settings.refined_question_conversation_awareness_prompt_template,
            system=self.model_settings.system_template,
            question=question,
            chat_history=chat_history,
        )

    def generate_refined_answer_conversation_awareness_prompt(self, question: str, chat_history: str) -> str:
        return generate_conversation_awareness_prompt(
            template=self.model_settings.refined_answer_conversation_awareness_prompt_template,
            system=self.model_settings.system_template,
            question=question,
            chat_history=chat_history,
        )
