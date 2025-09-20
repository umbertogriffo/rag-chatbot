import os
from pathlib import Path
from typing import Any, Iterator

import requests
from helpers.log import experimental
from llama_cpp import CreateCompletionResponse, CreateCompletionStreamResponse, Llama
from tqdm import tqdm

from bot.client.prompt import (
    CTX_PROMPT_TEMPLATE,
    QA_PROMPT_TEMPLATE,
    REFINED_ANSWER_CONVERSATION_AWARENESS_PROMPT_TEMPLATE,
    REFINED_CTX_PROMPT_TEMPLATE,
    REFINED_QUESTION_CONVERSATION_AWARENESS_PROMPT_TEMPLATE,
    TOOL_SYSTEM_TEMPLATE,
    generate_conversation_awareness_prompt,
    generate_ctx_prompt,
    generate_qa_prompt,
    generate_refined_ctx_prompt,
)
from bot.model.base_model import ModelSettings


class LamaCppClient:
    """
    Class for implementing language model client.
    """

    def __init__(self, model_folder: Path, model_settings: ModelSettings):
        self.model_settings = model_settings
        self.model_folder = model_folder
        self.model_path = self.model_folder / self.model_settings.file_name

        self._auto_download()

        self.llm = self._load_llm()
        # self.tokenizer = self._load_tokenizer()

    def _load_llm(self) -> Any:
        """
        Method to load the language model.
        """
        llm = Llama(model_path=str(self.model_path), **self.model_settings.config)
        return llm

    def _load_tokenizer(self) -> Any:
        """
        Method to load the tokenizer.
        """
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

    def generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generates an answer based on the given prompt using the language model.

        Args:
            prompt (str): The input prompt for generating the answer.
            max_new_tokens (int): The maximum number of new tokens to generate (default is 512).

        Returns:
            str: The generated answer.
        """

        output = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": self.model_settings.system_template},
                {"role": "user", "content": f"{prompt}"},
            ],
            max_tokens=max_new_tokens,
            **self.model_settings.config_answer,
        )

        answer = output["choices"][0]["message"].get("content", "")

        return answer

    async def async_generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generates an answer based on the given prompt using the language model asynchronously.

        Args:
            prompt (str): The input prompt for generating the answer.
            max_new_tokens (int): The maximum number of new tokens to generate (default is 512).

        Returns:
            str: The generated answer.
        """
        return self.generate_answer(prompt, max_new_tokens)

    def stream_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generates an answer by streaming tokens.

        Args:
            prompt (str): The input prompt for generating the answer.
            max_new_tokens (int): The maximum number of new tokens to generate (default is 512).

        Returns:
            str: The generated answer.
        """
        answer = ""
        stream = self.start_answer_iterator_streamer(prompt, max_new_tokens=max_new_tokens)

        for output in stream:
            token = output["choices"][0]["delta"].get("content", "")
            answer += token
            print(token, end="", flush=True)

        return answer

    def start_answer_iterator_streamer(
        self, prompt: str, max_new_tokens: int = 512
    ) -> CreateCompletionResponse | Iterator[CreateCompletionStreamResponse]:
        """
        Abstract method to start an answer iterator streamer for a given prompt.

        Args:
            prompt (str): The input prompt for generating the answer.
            max_new_tokens (int): The maximum number of new tokens to generate (default is 1000).

        """
        stream = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": self.model_settings.system_template},
                {"role": "user", "content": f"{prompt}"},
            ],
            max_tokens=max_new_tokens,
            stream=True,
            **self.model_settings.config_answer,
        )

        return stream

    async def async_start_answer_iterator_streamer(
        self, prompt: str, max_new_tokens: int = 512
    ) -> CreateCompletionResponse | Iterator[CreateCompletionStreamResponse]:
        """
        Abstract method to asynchronously start an answer iterator streamer,
        providing a flexible way to generate answers in a streaming fashion based on the given prompt.

        Args:
            prompt (str): The input prompt for generating the answer.
            max_new_tokens (int): The maximum number of new tokens to generate (default is 1000).

        """
        return self.start_answer_iterator_streamer(prompt, max_new_tokens)

    @experimental
    def retrieve_tools(
        self, prompt: str, max_new_tokens: int = 512, tools: list[dict] = None, tool_choice: str = None
    ) -> list[dict] | None:
        """
        Retrieves tools based on the given prompt using the language model.

        Args:
            prompt (str): The input prompt for retrieving tools.
            max_new_tokens (int): The maximum number of new tokens to generate (default is 512).
            tools (list[dict], optional): A list of tools that can be used by the language model.
            tool_choice (str, optional): The specific tool to use. If None, the tool choice is set to "auto".

        Returns:
            list[dict] | None: A list of tool calls made by the language model, or None if no tools were called.
        """
        tool_choice = {"type": "function", "function": {"name": tool_choice}} if tool_choice else "auto"

        output = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": TOOL_SYSTEM_TEMPLATE},
                {"role": "user", "content": f"{prompt}"},
            ],
            max_tokens=max_new_tokens,
            stream=False,
            tools=tools,
            tool_choice=tool_choice,
            **self.model_settings.config_answer,
        )

        tool_calls = output["choices"][0]["message"].get("tool_calls", None)
        # first_choice_function = output["choices"][0]['message'].get('function_call', None)
        # if first_choice_function:
        #     function_name = first_choice_function.get('name', None)
        #     function_args = first_choice_function.get('arguments', None)
        return tool_calls

    @staticmethod
    def parse_token(token):
        return token["choices"][0]["delta"].get("content", "")

    @staticmethod
    def generate_qa_prompt(question: str) -> str:
        """
        Generates a question-answering (QA) prompt using predefined templates.

        Args:
            question (str): The question for which the prompt is generated.

        Returns:
            str: The generated QA prompt.
        """
        return generate_qa_prompt(
            template=QA_PROMPT_TEMPLATE,
            question=question,
        )

    @staticmethod
    def generate_ctx_prompt(question: str, context: str) -> str:
        """
        Generates a context-based prompt using predefined templates.

        Args:
            question (str): The question for which the prompt is generated.
            context (str): The context information for the prompt.

        Returns:
            str: The generated context-based prompt.
        """
        return generate_ctx_prompt(
            template=CTX_PROMPT_TEMPLATE,
            question=question,
            context=context,
        )

    @staticmethod
    def generate_refined_ctx_prompt(question: str, context: str, existing_answer: str) -> str:
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
            template=REFINED_CTX_PROMPT_TEMPLATE,
            question=question,
            context=context,
            existing_answer=existing_answer,
        )

    @staticmethod
    def generate_refined_question_conversation_awareness_prompt(question: str, chat_history: str) -> str:
        return generate_conversation_awareness_prompt(
            template=REFINED_QUESTION_CONVERSATION_AWARENESS_PROMPT_TEMPLATE,
            question=question,
            chat_history=chat_history,
        )

    @staticmethod
    def generate_refined_answer_conversation_awareness_prompt(question: str, chat_history: str) -> str:
        return generate_conversation_awareness_prompt(
            template=REFINED_ANSWER_CONVERSATION_AWARENESS_PROMPT_TEMPLATE,
            question=question,
            chat_history=chat_history,
        )
