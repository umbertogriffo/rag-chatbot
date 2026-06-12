import os
from pathlib import Path
from typing import AsyncIterator, Iterator

import requests
from helpers.log import experimental, get_logger
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from schemas.model import ModelSettings
from tqdm import tqdm

from llm_providers.prompt import (
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

logger = get_logger(__name__)


class LlamaCppClient:
    """
    Client for communicating with llama.cpp server via OpenAI REST API.
    """

    def __init__(self, base_url: str, model_folder: Path, model_settings: ModelSettings, timeout: int = 300):
        """
        Initialize the OpenAI REST API client for llama.cpp server.
        https://github.com/ggml-org/llama.cpp/tree/master/tools/server#openai-compatible-api-endpoints

        Args:
            base_url: The base URL of the llama.cpp server (e.g., "http://localhost:8080")
            model_folder: The folder where the model was saved
            model_settings: Model configuration settings
            timeout: Request timeout in seconds (default: 300)
        """
        self.base_url = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
        self.timeout = timeout

        self.model_folder = model_folder
        self.model_settings = model_settings
        self.model_name = self.model_settings.name
        self.model_path = self.model_folder / self.model_settings.file_name

        # Initialize both sync and async clients
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="not-needed",  # llama.cpp server doesn't require authentication
            timeout=self.timeout,
        )

        self.async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="not-needed",
            timeout=self.timeout,
        )

        self._validate_connection()
        self._auto_download()
        self.reload_models_from_disk()

    def _validate_connection(self) -> None:
        """
        Validate that we can connect to the llama.cpp server.

        Raises:
            Exception: If connection to the server fails
        """
        try:
            # Try to list models as a health check
            models = self.client.models.list()
            logger.info(f"Connected to llama.cpp server at {self.base_url}")
            logger.info(f"Available models: {[model.id for model in models.data]}")
        except Exception as e:
            logger.error(f"Failed to connect to llama.cpp server at {self.base_url}: {e}")
            raise RuntimeError(
                f"Cannot connect to llama.cpp server at {self.base_url}. "
                f"Please ensure the server is running and accessible."
            )

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

    def reload_models_from_disk(self):
        """
        Force llama.cpp server to reload list of models from disk.
        """

        reload_url = f"{self.base_url}/models"
        try:
            reload_response = requests.get(
                reload_url,
                params={"reload": 1},
                timeout=self.timeout,
            )
            reload_response.raise_for_status()
            logger.info("Models list reloaded from disk")

        except Exception as e:
            logger.error(f"Failed to reload models from disk: {e}")
            raise RuntimeError("Failed to reload models from disk.")

    def load_model(self) -> None:
        """
        Load the model into the llama.cpp server.
        """

        server_url = self.base_url.removesuffix("/v1")
        load_url = f"{server_url}/models/load"

        try:
            response = requests.post(
                load_url,
                headers={"Content-Type": "application/json"},
                json={"model": self.model_settings.name},
                timeout=self.timeout,
            )
            response.raise_for_status()
            logger.info(f"Model {self.model_settings.file_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_settings.file_name}: {e}")
            raise RuntimeError(f"Failed to load model {self.model_settings.file_name}. ")

    def unload_model(self) -> None:
        """
        Unload the model into the llama.cpp server.
        """

        server_url = self.base_url.removesuffix("/v1")
        unload_url = f"{server_url}/models/unload"

        try:
            response = requests.post(
                unload_url,
                headers={"Content-Type": "application/json"},
                json={"model": self.model_settings.name},
                timeout=self.timeout,
            )
            response.raise_for_status()
            logger.info(f"Model {self.model_settings.file_name} unloaded successfully")
        except Exception as e:
            logger.error(f"Failed to unload model {self.model_settings.file_name}: {e}")
            raise RuntimeError(f"Failed to unload model {self.model_settings.file_name}. ")

    def close(self):
        """
        Closes the OpenAPI client.
        """
        self.client.close()

    def generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generates an answer based on the given prompt using the language model.

        Args:
            prompt: The input prompt for generating the answer
            max_new_tokens: The maximum number of new tokens to generate (default is 512)

        Returns:
            str: The generated answer
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.model_settings.system_template},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                stream=False,
            )

            answer = response.choices[0].message.content or ""
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    async def async_generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generates an answer based on the given prompt using the language model asynchronously.

        Args:
            prompt: The input prompt for generating the answer
            max_new_tokens: The maximum number of new tokens to generate (default is 512)

        Returns:
            str: The generated answer
        """
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.model_settings.system_template},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                stream=False,
            )

            answer = response.choices[0].message.content or ""
            return answer

        except Exception as e:
            logger.error(f"Error generating answer asynchronously: {e}")
            raise

    def stream_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generates an answer by streaming tokens.

        Args:
            prompt: The input prompt for generating the answer
            max_new_tokens: The maximum number of new tokens to generate (default is 512)

        Returns:
            str: The generated answer
        """
        answer = ""
        stream = self.start_answer_iterator_streamer(prompt, max_new_tokens=max_new_tokens)

        for output in stream:
            token = self.parse_token(output)
            if token:
                answer += token
                print(token, end="", flush=True)

        return answer

    def start_answer_iterator_streamer(self, prompt: str, max_new_tokens: int = 512) -> Iterator[ChatCompletionChunk]:
        """
        Start an answer iterator streamer for a given prompt.

        Args:
            prompt: The input prompt for generating the answer
            max_new_tokens: The maximum number of new tokens to generate (default is 512)

        Returns:
            Iterator yielding streaming response chunks
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.model_settings.system_template},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                stream=True,
            )

            return stream

        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            raise

    async def async_start_answer_iterator_streamer(
        self, prompt: str, max_new_tokens: int = 512
    ) -> AsyncIterator[ChatCompletionChunk]:
        """
        Asynchronously start an answer iterator streamer for streaming response generation.

        Args:
            prompt: The input prompt for generating the answer
            max_new_tokens: The maximum number of new tokens to generate (default is 512)

        Returns:
            AsyncIterator yielding streaming response chunks
        """
        try:
            stream = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.model_settings.system_template},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                stream=True,
            )

            return stream

        except Exception as e:
            logger.error(f"Error starting async stream: {e}")
            raise

    @experimental
    def retrieve_tools(
        self, prompt: str, max_new_tokens: int = 512, tools: list[dict] = None, tool_choice: str = None
    ) -> list[dict] | None:
        """
        Retrieves tools based on the given prompt using the language model.

        Args:
            prompt: The input prompt for retrieving tools
            max_new_tokens: The maximum number of new tokens to generate (default is 512)
            tools: A list of tools that can be used by the language model
            tool_choice: The specific tool to use. If None, the tool choice is set to "auto"

        Returns:
            list[dict] | None: A list of tool calls made by the language model, or None if no tools were called
        """
        # Convert tool_choice format if needed
        formatted_tool_choice = {"type": "function", "function": {"name": tool_choice}} if tool_choice else "auto"

        try:
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": TOOL_SYSTEM_TEMPLATE},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                stream=False,
                tools=tools,
                tool_choice=formatted_tool_choice,
            )

            # Extract tool calls from response
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                # Convert to dictionary format expected by the application
                return [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            return None

        except Exception as e:
            logger.error(f"Error retrieving tools: {e}")
            raise

    @staticmethod
    def parse_token(token: ChatCompletionChunk) -> str:
        """
        Parse a streaming token to extract content.

        Args:
            token: The streaming response chunk

        Returns:
            str: The content from the token, or empty string if no content
        """
        if token.choices and len(token.choices) > 0:
            delta = token.choices[0].delta
            if delta and delta.content:
                return delta.content
        return ""

    # Static prompt generation methods - unchanged from LamaCppClient
    @staticmethod
    def generate_qa_prompt(question: str) -> str:
        """
        Generates a question-answering (QA) prompt using predefined templates.

        Args:
            question: The question for which the prompt is generated

        Returns:
            str: The generated QA prompt
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
            question: The question for which the prompt is generated
            context: The context information for the prompt

        Returns:
            str: The generated context-based prompt
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
            question: The question for which the prompt is generated
            context: The context information for the prompt
            existing_answer: The existing answer to be refined

        Returns:
            str: The generated refined prompt
        """
        return generate_refined_ctx_prompt(
            template=REFINED_CTX_PROMPT_TEMPLATE,
            question=question,
            context=context,
            existing_answer=existing_answer,
        )

    @staticmethod
    def generate_refined_question_conversation_awareness_prompt(question: str, chat_history: str) -> str:
        """
        Generates a refined question prompt with conversation awareness.

        Args:
            question: The question to be refined
            chat_history: The conversation history

        Returns:
            str: The generated conversation-aware prompt
        """
        return generate_conversation_awareness_prompt(
            template=REFINED_QUESTION_CONVERSATION_AWARENESS_PROMPT_TEMPLATE,
            question=question,
            chat_history=chat_history,
        )

    @staticmethod
    def generate_refined_answer_conversation_awareness_prompt(question: str, chat_history: str) -> str:
        """
        Generates a refined answer prompt with conversation awareness.

        Args:
            question: The question for the prompt
            chat_history: The conversation history

        Returns:
            str: The generated conversation-aware prompt
        """
        return generate_conversation_awareness_prompt(
            template=REFINED_ANSWER_CONVERSATION_AWARENESS_PROMPT_TEMPLATE,
            question=question,
            chat_history=chat_history,
        )
