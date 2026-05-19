import asyncio
from typing import Any, AsyncIterator, Iterator

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

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
from helpers.log import experimental, get_logger

logger = get_logger(__name__)


class OpenAIClient:
    """
    Client for communicating with llama.cpp server via OpenAI-compatible API.
    
    This client replaces the direct llama-cpp-python integration with HTTP-based
    communication to a llama.cpp server, providing better scalability and deployment flexibility.
    """

    def __init__(self, base_url: str, model_name: str, model_settings: ModelSettings, timeout: int = 300):
        """
        Initialize the OpenAI-compatible client for llama.cpp server.

        Args:
            base_url: The base URL of the llama.cpp server (e.g., "http://localhost:8080")
            model_name: The name of the model loaded on the server
            model_settings: Model configuration settings
            timeout: Request timeout in seconds (default: 300)
        """
        self.base_url = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
        self.model_name = model_name
        self.model_settings = model_settings
        self.timeout = timeout

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

        # Validate connection to server
        self._validate_connection()

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
            raise Exception(
                f"Cannot connect to llama.cpp server at {self.base_url}. "
                f"Please ensure the server is running and accessible. Error: {e}"
            )

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
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.model_settings.system_template},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                stream=False,
                **self.model_settings.config_answer,
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
            response: ChatCompletion = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.model_settings.system_template},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                stream=False,
                **self.model_settings.config_answer,
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

    def start_answer_iterator_streamer(
        self, prompt: str, max_new_tokens: int = 512
    ) -> Iterator[ChatCompletionChunk]:
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
                **self.model_settings.config_answer,
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
                **self.model_settings.config_answer,
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
        formatted_tool_choice = (
            {"type": "function", "function": {"name": tool_choice}} if tool_choice else "auto"
        )

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
                **self.model_settings.config_answer,
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
                        }
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
