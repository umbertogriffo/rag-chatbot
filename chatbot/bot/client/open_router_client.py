import os
import json
import requests
from typing import Iterator

class OpenRouterClient:
    """
    A client for handling chat completions with the OpenRouter API.
    """
    def __init__(self):
        """
        Initializes the OpenRouterClient, retrieving the API key from environment variables.
        """
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "meta-llama/llama-3-8b-instruct"

    def start_answer_iterator_streamer(self, prompt: str, max_new_tokens: int = 512) -> Iterator[str]:
        """
        Starts a streaming request to the OpenRouter API and returns an iterator over the response chunks.

        Args:
            prompt (str): The user's prompt.
            max_new_tokens (int): The maximum number of tokens to generate.

        Returns:
            An iterator that yields the response content chunks.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": max_new_tokens
        }

        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(data), stream=True)
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    yield chunk.decode("utf-8")

        except requests.exceptions.RequestException as e:
            print(f"An error occurred with the API request: {e}")
            return

    @staticmethod
    def parse_token(token_chunk: str) -> str:
        """
        Parses a token from a streaming chunk.
        The streaming API returns chunks prefixed with 'data: '. This method extracts the JSON content.
        """
        if token_chunk.startswith("data: "):
            content = token_chunk[len("data: "):]
            if content.strip() == "[DONE]":
                return ""
            try:
                json_content = json.loads(content)
                return json_content["choices"][0]["delta"].get("content", "")
            except json.JSONDecodeError:
                return ""
        return ""
