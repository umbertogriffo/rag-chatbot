import pytest
import requests
from unittest.mock import patch, MagicMock
from chatbot.bot.client.open_router_client import OpenRouterClient

@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

def test_open_router_client_initialization(mock_env_vars):
    """
    Tests that the OpenRouterClient initializes correctly.
    """
    client = OpenRouterClient()
    assert client.api_key == "test-key"
    assert client.model == "meta-llama/llama-3-8b-instruct"

@patch("requests.post")
def test_start_answer_iterator_streamer_success(mock_post, mock_env_vars):
    """
    Tests the streaming functionality of the OpenRouterClient with a mocked successful API response.
    """
    # Mock the API response
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.iter_content.return_value = [
        b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n',
        b'data: [DONE]\n\n'
    ]
    mock_post.return_value = mock_response

    client = OpenRouterClient()
    prompt = "Test prompt"

    # Get the streamer iterator
    streamer = client.start_answer_iterator_streamer(prompt)

    # Consume the iterator and check the content
    result = "".join(client.parse_token(chunk) for chunk in streamer)

    assert result == "Hello world"
    mock_post.assert_called_once()

@patch("requests.post")
def test_start_answer_iterator_streamer_api_error(mock_post, mock_env_vars):
    """
    Tests the streaming functionality when the API returns an error.
    """
    # Mock an API error response
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError
    mock_post.return_value = mock_response

    client = OpenRouterClient()
    prompt = "Test prompt"

    # The iterator should be empty and not raise an exception
    streamer = client.start_answer_iterator_streamer(prompt)
    result = list(streamer)

    assert result == []
    mock_post.assert_called_once()

def test_parse_token():
    """
    Tests the token parsing logic.
    """
    # Valid token
    assert OpenRouterClient.parse_token('data: {"choices": [{"delta": {"content": "test"}}]}\n\n') == "test"

    # Done token
    assert OpenRouterClient.parse_token('data: [DONE]\n\n') == ""

    # Empty content
    assert OpenRouterClient.parse_token('data: {"choices": [{"delta": {}}]}\n\n') == ""

    # Invalid JSON
    assert OpenRouterClient.parse_token('data: {invalid json}\n\n') == ""

    # Non-data line
    assert OpenRouterClient.parse_token('some other line\n\n') == ""
