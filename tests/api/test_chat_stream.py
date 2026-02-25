"""Tests for the WebSocket chat streaming endpoint."""
import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient


def test_chat_stream_successful_response(client: TestClient):
    """Test successful WebSocket chat streaming."""
    with client.websocket_connect("/chat/stream") as websocket:
        # Send a chat request
        websocket.send_json({"text": "Hello, how are you?"})

        # Collect streamed tokens
        tokens = []
        done = False

        while not done:
            response = websocket.receive_json()
            done = response.get("done", False)
            if not done and response.get("token"):
                tokens.append(response["token"])

        # Verify we received tokens
        assert len(tokens) > 0
        assert done is True


def test_chat_stream_empty_message(client: TestClient):
    """Test WebSocket with empty message."""
    with client.websocket_connect("/chat/stream") as websocket:
        websocket.send_json({"text": ""})

        # Should still complete successfully
        done = False
        while not done:
            response = websocket.receive_json()
            done = response.get("done", False)

        assert done is True


def test_chat_stream_multiple_requests(client: TestClient):
    """Test sending multiple requests through the same WebSocket."""
    with client.websocket_connect("/chat/stream") as websocket:
        for i in range(3):
            websocket.send_json({"text": f"Question {i + 1}"})

            done = False
            while not done:
                response = websocket.receive_json()
                done = response.get("done", False)

            assert done is True


def test_chat_stream_invalid_payload(client: TestClient):
    """Test WebSocket with invalid payload."""
    with pytest.raises(ValidationError, match="validation error for ChatRequest"):
        with client.websocket_connect("/chat/stream") as websocket:
            # Send invalid data (missing 'text' field)
            websocket.send_json({"invalid": "data"})


def test_chat_stream_connection_and_disconnection(client: TestClient):
    """Test WebSocket connection can be established and closed."""
    with client.websocket_connect("/chat/stream") as websocket:
        # Connection established successfully
        websocket.send_json({"text": "Test message"})

        # Receive at least one response
        response = websocket.receive_json()
        assert response is not None
