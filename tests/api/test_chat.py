"""Tests for the POST /chat/ endpoint."""

from fastapi import status
from starlette.testclient import TestClient


def test_chat_successful_response(client: TestClient):
    """Test successful chat request."""
    response = client.post("/chat/", json={"text": "Hello, how are you?"})

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0


def test_chat_empty_message(client: TestClient):
    """Test chat with empty message."""
    response = client.post("/chat/", json={"text": ""})

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "response" in data


def test_chat_long_message(client: TestClient):
    """Test chat with a longer message."""
    long_text = "Explain quantum computing in detail. " * 10
    response = client.post("/chat/", json={"text": long_text})

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "response" in data
    assert len(data["response"]) > 0


def test_chat_multiple_requests(client: TestClient):
    """Test sending multiple chat requests."""
    for i in range(3):
        response = client.post("/chat/", json={"text": f"Question {i + 1}"})
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "response" in data


def test_chat_invalid_payload_missing_text(client: TestClient):
    """Test chat with missing text field."""
    response = client.post("/chat/", json={"invalid": "data"})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_chat_invalid_payload_wrong_type(client: TestClient):
    """Test chat with wrong data type for text."""
    response = client.post("/chat/", json={"text": 123})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_chat_no_body(client: TestClient):
    """Test chat without request body."""
    response = client.post("/chat/")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
