"""Tests for the WebSocket chat streaming endpoint."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx_ws import aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

pytestmark = pytest.mark.asyncio


def _make_ws_client(app):
    """Create a fresh httpx AsyncClient with the ASGI WebSocket transport."""
    return httpx.AsyncClient(transport=ASGIWebSocketTransport(app=app), base_url="http://test")


class TestChatStreamWebSocket:
    async def test_websocket_streams_tokens(self):
        """The WebSocket endpoint should yield individual tokens then a done=True message."""
        from main import app  # noqa: PLC0415

        fake_stream = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
        ]

        mock_llm = MagicMock()
        mock_llm.async_start_answer_iterator_streamer = AsyncMock(return_value=iter(fake_stream))
        mock_llm.parse_token.side_effect = lambda o: o["choices"][0]["delta"].get("content", "")

        messages = []
        with patch("api.endpoints.chat_stream._get_llm_client", return_value=mock_llm):
            async with _make_ws_client(app) as client:
                async with aconnect_ws("http://test/api/chat/stream", client) as ws:
                    await ws.send_text(json.dumps({"text": "Say hello", "rag": False}))
                    while True:
                        msg = await ws.receive_text()
                        data = json.loads(msg)
                        messages.append(data)
                        if data.get("done"):
                            break

        tokens = [m["token"] for m in messages if not m.get("done")]
        assert tokens == ["Hello", " world"]
        assert messages[-1]["done"] is True

    async def test_websocket_returns_error_on_exception(self):
        """When the LLM raises an exception the endpoint should send an error message."""
        from main import app  # noqa: PLC0415

        mock_llm = MagicMock()
        mock_llm.async_start_answer_iterator_streamer = AsyncMock(side_effect=RuntimeError("LLM failure"))

        data = {}
        with patch("api.endpoints.chat_stream._get_llm_client", return_value=mock_llm):
            async with _make_ws_client(app) as client:
                async with aconnect_ws("http://test/api/chat/stream", client) as ws:
                    await ws.send_text(json.dumps({"text": "fail", "rag": False}))
                    msg = await ws.receive_text()
                    data = json.loads(msg)

        assert data["done"] is True
        assert "LLM failure" in data.get("error", "")
