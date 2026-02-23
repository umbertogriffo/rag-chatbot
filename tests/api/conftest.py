"""Shared fixtures for API tests.

The LLM client is mocked at the sys.modules level *before* the FastAPI app
is imported so that no model download is attempted during testing.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Mock the heavy llm_client module before the FastAPI app imports it.
# ---------------------------------------------------------------------------

_mock_llm_client = MagicMock()
_mock_llm_client.generate_answer.return_value = "Mocked answer"
_mock_llm_client.parse_token.side_effect = lambda output: output["choices"][0]["delta"].get("content", "")

_mock_llm_module = MagicMock()
_mock_llm_module.llm_client = _mock_llm_client

sys.modules.setdefault("llm_client", _mock_llm_module)


def _get_app():
    """Return the FastAPI app, importing it lazily so mocks are in place first."""
    from main import app  # noqa: PLC0415

    return app


@pytest.fixture(autouse=True)
def reset_document_store(tmp_path, monkeypatch):
    """Clear the in-memory document store between tests and redirect DOCS_PATH to a temp dir."""
    from api.endpoints.documents import _documents  # noqa: PLC0415
    from core.config import settings  # noqa: PLC0415

    monkeypatch.setattr(settings, "DOCS_PATH", tmp_path / "docs")
    _documents.clear()
    yield
    _documents.clear()


@pytest_asyncio.fixture
async def async_client():
    app = _get_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
