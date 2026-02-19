import sys
from types import ModuleType
from unittest.mock import MagicMock, patch


def test_list_models(client):
    response = client.get("/api/models/")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "default_model" in data
    assert len(data["models"]) > 0


def test_list_strategies(client):
    mock_strategies = ["create-and-refine", "tree-summarization", "async-tree-summarization"]

    # The endpoint lazily imports from chatbot, which requires llama_cpp.
    # Mock the module in sys.modules to avoid the deep import chain.
    mock_module = ModuleType("chatbot.bot.conversation.ctx_strategy")
    mock_module.get_ctx_synthesis_strategies = MagicMock(return_value=mock_strategies)

    with patch.dict(sys.modules, {"chatbot.bot.conversation.ctx_strategy": mock_module}):
        response = client.get("/api/models/strategies")
    assert response.status_code == 200
    data = response.json()
    assert "strategies" in data
    assert "default_strategy" in data
    assert len(data["strategies"]) > 0
