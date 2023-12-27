from unittest.mock import patch

import pytest
import torch
from bot.client.ctransformers_client import CtransformersClient
from bot.model.model_settings import ModelType, get_model_setting


@pytest.fixture
def valid_model_settings():
    config = {
        "top_k": 40,
        "top_p": 0.95,
        "temperature": 0.7,
        "repetition_penalty": 1.1,
        "last_n_tokens": 64,
        "seed": -1,
        "batch_size": 8,
        "threads": -1,
        "max_new_tokens": 1024,
        "stop": None,
        "stream": False,
        "reset": True,
        "context_length": 2048,
        "gpu_layers": 0,
        "mmap": True,
        "mlock": False,
    }
    model_setting = get_model_setting(ModelType.ZEPHYR.value)
    with patch.object(model_setting, "config", config):
        return model_setting


@pytest.fixture
def invalid_model_settings():
    return get_model_setting(ModelType.OPENCHAT.value)


@pytest.fixture
def ctransformers_client(mock_model_folder, valid_model_settings):
    return CtransformersClient(mock_model_folder, valid_model_settings)


def test_init_raises_value_error_for_invalid_client_type(mock_model_folder, invalid_model_settings):
    with pytest.raises(ValueError, match="openchat_3.5.Q4_K_M.gguf is a not supported by the ctransformers client."):
        CtransformersClient(mock_model_folder, invalid_model_settings)


def test_encode_prompt(ctransformers_client):
    prompt = "Test prompt"
    encoded_prompt = ctransformers_client._encode_prompt(prompt)
    assert encoded_prompt is not None


def test_decode_answer(ctransformers_client):
    prompt_ids = torch.tensor([[1, 2, 3]])
    answer_ids = torch.tensor([[4, 5, 6]])

    decoded_answer = ctransformers_client._decode_answer(prompt_ids, answer_ids)
    assert decoded_answer is not None


def test_generate_answer(ctransformers_client):
    prompt = "Tell me a joke"
    generated_answer = ctransformers_client.generate_answer(prompt, max_new_tokens=10)
    assert generated_answer is not None
