import pytest
import torch
from bot.client.ctransformers_client import CtransformersClient
from bot.model.model_settings import ModelType, get_model_setting


@pytest.fixture
def valid_model_settings():
    return get_model_setting(ModelType.ZEPHYR.value)


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
