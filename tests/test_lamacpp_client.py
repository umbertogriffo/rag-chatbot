from unittest.mock import patch

import pytest
from bot.client.lama_cpp_client import LamaCppClient
from bot.model.model_settings import ModelType, get_model_setting


@pytest.fixture
def cpu_config():
    config = {
        "n_ctx": 512,
        "n_threads": 2,
        "n_gpu_layers": 0,
    }
    return config


@pytest.fixture
def valid_model_settings():
    model_setting = get_model_setting(ModelType.OPENCHAT.value)
    return model_setting


@pytest.fixture
def invalid_model_settings():
    return get_model_setting(ModelType.ZEPHYR.value)


@pytest.fixture
def lamacpp_client(mock_model_folder, valid_model_settings, cpu_config):
    with patch.object(valid_model_settings, "config", cpu_config):
        return LamaCppClient(mock_model_folder, valid_model_settings)


def test_init_raises_value_error_for_invalid_client_type(mock_model_folder, invalid_model_settings):
    with pytest.raises(ValueError):
        LamaCppClient(mock_model_folder, invalid_model_settings)


def test_generate_answer(lamacpp_client):
    prompt = "Tell me a joke"
    generated_answer = lamacpp_client.generate_answer(prompt, max_new_tokens=10)
    assert generated_answer is not None
