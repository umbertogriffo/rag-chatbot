import asyncio
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
    model_setting = get_model_setting(ModelType.OPENCHAT_3_5.value)
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
    prompt = "What is the capital city of Italy?"
    generated_answer = lamacpp_client.generate_answer(prompt, max_new_tokens=10)
    assert "rome" in generated_answer.lower()


def test_generate_stream_answer(lamacpp_client):
    prompt = "What is the capital city of Italy?"
    generated_answer = lamacpp_client.stream_answer(prompt, max_new_tokens=10)
    assert "rome" in generated_answer.lower()


def test_start_answer_iterator_streamer(lamacpp_client):
    prompt = "What is the capital city of Italy?"
    stream = lamacpp_client.start_answer_iterator_streamer(prompt, max_new_tokens=10)
    generated_answer = ""
    for output in stream:
        generated_answer += output["choices"][0]["text"]
    assert "rome" in generated_answer.lower()


def test_parse_token(lamacpp_client):
    prompt = "What is the capital city of Italy?"
    stream = lamacpp_client.start_answer_iterator_streamer(prompt, max_new_tokens=10)
    generated_answer = ""
    for output in stream:
        generated_answer += lamacpp_client.parse_token(output)
    assert "rome" in generated_answer.lower()


@pytest.mark.asyncio
async def test_async_generate_answer(lamacpp_client):
    prompt = "What is the capital city of Italy?"
    task = lamacpp_client.async_generate_answer(prompt, max_new_tokens=10)
    generated_answer = await asyncio.gather(task)
    assert "rome" in generated_answer[0].lower()


@pytest.mark.asyncio
async def test_async_start_answer_iterator_streamer(lamacpp_client):
    prompt = "What is the capital city of Italy?"
    task = lamacpp_client.async_start_answer_iterator_streamer(prompt, max_new_tokens=10)
    stream = await asyncio.gather(task)
    generated_answer = ""
    for output in stream[0]:
        generated_answer += output["choices"][0]["text"]
    assert "rome" in generated_answer.lower()
