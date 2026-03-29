import asyncio

import pytest


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
        generated_answer += output["choices"][0]["delta"].get("content", "")
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
        generated_answer += output["choices"][0]["delta"].get("content", "")
    assert "rome" in generated_answer.lower()
