import asyncio

import pytest


def test_generate_answer(openai_client):
    prompt = "What is the capital city of Italy?"
    generated_answer = openai_client.generate_answer(prompt, max_new_tokens=10)
    assert "rome" in generated_answer.lower()


def test_generate_stream_answer(openai_client):
    prompt = "What is the capital city of Italy?"
    generated_answer = openai_client.stream_answer(prompt, max_new_tokens=10)
    assert "rome" in generated_answer.lower()


def test_start_answer_iterator_streamer(openai_client):
    prompt = "What is the capital city of Italy?"
    stream = openai_client.start_answer_iterator_streamer(prompt, max_new_tokens=10)
    generated_answer = ""
    for output in stream:
        token = openai_client.parse_token(output)
        if token:
            generated_answer += token
    assert "rome" in generated_answer.lower()


def test_parse_token(openai_client):
    prompt = "What is the capital city of Italy?"
    stream = openai_client.start_answer_iterator_streamer(prompt, max_new_tokens=10)
    generated_answer = ""
    for output in stream:
        generated_answer += openai_client.parse_token(output)
    assert "rome" in generated_answer.lower()


@pytest.mark.asyncio
async def test_async_generate_answer(openai_client):
    prompt = "What is the capital city of Italy?"
    task = openai_client.async_generate_answer(prompt, max_new_tokens=10)
    generated_answer = await asyncio.gather(task)
    assert "rome" in generated_answer[0].lower()


@pytest.mark.asyncio
async def test_async_start_answer_iterator_streamer(openai_client):
    prompt = "What is the capital city of Italy?"
    task = openai_client.async_start_answer_iterator_streamer(prompt, max_new_tokens=10)
    stream = await asyncio.gather(task)
    generated_answer = ""
    async for output in stream[0]:
        token = output.choices[0].delta.content
        if token is not None:
            generated_answer += token

    assert "rome" in generated_answer.lower()
