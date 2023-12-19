from bot.model.client.client import LlmClient
from bot.model.model import Model


class StableLMZephyrSettings(Model):
    url = "https://huggingface.co/TheBloke/stablelm-zephyr-3b-GGUF/resolve/main/stablelm-zephyr-3b.Q5_K_M.gguf"
    file_name = "stablelm-zephyr-3b.Q5_K_M.gguf"
    clients = [LlmClient.LAMA_CPP]
    config = {
        "n_ctx": 4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
        "n_threads": 8,  # The number of CPU threads to use, tailor to your system and the resulting performance
        "n_gpu_layers": 35,  # The number of layers to offload to GPU, if you have GPU acceleration available
    }

    system_template = (
        "You are a helpful, respectful and honest assistant. "
    )
    qa_prompt_template = """<|user|>{system}\nAnswer the question below:
{question}<|endoftext|>
<|assistant|>
"""
    ctx_prompt_template = """<|user|>{system}\nContext information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the question below:
{question}<|endoftext|>
<|assistant|>
"""
    refined_ctx_prompt_template = """<|user|>{system}\nThe original query is as follows: {question}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer
(only if needed) with some more context below.
---------------------
{context}
---------------------
Given the new context, refine the original answer to better answer the query.
If the context isn't useful, return the original answer.
Refined Answer:<|endoftext|>
<|assistant|>
"""
    conversation_awareness_prompt_template = """<|user|>{system}\nChat History:
{chat_history}
Follow Up Question: {question}
Given the above conversation and a follow up question, rephrase the follow up question to be a standalone question.
Standalone question:<|endoftext|>
<|assistant|>
"""
