from bot.client.llm_client import LlmClientType
from bot.model.model import Model


class DolphinSettings(Model):
    url = "https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-dpo-laser-GGUF/resolve/main/dolphin-2.6-mistral-7b-dpo-laser.Q4_K_M.gguf"
    file_name = "dolphin-2.6-mistral-7b-dpo-laser.Q4_K_M.gguf"
    clients = [LlmClientType.LAMA_CPP]
    config = {
        "n_ctx": 4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
        "n_threads": 8,  # The number of CPU threads to use, tailor to your system and the resulting performance
        "n_gpu_layers": -1,  # The number of layers to offload to GPU, if you have GPU acceleration available
    }
    config_answer = {"temperature": 0.7, "stop": ["<|im_end|>"]}
    system_template = "You are a helpful, respectful and honest assistant."
    qa_prompt_template = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

    ctx_prompt_template = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the question below:
{question}<|im_end|>
<|im_start|>assistant
"""
    refined_ctx_prompt_template = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
The original query is as follows: {question}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer
(only if needed) with some more context below.
---------------------
{context}
---------------------
Given the new context, refine the original answer to better answer the query.
If the context isn't useful, return the original answer.
Refined Answer:<|im_end|>
<|im_start|>assistant
"""
    refined_question_conversation_awareness_prompt_template = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
Chat History:
---------------------
{chat_history}
---------------------
Follow Up Question: {question}
Given the above conversation and a follow up question, rephrase the follow up question to be a standalone question.
Standalone question:<|im_end|>
<|im_start|>assistant
"""

    refined_answer_conversation_awareness_prompt_template = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
Chat History:
---------------------
{chat_history}
---------------------
Considering the context provided in the Chat History, answer the question below with conversation awareness:
{question}<|im_end|>
<|im_start|>assistant
"""
