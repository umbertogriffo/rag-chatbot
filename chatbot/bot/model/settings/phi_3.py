from bot.client.llm_client import LlmClientType
from bot.model.model import Model


class PhiThreeSettings(Model):
    url = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
    file_name = "Phi-3-mini-4k-instruct-q4.gguf"
    clients = [LlmClientType.LAMA_CPP]
    config = {
        "n_ctx": 4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
        "n_threads": 8,  # The number of CPU threads to use, tailor to your system and the resulting performance
        "n_gpu_layers": 50,  # The number of layers to offload to GPU, if you have GPU acceleration available
    }
    config_answer = {"temperature": 0.7, "stop": []}
    system_template = "You are a helpful, respectful and honest assistant. "
    qa_prompt_template = """{system}\n
<|user|>\n Answer the question below:
{question}<|end|>\n<|assistant|>
"""
    ctx_prompt_template = """{system}\n
<|user|>\n Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the question below:
{question}<|end|>\n<|assistant|>
"""
    refined_ctx_prompt_template = """{system}\n
<|user|>\n {question}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer
(only if needed) with some more context below.
---------------------
{context}
---------------------
Given the new context, refine the original answer to better answer the query.
If the context isn't useful, return the original answer.
Refined Answer:<|end|>\n<|assistant|>
"""
    refined_question_conversation_awareness_prompt_template = """{system}\n
<|user|>\n Chat History:
---------------------
{chat_history}
---------------------
Follow Up Question: {question}
Given the above conversation and a follow up question, rephrase the follow up question to be a standalone question.
Standalone question:<|end|>\n<|assistant|>
"""

    refined_answer_conversation_awareness_prompt_template = """
<|user|>\n You are engaging in a conversation with a human participant who is unaware that they might be
interacting with a machine. \n
Your goal is to respond in a way that convincingly simulates human-like intelligence and behavior. \n
The conversation should be natural, coherent, and contextually relevant. \n
Chat History:
---------------------
{chat_history}
---------------------
Follow Up Question: {question}\n
Given the context provided in the Chat History and the follow up question, please answer the follow up question above.
If the follow up question isn't correlated to the context provided in the Chat History, please just answer the follow up
question, ignoring the context provided in the Chat History.
Please also don't reformulate the follow up question, and write just a concise answer.
<|end|>\n<|assistant|>
"""
