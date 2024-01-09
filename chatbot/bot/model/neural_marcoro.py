from bot.client.llm_client import LlmClientType
from bot.model.model import Model


class NeuralMarcoroSettings(Model):
    # https://huggingface.co/mlabonne/NeuralMarcoro14-7B
    url = "https://huggingface.co/mlabonne/NeuralMarcoro14-7B-GGUF/resolve/main/neuralmarcoro14-7b.Q4_K_M.gguf"
    file_name = "neuralmarcoro14-7b.Q4_K_M.gguf"
    clients = [LlmClientType.LAMA_CPP]
    config = {
        "n_ctx": 4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
        "n_threads": 8,  # The number of CPU threads to use, tailor to your system and the resulting performance
        "n_gpu_layers": 50,  # The number of layers to offload to GPU, if you have GPU acceleration available
    }
    system_template = "You are a helpful, respectful and honest assistant. "
    qa_prompt_template = """### System:
{system}
### User:
{question}
### Assistant:
"""
    ctx_prompt_template = """### System:
{system}
### User:
Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the question below:
{question}
### Assistant:
"""
    refined_ctx_prompt_template = """### System:
{system}
### User:
The original query is as follows: {question}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer
(only if needed) with some more context below.
---------------------
{context}
---------------------

Given the new context, refine the original answer to better answer the query.
If the context isn't useful, return the original answer.
Refined Answer:
### Assistant:
"""
    refined_question_conversation_awareness_prompt_template = """### System:
{system}\n
### User:
Chat History:
---------------------
{chat_history}
---------------------
Follow Up Question: {question}

Given the above conversation and a follow up question, rephrase the follow up question to be a standalone question.
Standalone question:
### Assistant:
"""

    refined_answer_conversation_awareness_prompt_template = """### System:\n
You are engaging in a conversation with a human participant who is unaware that they might be interacting with a
machine. \n
Your goal is to respond in a way that convincingly simulates human-like intelligence and behavior. \n
The conversation should be natural, coherent, and contextually relevant. \n
### User:
Chat History:
---------------------
{chat_history}
---------------------
Follow Up Question: {question}
Given the context provided in the Chat History and the follow up question, please craft an answer that demonstrates
a deep understanding of the conversation and exhibits conversational awareness.
If the follow up question isn't correlated to the context provided, please answer the Follow Up Question ignoring
the context provided in the Chat History.
Don't reformulate the Follow Up Question, and don't write "Answer:" or "Response:".
### Assistant:
"""
