from abc import ABC

from ctransformers import Config


class ModelSettings(ABC):
    """
    Config:
    - top_k="The top-k value to use for sampling."
    - top_p="The top-p value to use for sampling."
    - temperature="The temperature to use for sampling."
    - repetition_penalty="The repetition penalty to use for sampling."
    - last_n_tokens="The number of last tokens to use for repetition penalty."
    - seed="The seed value to use for sampling tokens."
    - max_new_tokens="The maximum number of new tokens to generate."
    - stop="A list of sequences to stop generation when encountered."
    - stream="Whether to stream the generated text."
    - reset="Whether to reset the model state before generating text."
    - batch_size="The batch size to use for evaluating tokens in a single prompt."
    - threads="The number of threads to use for evaluating tokens."
    - context_length="The maximum context length to use."
    - gpu_layers="The number of layers to run on GPU."
        - Set gpu_layers to the number of layers to offload to GPU.
        - Set to 0 if no GPU acceleration is available on your system.
    """

    url: str
    file_name: str
    type: str
    system_template: str
    qa_prompt_template: str
    ctx_prompt_template: str
    refine_prompt_template: str
    config: Config


class ZephyrSettings(ModelSettings):
    url = "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf"
    file_name = "zephyr-7b-beta.Q4_K_M.gguf"
    type = "mistral"
    config = Config(
        top_k=40,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.1,
        last_n_tokens=64,
        seed=-1,
        batch_size=8,
        threads=-1,
        max_new_tokens=1024,
        stop=None,
        stream=False,
        reset=True,
        context_length=2048,
        gpu_layers=50,
        mmap=True,
        mlock=False,
    )
    system_template = "You are a helpful, respectful and honest assistant. "
    qa_prompt_template = """<|system|> {system}
Answer the question below:
</s>
<|user|>
{question}</s>
<|assistant|>
"""
    ctx_prompt_template = """<|system|> {system}
Context information is below.
---------------------
{context}
---------------------
</s>
<|user|>
Given the context information and not prior knowledge, answer the question below:
{question}</s>
<|assistant|>
"""
    refine_prompt_template = """<|system|> {system}
The original query is as follows: {question}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer
(only if needed) with some more context below.
---------------------
{context}
---------------------
</s>
<|user|>
Given the new context, refine the original answer to better answer the query.
If the context isn't useful, return the original answer.
Refined Answer:</s>
<|assistant|>
"""


class MistralSettings(ModelSettings):
    url = "https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q4_K_M.gguf"
    file_name = "mistral-7b-openorca.Q4_K_M.gguf"
    type = "mistral"
    config = Config(
        top_k=40,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.1,
        last_n_tokens=64,
        seed=-1,
        batch_size=8,
        threads=-1,
        max_new_tokens=1024,
        stop=None,
        stream=False,
        reset=True,
        context_length=2048,
        gpu_layers=50,
        mmap=True,
        mlock=False,
    )
    system_template = "You are a helpful, respectful and honest assistant."
    prompt_template = """<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
<|im_start|>system
"""
    ctx_prompt_template = """<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the question below:
{question}<|im_end|>
<|im_start|>assistant
<|im_start|>system
"""
    refine_prompt_template = """<|im_start|>system
{system}
<|im_end|>
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
<|im_start|>system
"""


SUPPORTED_MODELS = {"zephyr": ZephyrSettings, "mistral": MistralSettings}


def get_models():
    return list(SUPPORTED_MODELS.keys())


def get_model_setting(model_name: str):
    model_settings = SUPPORTED_MODELS.get(model_name)

    # validate input
    if model_settings is None:
        raise KeyError(model_name + " is a not supported model")

    return model_settings
