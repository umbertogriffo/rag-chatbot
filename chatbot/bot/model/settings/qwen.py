from bot.model.base_model import ModelSettings


class Qwen25ThreeSettings(ModelSettings):
    url = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q5_k_m.gguf"
    file_name = "qwen2.5-3b-instruct-q5_k_m.gguf"
    config = {
        "n_ctx": 4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
        "n_threads": 8,  # The number of CPU threads to use, tailor to your system and the resulting performance
        "n_gpu_layers": 50,  # The number of layers to offload to GPU, if you have GPU acceleration available
    }
    config_answer = {"temperature": 0.7, "top_p": 0.95, "stop": []}


class Qwen25ThreeMathReasoningSettings(ModelSettings):
    url = "https://huggingface.co/ugriffo/Qwen2.5-3B-Instruct-Math-Reasoning-GGUF/resolve/main/qwen2.5-3b-instruct-math-reasoning-Q5_K_M.gguf"
    file_name = "qwen2.5-3b-instruct-math-reasoning-Q5_K_M.gguf"
    config = {
        "n_ctx": 4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
        "n_threads": 8,  # The number of CPU threads to use, tailor to your system and the resulting performance
        "n_gpu_layers": 50,  # The number of layers to offload to GPU, if you have GPU acceleration available
    }
    config_answer = {"temperature": 0.7, "top_p": 0.95, "stop": []}


# Qwen3.5 family, including Qwen3.5-35B-A3B, 27B, 122B-A10B and 397B-A17B and the new Small series: Qwen3.5-0.8B, 2B,
# 4B and 9B.
# How to Run Locally Guide - https://unsloth.ai/docs/models/qwen3.5
# llama.cpp reasoning issue - https://github.com/ggml-org/llama.cpp/issues/20182
# Qwen3.5 0.8B, 2B, 4B and 9B, reasoning is disabled by default.


# https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF
class Qwen35ZeroEightSettings(ModelSettings):
    url = "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q5_K_M.gguf"
    file_name = "Qwen3.5-0.8B-Q5_K_M.gguf"
    config = {
        "n_ctx": 4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
        "n_threads": 8,  # The number of CPU threads to use, tailor to your system and the resulting performance
        "n_gpu_layers": 32,  # The number of layers to offload to GPU, if you have GPU acceleration available
        "flash_attn": True,  # Whether to use flash attention, which can speed up inference on compatible hardware
    }
    config_answer = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
        "stop": [],
    }


# https://huggingface.co/unsloth/Qwen3.5-2B-GGUF
class Qwen35TwoSettings(ModelSettings):
    url = "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q5_K_M.gguf"
    file_name = "Qwen3.5-2B-Q5_K_M.gguf"
    config = {
        "n_ctx": 4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
        "n_threads": 8,  # The number of CPU threads to use, tailor to your system and the resulting performance
        "n_gpu_layers": 32,  # The number of layers to offload to GPU, if you have GPU acceleration available
        "flash_attn": True,  # Whether to use flash attention, which can speed up inference on compatible hardware
    }
    config_answer = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
        "stop": [],
    }


# https://huggingface.co/unsloth/Qwen3.5-4B-GGUF
class Qwen35FourSettings(ModelSettings):
    url = "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q5_K_M.gguf"
    file_name = "Qwen3.5-4B-Q5_K_M.gguf"
    config = {
        "n_ctx": 4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
        "n_threads": 8,  # The number of CPU threads to use, tailor to your system and the resulting performance
        "n_gpu_layers": 32,  # The number of layers to offload to GPU, if you have GPU acceleration available
        "flash_attn": True,  # Whether to use flash attention, which can speed up inference on compatible hardware
    }
    config_answer = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
        "stop": [],
    }


# https://huggingface.co/unsloth/Qwen3.5-9B-GGUF
class Qwen35NineSettings(ModelSettings):
    url = "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q3_K_M.gguf"
    file_name = "Qwen3.5-9B-Q3_K_M.gguf"
    config = {
        "n_ctx": 4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
        "n_threads": 8,  # The number of CPU threads to use, tailor to your system and the resulting performance
        "n_gpu_layers": 32,  # The number of layers to offload to GPU, if you have GPU acceleration available
        "flash_attn": True,  # Whether to use flash attention, which can speed up inference on compatible hardware
    }
    config_answer = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
        "stop": [],
    }
