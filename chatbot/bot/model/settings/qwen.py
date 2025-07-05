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
