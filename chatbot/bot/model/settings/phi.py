from bot.model.model import ModelSettings


class Phi31Settings(ModelSettings):
    url = "https://huggingface.co/bartowski/Phi-3.1-mini-128k-instruct-GGUF/resolve/main/Phi-3.1-mini-128k-instruct-Q5_K_M.gguf"
    file_name = "Phi-3.1-mini-128k-instruct-Q5_K_M.gguf"
    config = {
        "n_ctx": 4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
        "n_threads": 8,  # The number of CPU threads to use, tailor to your system and the resulting performance
        "n_gpu_layers": 33,  # The number of layers to offload to GPU, if you have GPU acceleration available
    }
    config_answer = {"temperature": 0.7, "stop": []}


class Phi35Settings(ModelSettings):
    url = "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q5_K_M.gguf"
    file_name = "Phi-3.5-mini-instruct-Q5_K_M.gguf.gguf"
    config = {
        "n_ctx": 4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
        "n_threads": 8,  # The number of CPU threads to use, tailor to your system and the resulting performance
        "n_gpu_layers": 33,  # The number of layers to offload to GPU, if you have GPU acceleration available
    }
    config_answer = {"temperature": 0.7, "stop": []}
