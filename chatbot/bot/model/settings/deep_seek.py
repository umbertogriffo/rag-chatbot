from bot.model.base_model import ModelSettings


# https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B#usage-recommendations
class DeepSeekR1SevenSettings(ModelSettings):
    url = "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-7B-Q5_K_M.gguf"
    file_name = "DeepSeek-R1-Distill-Qwen-7B-Q5_K_M.gguf"
    system_template = ""  # Avoid adding a system prompt; all instructions should be contained within the user prompt
    reasoning = True
    reasoning_start_tag = "<think>"
    reasoning_stop_tag = "</think>"
    config = {
        "n_ctx": 4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
        "n_threads": 8,  # The number of CPU threads to use, tailor to your system and the resulting performance
        "n_gpu_layers": 50,  # The number of layers to offload to GPU, if you have GPU acceleration available
    }
    config_answer = {
        "temperature": 0.6,  # Recommended to prevent endless repetitions or incoherent outputs
        "stop": [],
    }
