import time

from llama_cpp import Llama

# CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
    model_path="/home/umberto/PycharmProjects/contextual-chatbot-gpt4all/models/stablelm-zephyr-3b.Q4_K_M.gguf",
    # Download the model file first
    n_ctx=4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
    n_threads=8,  # The number of CPU threads to use, tailor to your system and the resulting performance
    n_gpu_layers=35,  # The number of layers to offload to GPU, if you have GPU acceleration available
    seed=0
)

prompt = "<|user|>\nCreate a regex to extract dates from logs in Python<|endoftext|>\n<|assistant|>"

start_time = time.time()

output = llm(prompt, max_tokens=256, echo=True)
took = time.time() - start_time
print(output["choices"][0]["text"].split("<|assistant|>")[-1])

print(f"--- Took {took:.2f} seconds ---")
