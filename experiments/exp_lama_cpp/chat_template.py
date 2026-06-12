import time
from pathlib import Path

from llm_providers.llamacpp_client import LlamaCppClient
from schemas.model import ModelSettings

LLAMA_SERVER_BASE_URL = "http://localhost:8080"

if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parents[2]
    model_folder = root_folder / "models"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    model_settings = ModelSettings(
        url="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q5_K_M.gguf",
        name="Llama-3.2-1B-Instruct-Q5_K_M",
        file_name="Llama-3.2-1B-Instruct-Q5_K_M.gguf",
        reasoning_start_tag="<think>",
        reasoning_stop_tag="</think>",
        system_template="",
        reasoning=False,
    )

    llm = LlamaCppClient(
        base_url=LLAMA_SERVER_BASE_URL,
        model_folder=model_folder,
        model_settings=model_settings,
        timeout=300,
    )

    prompt = "tell me something about Italy"

    start_time = time.time()
    output = llm.generate_answer(prompt, max_new_tokens=512)
    print(output)
    took = time.time() - start_time
    print(f"\n--- Took {took:.2f} seconds ---")

    start_time = time.time()
    stream = llm.start_answer_iterator_streamer(prompt, max_new_tokens=256)
    for output in stream:
        print(output.choices[0].delta.content or "", end="", flush=True)
    took = time.time() - start_time

    print(f"\n--- Took {took:.2f} seconds ---")
