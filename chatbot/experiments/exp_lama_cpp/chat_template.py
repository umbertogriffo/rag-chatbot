import time
from pathlib import Path

from bot.client.lama_cpp_client import LamaCppClient
from bot.model.model_registry import get_model_settings

if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent.parent.parent.parent
    model_folder = root_folder / "models"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    model_settings = get_model_settings("llama-3.1")

    llm = LamaCppClient(model_folder, model_settings)

    prompt = "tell me something about Italy"

    start_time = time.time()
    output = llm.generate_answer(prompt, max_new_tokens=512)
    print(output)
    took = time.time() - start_time
    print(f"\n--- Took {took:.2f} seconds ---")

    start_time = time.time()
    stream = llm.start_answer_iterator_streamer(prompt, max_new_tokens=256)
    for output in stream:
        print(output["choices"][0]["delta"].get("content", ""), end="", flush=True)
    took = time.time() - start_time

    print(f"\n--- Took {took:.2f} seconds ---")
