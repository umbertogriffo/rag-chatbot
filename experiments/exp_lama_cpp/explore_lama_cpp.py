import time
from pathlib import Path

from exp_lama_cpp.model import Model, get_model_setting

# CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent.parent.parent
    model_folder = root_folder / "models"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    model_settings = get_model_setting("stablelm-zephyr")

    llm = Model(model_folder, model_settings)

    start_time = time.time()
    prompt = llm.generate_summarization_prompt(text="<put the text here>")
    output = llm.generate_answer(prompt, max_new_tokens=512)
    print(output)
    took = time.time() - start_time
    print(f"\n--- Took {took:.2f} seconds ---")

    start_time = time.time()
    stream = llm.start_answer_iterator_streamer(prompt, max_new_tokens=256)
    for output in stream:
        print(output["choices"][0]["text"], end="", flush=True)
    took = time.time() - start_time

    print(f"\n--- Took {took:.2f} seconds ---")
