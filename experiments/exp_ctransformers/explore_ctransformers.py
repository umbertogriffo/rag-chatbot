import time
from pathlib import Path

from exp_ctransformers.model import Model, get_model_setting

if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent.parent.parent
    model_folder = root_folder / "models"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    model_settings = get_model_setting("zephyr")

    llm = Model(model_folder, model_settings)

    # question_p = """What is the date for announcement"""
    # context_p = """ On August 10 said that its arm JSW Neo Energy has agreed to buy a portfolio of 1753 mega watt
    # renewable energy generation capacity from Mytrah Energy India Pvt Ltd for Rs 10,530 crore."""

    question_p = """Create a regex to extract dates from logs in Python"""

    prompt = llm.generate_prompt(question=question_p)

    start_time = time.time()
    _ = llm.generate_output(prompt, max_new_tokens=1000)
    took = time.time() - start_time
    print(f"--- Took {took:.2f} seconds ---")
