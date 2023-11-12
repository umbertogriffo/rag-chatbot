import time
from pathlib import Path

from ctransformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import TextStreamer

from exp_ctransformers.model import get_model_setting, auto_download, config
from exp_ctransformers.prompts import generate_prompt

if __name__ == '__main__':
    root_folder = Path(__file__).resolve().parent.parent.parent
    model_folder = root_folder / "models"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    model_settings = get_model_setting("mistral")
    model_file = model_settings.name
    system_template = model_settings.system_template
    prompt_template = model_settings.prompt_template

    model_path = model_folder / model_settings.name

    auto_download(model_settings, model_path)
    llm = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id=str(model_folder),
                                               model_file=model_file,
                                               model_type="mistral",
                                               config=AutoConfig(config=config),
                                               hf=True)
    tokenizer = AutoTokenizer.from_pretrained(llm)

    # question_p = """What is the date for announcement"""
    # context_p = """ On August 10 said that its arm JSW Neo Energy has agreed to buy a portfolio of 1753 mega watt
    # renewable energy generation capacity from Mytrah Energy India Pvt Ltd for Rs 10,530 crore."""

    question_p = """Create a regex to extract dates from logs in Python
    """
    context_p = """ """

    prompt = generate_prompt(template=prompt_template, system=system_template, question=question_p, context=context_p)
    inputs = tokenizer(text=prompt, return_tensors="pt").input_ids

    streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True)

    start_time = time.time()
    _ = llm.generate(inputs, streamer=streamer, max_new_tokens=1000)
    took = time.time() - start_time
    print(f"--- Took {took:.2f} seconds ---")
