import time
from pathlib import Path

from gpt4all import GPT4All
from model import auto_download, get_generate_config, get_model_setting

root_folder = Path(__file__).resolve().parent.parent.parent
model_folder = root_folder / "models"
Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

model_settings = get_model_setting("mistral")
config = get_generate_config()
model_path = model_folder / model_settings.name

auto_download(model_settings, model_path)

model = GPT4All(
    model_name=model_settings.name,
    model_path=model_folder,
    device=model_settings.device,
    allow_download=False,
    verbose=False,
)

system_template = model_settings.system_template
prompt_template = model_settings.prompt_template

with model.chat_session(system_template, prompt_template):
    start_time = time.time()

    question = "Create a regex to extract dates from logs in Python"
    output = model.generate(prompt=question, **config)
    print(output)

    took = time.time() - start_time
    print(f"--- Took {took:.2f} seconds ---")
