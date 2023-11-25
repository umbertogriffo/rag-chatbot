import argparse
import sys
from pathlib import Path

from helpers.log import get_logger
from helpers.reader import read_input
from pyfiglet import Figlet
from rich.console import Console
from rich.markdown import Markdown

from chat_new.model import Model, get_model_setting, get_models

logger = get_logger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chatbot")

    model_list = get_models()
    default_model = model_list[0]

    parser.add_argument(
        "--model",
        type=str,
        choices=model_list,
        help=f"Model to be used. Defaults to {default_model}.",
        required=False,
        const=default_model,
        nargs="?",
        default=default_model,
    )

    return parser.parse_args()


def run_chatbot_loop(llm):
    console = Console(color_system="windows")
    # Chatbot loop
    custom_fig = Figlet(font="graffiti")
    console.print(custom_fig.renderText("ChatBot"))
    console.print(
        "[bold magenta]Hi! 👋, I'm your friendly chatbot 🦜 here to assist you. "
        "\nHow can I help you today? [/bold "
        "magenta]Type 'exit' to stop."
    )
    while True:
        console.print("[bold green]Please enter your question:[/bold green]")
        question = read_input()

        if question.lower() == "exit":
            break

        prompt = llm.generate_prompt(question=question)

        console.print(f"\n[bold green]Question:[/bold green] {question}")
        console.print("\n[bold green]Answer:[/bold green]")

        answer = llm.generate_answer(prompt, max_new_tokens=1000)
        console.print("\n[bold magenta]Formatted Answer:[/bold magenta]")
        console.print(Markdown(answer))


def main(parameters):
    model_settings = get_model_setting(parameters.model)

    root_folder = Path(__file__).resolve().parent.parent
    model_folder = root_folder / "models"
    Path(model_folder).parent.mkdir(parents=True, exist_ok=True)

    llm = Model(model_folder, model_settings)
    run_chatbot_loop(llm)


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)
