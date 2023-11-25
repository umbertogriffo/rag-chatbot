import argparse
import sys
from pathlib import Path

from helpers.log import get_logger
from helpers.reader import read_input
from memory.vector_memory import VectorMemory, initialize_embedding
from pyfiglet import Figlet
from rich.console import Console
from rich.markdown import Markdown

from model import Model, get_models, get_model_setting

logger = get_logger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Chatbot")

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

    parser.add_argument(
        "--k",
        type=int,
        help="Number of chunks to return from the similarity search. Defaults to 2.",
        required=False,
        default=2,
    )

    parser.add_argument(
        "--n-threads",
        type=int,
        help="Number of threads to use. Defaults to 4.",
        required=False,
        default=4,
    )

    return parser.parse_args()


def run_chatbot_loop(llm, index) -> None:
    custom_fig = Figlet(font="graffiti")
    console = Console(color_system="windows")
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

        docs = index.similarity_search(query=question, k=2)

        console.print("\n[bold magenta]Sources:[/bold magenta]")
        for doc in docs:
            # logger.info(doc.page_content)
            console.print(Markdown(f"- {doc.metadata.get('source', '')}"))

        console.print("\n[bold magenta]Answer:[/bold magenta]")
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = llm.generate_contextual_prompt(question=question, context=context)
        answer = llm.generate_answer(prompt, max_new_tokens=1000)

        console.print("\n[bold magenta]Formatted Answer:[/bold magenta]")
        if answer:
            console.print(Markdown(answer))
        else:
            console.print("[bold red]Something went wrong![/bold red]")


def main(parameters):
    model_settings = get_model_setting(parameters.model)

    root_folder = Path(__file__).resolve().parent.parent
    model_folder = root_folder / "models"
    vector_store_path = root_folder / "vector_store" / "docs_index"

    llm = Model(model_folder, model_settings)

    embedding = initialize_embedding()

    memory = VectorMemory(embedding=embedding)
    index = memory.load_memory_index(str(vector_store_path))

    run_chatbot_loop(llm, index)


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)
