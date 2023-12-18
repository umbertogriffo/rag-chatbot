import argparse
import sys
import time
from pathlib import Path

from bot.conversation.conversation import Conversation
from bot.memory.embedder import EmbedderHuggingFace
from bot.memory.vector_memory import VectorMemory
from bot.model import Model
from bot.model_settings import get_model_setting, get_models
from helpers.log import get_logger
from helpers.prettier import prettify_source
from helpers.reader import read_input
from pyfiglet import Figlet
from rich.console import Console
from rich.markdown import Markdown

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
        "--max-new-tokens",
        type=int,
        help="The maximum number of new tokens to generate.. Defaults to 512.",
        required=False,
        default=512,
    )

    return parser.parse_args()


def loop(conversation, index, parameters) -> None:
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

        logger.info(f"--- Question: {question}, Chat_history: {conversation.get_chat_history()} ---")

        start_time = time.time()
        question = conversation.refine_question(question)

        retrieved_contents, sources = index.similarity_search(query=question, k=parameters.k)

        console.print("\n[bold magenta]Sources:[/bold magenta]")
        for source in sources:
            console.print(Markdown(prettify_source(source)))

        console.print("\n[bold magenta]Answer:[/bold magenta]")

        answer = conversation.answer(question, retrieved_contents)

        conversation.update_chat_history(question, answer)

        console.print("\n[bold magenta]Formatted Answer:[/bold magenta]")
        if answer:
            console.print(Markdown(answer))
            took = time.time() - start_time
            print(f"\n--- Took {took:.2f} seconds ---")
        else:
            console.print("[bold red]Something went wrong![/bold red]")


def main(parameters):
    model_settings = get_model_setting(parameters.model)

    root_folder = Path(__file__).resolve().parent.parent.parent
    model_folder = root_folder / "models"
    vector_store_path = root_folder / "vector_store" / "docs_index"

    llm = Model(model_folder, model_settings)
    conversation = Conversation(llm)

    embedding = EmbedderHuggingFace().get_embedding()
    index = VectorMemory(vector_store_path=str(vector_store_path), embedding=embedding)

    loop(conversation, index, parameters)


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)
