import argparse
import sys
from pathlib import Path

from conversation.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from conversation.question_answer import QuestionAndAnswer, QuestionAndAnswerConfig
from helpers.log import get_logger
from helpers.model import (
    SUPPORTED_MODELS,
    auto_download,
    get_model_setting,
    load_gpt4all,
)
from helpers.reader import read_input
from memory.vector_memory import VectorMemory, initialize_embedding
from pyfiglet import Figlet
from rich.console import Console
from rich.markdown import Markdown

logger = get_logger(__name__)


def run_chatbot_loop(qa: QuestionAndAnswer) -> None:
    custom_fig = Figlet(font="graffiti")
    console = Console(color_system="windows")
    console.print(custom_fig.renderText("ChatBot"))
    console.print(
        "[bold magenta]Hi! 👋, I'm your friendly chatbot 🦜 here to assist you. "
        "\nHow can I help you today? [/bold "
        "magenta]Type 'exit' to stop."
    )
    chat_history = []
    while True:
        console.print("[bold green]Please enter your question:[/bold green]")
        question = read_input()

        if question.lower() == "exit":
            break
        logger.info(f"question: {question}, chat_history: {chat_history}")

        # Generate the answer using the ConversationalRetrievalChain
        result = qa.generate_answer(question, chat_history)
        answer = result["answer"]

        # Update the history
        chat_history.append((question, answer))

        console.print("\n[bold magenta]Formatted Answer:[/bold magenta]")
        if answer:
            console.print(Markdown(answer))
        else:
            console.print("[bold red]Something went wrong![/bold red]")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Chatbot")

    model_list = list(SUPPORTED_MODELS.keys())
    default_model = list(SUPPORTED_MODELS.keys())[0]

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


def main(parameters):
    model_settings = get_model_setting(parameters.model)

    root_folder = Path(__file__).resolve().parent.parent
    model_folder = root_folder / "models"
    model_path = model_folder / model_settings.name

    vector_store_path = root_folder / "vector_store" / "docs_index"

    auto_download(model_settings, model_path)
    llm = load_gpt4all(
        str(model_path),
        answer_prefix_tokens=model_settings.answer_prefix_tokens,
        n_ctx=model_settings.n_ctx,
        n_predict=model_settings.n_predict,
        temperature=model_settings.temperature,
        repeat_penalty=model_settings.repeat_penalty,
        n_threads=parameters.n_threads,
    )
    embedding = initialize_embedding()

    memory = VectorMemory(embedding=embedding)
    index = memory.load_memory_index(str(vector_store_path))

    qa_config = QuestionAndAnswerConfig(
        llm, index, CONDENSE_QUESTION_PROMPT, QA_PROMPT, k=parameters.k
    )
    qa = QuestionAndAnswer(qa_config)

    run_chatbot_loop(qa)


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)
