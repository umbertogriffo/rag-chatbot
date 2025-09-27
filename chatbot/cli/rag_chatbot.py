import argparse
import sys
import time
from pathlib import Path

from bot.client.lama_cpp_client import LamaCppClient
from bot.conversation.chat_history import ChatHistory
from bot.conversation.conversation_handler import answer_with_context, refine_question
from bot.conversation.ctx_strategy import get_ctx_synthesis_strategies, get_ctx_synthesis_strategy
from bot.memory.embedder import Embedder
from bot.memory.vector_database.chroma import Chroma
from bot.model.model_registry import Model, get_model_settings, get_models
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
    default_model = Model.LLAMA_3_1.value

    synthesis_strategy_list = get_ctx_synthesis_strategies()
    default_synthesis_strategy = synthesis_strategy_list[0]

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
        "--synthesis-strategy",
        type=str,
        choices=synthesis_strategy_list,
        help=f"Model to be used. Defaults to {default_synthesis_strategy}.",
        required=False,
        const=default_synthesis_strategy,
        nargs="?",
        default=default_synthesis_strategy,
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


def loop(llm, chat_history, synthesis_strategy, index, parameters) -> None:
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

        logger.info(f"--- Question: {question}, Chat_history: {chat_history} ---")

        start_time = time.time()
        refined_question = refine_question(llm, question, chat_history)

        retrieved_contents, sources = index.similarity_search_with_threshold(query=refined_question, k=parameters.k)

        console.print("\n[bold magenta]Sources:[/bold magenta]")
        for source in sources:
            console.print(Markdown(prettify_source(source)))

        console.print("\n[bold magenta]Answer:[/bold magenta]")

        streamer, _ = answer_with_context(
            llm=llm,
            ctx_synthesis_strategy=synthesis_strategy,
            question=refined_question,
            chat_history=chat_history,
            retrieved_contents=retrieved_contents,
            max_new_tokens=parameters.max_new_tokens,
        )
        answer = ""
        for token in streamer:
            parsed_token = llm.parse_token(token)
            answer += parsed_token
            print(parsed_token, end="", flush=True)

        chat_history.append(
            f"question: {refined_question}, answer: {answer}",
        )

        console.print("\n[bold magenta]Formatted Answer:[/bold magenta]")
        if answer:
            console.print(Markdown(answer))
            took = time.time() - start_time
            print(f"\n--- Took {took:.2f} seconds ---")
        else:
            console.print("[bold red]Something went wrong![/bold red]")


def main(parameters):
    model_settings = get_model_settings(parameters.model)

    root_folder = Path(__file__).resolve().parent.parent.parent
    model_folder = root_folder / "models"
    vector_store_path = root_folder / "vector_store" / "docs_index"

    llm = LamaCppClient(model_folder=model_folder, model_settings=model_settings)

    synthesis_strategy = get_ctx_synthesis_strategy(parameters.synthesis_strategy, llm=llm)
    chat_history = ChatHistory(total_length=2)

    embedding = Embedder()
    index = Chroma(persist_directory=str(vector_store_path), embedding=embedding)

    loop(llm, chat_history, synthesis_strategy, index, parameters)


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)
