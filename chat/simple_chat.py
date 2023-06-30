import argparse
import sys
from pathlib import Path

from helpers.log import get_logger
from helpers.model import load_gpt4all, Model
from langchain import LLMChain, PromptTemplate

from rich.console import Console
from rich.markdown import Markdown

logger = get_logger(__name__)

template = """
You are an exceptional Senior Software Engineer that gently answer technical questions.
---
Question: {question}
Answer:"""


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Software Engineer Chatbot")

    parser.add_argument(
        "--model",
        type=Model,
        choices=list(Model),
        help=f"Model to be used. Defaults to {Model.wizard}.",
        required=False,
        const=Model.wizard,
        nargs='?',
        default=Model.wizard,
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
    root_folder = Path(__file__).resolve().parent.parent
    model_path = root_folder / "models" / parameters.model.value

    console = Console(color_system="windows")

    llm = load_gpt4all(str(model_path),
                       n_threads=parameters.n_threads,
                       streaming=True,
                       verbose=True)

    # Chatbot loop
    console.print(
        "[bold magenta]Hi! 👋, I'm your friendly chatbot 🦜 here to assist you. How can I help you today? [/bold "
        "magenta]Type 'exit' to stop."
    )
    while True:
        console.print("[bold green]Please enter your question:[/bold green]")
        question = input("")

        if question.lower() == "exit":
            break

        prompt = PromptTemplate(
            template=template, input_variables=["question"]
        )
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        console.print(f"\n[bold green]Question:[/bold green] {question}")
        console.print("\n[bold green]Answer:[/bold green]")

        answer = llm_chain.run(question)
        console.print("\n[bold magenta]Formatted Answer:[/bold magenta]")
        console.print(Markdown(answer))


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as error:
        logger.error(f"An error occurred: {str(error)}", exc_info=True, stack_info=True)
        sys.exit(1)
