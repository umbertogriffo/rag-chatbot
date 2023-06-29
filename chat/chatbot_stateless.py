from pathlib import Path

from conversation.prompts import SUMMARIZATION_PROMPT
from helpers.extractor import extract_summary
from helpers.log import get_logger
from helpers.model import load_gpt4all
from langchain import LLMChain, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from memory.vector_memory import (
    VectorMemory,
    initialize_embedding,
    search_most_similar_doc,
)
from rich.console import Console

logger = get_logger(__name__)

template = """
You are an exceptional Senior Software Engineer support chatbot that gently answer questions.
You know the following context information:
{context}
Answer to the following question from a customer. Use only information from the previous context information.
Do not invent stuff.
---
Question: {question}
Answer:"""


def main():
    root_folder = Path(__file__).resolve().parent.parent
    model_path = root_folder / "models" / "ggml-model-q4_0.bin"
    vector_store_path = root_folder / "vector_store" / "docs_index"

    n_threads = 4

    console = Console(color_system="windows")

    llm = load_gpt4all(str(model_path), n_threads)
    embedding = initialize_embedding()
    memory = VectorMemory(embedding=embedding)
    # Load the vector store to use as the index
    index = memory.load_memory_index(str(vector_store_path))

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

        matched_doc = search_most_similar_doc(question, index)

        chain = load_summarize_chain(
            llm, chain_type="stuff", prompt=SUMMARIZATION_PROMPT
        )

        summary = extract_summary(chain.run([matched_doc[0]]))

        context = summary

        prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        ).partial(context=context)
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        console.print(f"\n[bold green]Question:[/bold green]{question}")
        console.print("\n[bold green]Answer:[/bold green]")
        console.print(llm_chain.run(question))


if __name__ == "__main__":
    main()
