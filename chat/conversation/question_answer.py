import os
from dataclasses import dataclass
from typing import List

from langchain import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import GPT4All
from langchain.vectorstores import Chroma


@dataclass
class QuestionAndAnswerConfig:
    """
    Configuration settings for QuestionAndAnswer class.

    Attributes:
    -----------
    llm : GPT4All
        The GPT4All model for language generation.

    index : Chroma
        The Chroma index for document retrieval.

    condense_question_prompt : str
        The prompt used for generating condensed questions.

    qa_prompt : str
        The prompt used for question answering.

    k : int, optional
        The number of retrievals to consider (default is 2).

    max_tokens_limit : int, optional
        The maximum number of tokens allowed in the answer (default is 1024).

    verbose : bool, optional
        Whether to enable verbose mode (default is False).
    """

    llm: GPT4All
    index: Chroma
    condense_question_prompt: str
    qa_prompt: str
    k: int = 2
    max_tokens_limit: int = 1024
    verbose: bool = False


def generate_source_links(results: dict) -> List[str]:
    """
    Generate source links based on a dictionary of results.

    Args:
        results (dict): A dictionary containing source document information.

    Returns:
        list: A list of unique source links formatted as "<file_name | file_path>".

    Example:
        results = {
            "source_documents": [
                {
                    "metadata": {
                        "source": "/path/to/source/file.txt"
                    }
                },
                # Add more source documents as needed
            ]
        }
        links = generate_source_links(results)
        # Sample output: ["<file.txt | /path/to/source/file.txt>"]
    """
    links = []
    source_documents = results["source_documents"]
    for source in source_documents:
        file_path = source.metadata["source"]
        file_name = os.path.basename(file_path).replace(" ", "-").lower()
        links.append(f"<{file_name} | {file_path}>")
    links = list(set(links))
    return links


class QuestionAndAnswer:
    """
    Question and Answer system using ConversationalRetrievalChain.

    Initializes a question and answer system with the provided configuration.

    Parameters:
    -----------
    config : QuestionAndAnswerConfig
        The configuration settings for the question and answer system.

    """

    def __init__(self, config: QuestionAndAnswerConfig) -> None:
        question_generator = LLMChain(
            llm=config.llm, prompt=config.condense_question_prompt
        )
        doc_chain = load_qa_chain(
            config.llm, chain_type="stuff", prompt=config.qa_prompt
        )
        retriever = config.index.as_retriever(
            search_type="similarity", search_kwargs={"k": config.k}
        )
        self.qa = ConversationalRetrievalChain(
            retriever=retriever,
            combine_docs_chain=doc_chain,
            question_generator=question_generator,
            max_tokens_limit=config.max_tokens_limit,
            return_source_documents=True,
            verbose=config.verbose,
        )

    def generate_answer(self, question: str, chat_history: List):
        """
        Generates an answer for the given question based on the chat history.

        Parameters:
        -----------
        question : str
            The question to generate an answer for.

        chat_history : List
            The list of chat history containing previous questions and answers.

        Returns:
        -------
        str
            The generated answer for the question.

        """
        return self.qa({"question": question, "chat_history": chat_history})
