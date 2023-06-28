from dataclasses import dataclass
from typing import List

from langchain import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import GPT4All
from langchain.vectorstores import Chroma


@dataclass
class QuestionAndAnswerConfig:
    llm: GPT4All
    index: Chroma
    condense_question_prompt: str
    qa_prompt: str
    k: int = 2
    max_tokens_limit: int = 1024
    verbose: bool = False


class QuestionAndAnswer:
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
            verbose=config.verbose,
        )

    def generate_answer(self, question: str, chat_history: List):
        return self.qa({"question": question, "chat_history": chat_history})
