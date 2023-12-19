from typing import List, Tuple, Union

from transformers import TextIteratorStreamer

from bot.conversation.ctx_strategy import ContextSynthesisStrategy
from bot.model.client.client import Client
from helpers.log import get_logger

logger = get_logger(__name__)


class Conversation:
    """
    Question and Answer system using ContextSynthesisStrategy.
    """

    def __init__(self, llm: Client) -> None:
        self.llm = llm
        self.chat_history = []

    def get_chat_history(self) -> List[Tuple[str, str]]:
        """
        Gets the chat history.

        Returns:
            List[Tuple[str, str]]: The chat history, a list of tuples where each tuple
                consists of the question and answer.
        """
        return self.chat_history

    def update_chat_history(self, question: str, answer: str) -> List[Tuple[str, str]]:
        """
        Updates the chat history.

        Args:
            question: The question that was asked.
            answer: The answer that was given.

        Returns:
            List[Tuple[str, str]]: The updated chat history, a list of tuples where each tuple
                consists of the question and answer.
        """
        self.chat_history.append((question, answer))
        self.chat_history = self.keep_chat_history_size()

        return self.chat_history

    def keep_chat_history_size(self, max_size: int = 2) -> List[Tuple[str, str]]:
        """
        Keeps the list of chat history at the specified maximum size by popping out the oldest elements.

        Args:
            max_size: The maximum size of the list.

        Returns:
            The updated list of chat history.
        """

        if len(self.chat_history) > max_size:
            self.chat_history = self.chat_history[-max_size:]
        return self.chat_history

    def refine_question(self, question: str) -> str:
        if self.get_chat_history():
            questions_and_answers = ["\n".join([qa[0], qa[1]]) for qa in self.get_chat_history()]
            chat_history = "\n".join(questions_and_answers)
            logger.info(f"--- Refining the question based on the chat history... ---")
            conversation_awareness_prompt = self.llm.generate_conversation_awareness_prompt(question,
                                                                                            chat_history)
            refined_question = self.llm.generate_answer(conversation_awareness_prompt, max_new_tokens=128)
            logger.info(f"--- Refined Question: {refined_question} ---")

            return refined_question
        else:
            return question

    def answer(self, question: str, retrieved_contents, return_generator=False) -> Union[str | TextIteratorStreamer]:
        """
        Generates an answer using the `ContextSynthesisStrategy` for the given question.

        Parameters:
        -----------
        question : str
            The question to generate an answer for.

        Returns:
        -------
        str
            The generated answer for the question.

        """
        strategy = ContextSynthesisStrategy(self.llm)

        answer, fmt_prompts = strategy.generate_response_cr(
            retrieved_contents, question, return_generator=return_generator
        )

        return answer
