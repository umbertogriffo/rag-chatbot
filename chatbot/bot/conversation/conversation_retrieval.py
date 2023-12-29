from asyncio import get_event_loop
from typing import Any, List, Tuple

from helpers.log import get_logger
from langchain_core.documents import Document

from bot.client.llm_client import LlmClient
from bot.conversation.ctx_strategy import AsyncTreeSummarizationStrategy, BaseSynthesisStrategy

logger = get_logger(__name__)


class ConversationRetrieval:
    """
    A class for managing conversation retrieval using a language model.

    Attributes:
        llm (LlmClient): The language model client for conversation-related tasks.
        chat_history (List[Tuple[str, str]]): A list to store the conversation
            history as tuples of questions and answers.
    """

    def __init__(self, llm: LlmClient) -> None:
        """
        Initializes a new instance of the ConversationRetrieval class.

        Args:
            llm (LlmClient): The language model client for conversation-related tasks.
        """
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

    def refine_question(self, question: str, max_new_tokens: int = 128) -> str:
        """
        Refines the given question based on the chat history.

        Args:
            question (str): The original question.
            max_new_tokens (int, optional): The maximum number of tokens to generate in the answer.
                Defaults to 128.

        Returns:
            str: The refined question.
        """
        if self.get_chat_history():
            questions_and_answers = [
                "\n".join([f"question: {qa[0]}", f"answer: {qa[1]}"]) for qa in self.get_chat_history()
            ]
            chat_history = "\n".join(questions_and_answers)

            logger.info("--- Refining the question based on the chat history... ---")

            conversation_awareness_prompt = self.llm.generate_refined_question_conversation_awareness_prompt(
                question, chat_history
            )
            refined_question = self.llm.generate_answer(conversation_awareness_prompt, max_new_tokens=max_new_tokens)

            logger.info(f"--- Refined Question: {refined_question} ---")

            return refined_question
        else:
            return question

    def answer(self, question: str, max_new_tokens: int = 512) -> Any:
        """
        Generates an answer to the given question based on the chat history or a direct prompt.

        Args:
            question (str): The input question for which an answer is generated.
            max_new_tokens (int, optional): The maximum number of tokens to generate in the answer.
                Defaults to 512.

        Returns:
            A streaming iterator (Any) for progressively generating the answer.

        Notes:
            The method checks if there is existing chat history. If chat history is available,
            it constructs a conversation-awareness prompt using the question and chat history.
            The answer is then generated using the LLM with the conversation-awareness prompt.
            If no chat history is available, a prompt is generated directly from the input question,
            and the answer is generated accordingly.

        Example:
            >>> conversation_retrieval = ConversationRetrieval(llm)
            >>> answer_streamer = conversation_retrieval.answer("What is the meaning of life?")
            >>> for token in answer_streamer:
            ...     print(token)
        """
        if self.get_chat_history():
            questions_and_answers = [
                "\n".join([f"question: {qa[0]}", f"answer: {qa[1]}"]) for qa in self.get_chat_history()
            ]
            chat_history = "\n".join(questions_and_answers)

            logger.info("--- Answer the question based on the chat history... ---")

            conversation_awareness_prompt = self.llm.generate_refined_answer_conversation_awareness_prompt(
                question, chat_history
            )
            streamer = self.llm.start_answer_iterator_streamer(
                conversation_awareness_prompt, max_new_tokens=max_new_tokens
            )

            return streamer
        else:
            prompt = self.llm.generate_qa_prompt(question=question)
            streamer = self.llm.start_answer_iterator_streamer(prompt, max_new_tokens=max_new_tokens)
            return streamer

    @staticmethod
    def context_aware_answer(
        ctx_synthesis_strategy: BaseSynthesisStrategy,
        question: str,
        retrieved_contents: List[Document],
        max_new_tokens: int = 512,
    ):
        if isinstance(ctx_synthesis_strategy, AsyncTreeSummarizationStrategy):
            loop = get_event_loop()
            streamer, fmt_prompts = loop.run_until_complete(
                ctx_synthesis_strategy.generate_response(retrieved_contents, question, max_new_tokens=max_new_tokens)
            )
        else:
            streamer, fmt_prompts = ctx_synthesis_strategy.generate_response(
                retrieved_contents, question, max_new_tokens=max_new_tokens
            )
        return streamer, fmt_prompts
