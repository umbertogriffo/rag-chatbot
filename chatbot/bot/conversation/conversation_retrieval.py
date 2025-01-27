from asyncio import get_event_loop
from typing import Any

from entities.document import Document
from helpers.log import get_logger

from bot.client.lama_cpp_client import LamaCppClient
from bot.conversation.chat_history import ChatHistory
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

    def __init__(self, llm: LamaCppClient, chat_history: ChatHistory) -> None:
        """
        Initializes a new instance of the ConversationRetrieval class.

        Args:
            llm (LlmClient): The language model client for conversation-related tasks.
            chat_history (ChatHistory): The chat history object to store conversation history.
        """
        self.llm = llm
        self.chat_history = chat_history

    def get_chat_history(self) -> str:
        """
        Retrieves the chat history as a single string.

        Returns:
            str: The chat history concatenated into a single string, with each message separated by a newline.
        """
        chat_history = "\n".join([msg for msg in self.chat_history])
        return chat_history

    def append_chat_history(self, question: str, answer: str) -> None:
        """
        Append a new question and answer to the chat history with a new question and answer.

        Args:
            question (str): The question to add to the chat history.
            answer (str): The answer to add to the chat history.

        Returns:
            list[str]: The updated chat history.
        """
        self.chat_history.append(f"question: {question}, answer: {answer}")

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
        chat_history = self.get_chat_history()
        if chat_history:
            logger.info("--- Refining the question based on the chat history... ---")

            conversation_awareness_prompt = self.llm.generate_refined_question_conversation_awareness_prompt(
                question, chat_history
            )

            logger.info(f"--- Prompt:\n {conversation_awareness_prompt} \n---")

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
        """
        chat_history = self.get_chat_history()

        if chat_history:
            logger.info("--- Answer the question based on the chat history... ---")

            conversation_awareness_prompt = self.llm.generate_refined_answer_conversation_awareness_prompt(
                question, chat_history
            )

            logger.debug(f"--- Prompt:\n {conversation_awareness_prompt} \n---")

            streamer = self.llm.start_answer_iterator_streamer(
                conversation_awareness_prompt, max_new_tokens=max_new_tokens
            )

            return streamer
        else:
            prompt = self.llm.generate_qa_prompt(question=question)
            logger.debug(f"--- Prompt:\n {prompt} \n---")
            streamer = self.llm.start_answer_iterator_streamer(prompt, max_new_tokens=max_new_tokens)
            return streamer

    def context_aware_answer(
        self,
        ctx_synthesis_strategy: BaseSynthesisStrategy,
        question: str,
        retrieved_contents: list[Document],
        max_new_tokens: int = 512,
    ):
        if not retrieved_contents:
            return self.answer(question, max_new_tokens=max_new_tokens), []

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
