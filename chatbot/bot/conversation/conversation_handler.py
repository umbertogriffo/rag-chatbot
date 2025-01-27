from asyncio import get_event_loop
from typing import Any

from entities.document import Document
from helpers.log import get_logger

from bot.client.lama_cpp_client import LamaCppClient
from bot.conversation.chat_history import ChatHistory
from bot.conversation.ctx_strategy import AsyncTreeSummarizationStrategy, BaseSynthesisStrategy

logger = get_logger(__name__)


def refine_question(llm: LamaCppClient, question: str, chat_history: ChatHistory, max_new_tokens: int = 128) -> str:
    """
    Refines the given question based on the chat history.

    Args:
        llm (LlmClient): The language model client for conversation-related tasks.
        question (str): The original question.
        chat_history (List[Tuple[str, str]]): A list to store the conversation
        history as tuples of questions and answers.
        max_new_tokens (int, optional): The maximum number of tokens to generate in the answer.
            Defaults to 128.

    Returns:
        str: The refined question.
    """

    if chat_history:
        logger.info("--- Refining the question based on the chat history... ---")

        conversation_awareness_prompt = llm.generate_refined_question_conversation_awareness_prompt(
            question, str(chat_history)
        )

        logger.info(f"--- Prompt:\n {conversation_awareness_prompt} \n---")

        refined_question = llm.generate_answer(conversation_awareness_prompt, max_new_tokens=max_new_tokens)

        logger.info(f"--- Refined Question: {refined_question} ---")

        return refined_question
    else:
        return question


def answer(llm: LamaCppClient, question: str, chat_history: ChatHistory, max_new_tokens: int = 512) -> Any:
    """
    Generates an answer to the given question based on the chat history or a direct prompt.

    Args:
        llm (LlmClient): The language model client for conversation-related tasks.
        question (str): The input question for which an answer is generated.
        chat_history (List[Tuple[str, str]]): A list to store the conversation
        history as tuples of questions and answers.
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

    if chat_history:
        logger.info("--- Answer the question based on the chat history... ---")

        conversation_awareness_prompt = llm.generate_refined_answer_conversation_awareness_prompt(
            question, str(chat_history)
        )

        logger.debug(f"--- Prompt:\n {conversation_awareness_prompt} \n---")

        streamer = llm.start_answer_iterator_streamer(conversation_awareness_prompt, max_new_tokens=max_new_tokens)

        return streamer
    else:
        prompt = llm.generate_qa_prompt(question=question)
        logger.debug(f"--- Prompt:\n {prompt} \n---")
        streamer = llm.start_answer_iterator_streamer(prompt, max_new_tokens=max_new_tokens)
        return streamer


def answer_with_context(
    llm: LamaCppClient,
    ctx_synthesis_strategy: BaseSynthesisStrategy,
    question: str,
    chat_history: ChatHistory,
    retrieved_contents: list[Document],
    max_new_tokens: int = 512,
):
    """
    Generates an answer to the given question using a context synthesis strategy and retrieved contents.
    If the content is not provided generates an answer based on the chat history or a direct prompt.

    Args:
        llm (LlmClient): The language model client for conversation-related tasks.
        ctx_synthesis_strategy (BaseSynthesisStrategy): The strategy to use for context synthesis.
        question (str): The input question for which an answer is generated.
        chat_history (List[Tuple[str, str]]): A list to store the conversation
        history as tuples of questions and answers.
        retrieved_contents (list[Document]): A list of documents retrieved for context.
        max_new_tokens (int, optional): The maximum number of tokens to generate in the answer. Defaults to 512.

    Returns:
        tuple: A tuple containing the answer streamer and formatted prompts.
    """
    if not retrieved_contents:
        return answer(llm, question, chat_history, max_new_tokens=max_new_tokens), []

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
