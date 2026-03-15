import time

from bot.conversation.conversation_handler import (
    answer,
    answer_with_context,
    extract_content_after_reasoning,
    refine_question,
)
from bot.conversation.ctx_strategy import get_ctx_synthesis_strategy
from core.config import settings
from fastapi import WebSocket
from helpers.log import get_logger
from helpers.prettier import prettify_source
from schemas.chat import ChatRequest

from api.deps import ChatHistoryDep, LamaCppClientDep, VectorDatabaseDep

logger = get_logger(__name__)


async def stream_chat_response(
    websocket: WebSocket, llm_client: LamaCppClientDep, query: ChatRequest, chat_history: ChatHistoryDep
):
    """
    Helper function to stream chat responses token by token.
     Args:
        websocket (WebSocket): The WebSocket connection to send responses through.
        llm_client (LamaCppClientDep): The LLM client dependency for generating responses.
        query (ChatRequest): The chat request containing the user's query.
        chat_history (ChatHistoryDep): The chat history dependency to maintain conversation context.
    """
    try:
        start_time = time.time()

        full_response = ""
        stream = await answer(
            llm=llm_client,
            question=query.text,
            chat_history=chat_history,
            max_new_tokens=settings.MAX_NEW_TOKENS,
        )
        for output in stream:
            token = llm_client.parse_token(output)
            if token:
                full_response += token
                await websocket.send_text(token)

        if llm_client.model_settings.reasoning:
            final_answer = extract_content_after_reasoning(full_response, llm_client.model_settings.reasoning_stop_tag)
            if final_answer == "":
                final_answer = "I didn't provide the answer; perhaps I can try again."
        else:
            final_answer = full_response

        chat_history.append(f"question: {query.text}, answer: {final_answer}")
        logger.debug(f"Updated chat history: {chat_history}")

        took = time.time() - start_time
        logger.info(f"\n--- Took {took:.2f} seconds ---")
    except Exception as exc:
        logger.exception("Error during streaming: %s", exc)
        await websocket.send_text("Error during streaming.")


async def stream_rag_response(
    websocket: WebSocket,
    llm_client: LamaCppClientDep,
    query: ChatRequest,
    chat_history: ChatHistoryDep,
    index: VectorDatabaseDep,
):
    """
    Helper function to stream RAG responses token by token.
     Args:
        websocket (WebSocket): The WebSocket connection to send responses through.
        llm_client (LamaCppClientDep): The LLM client dependency for generating responses.
        query (ChatRequest): The chat request containing the user's query.
        chat_history (ChatHistoryDep): The chat history dependency to maintain conversation context.
        index (VectorDatabaseDep): The vector database dependency for retrieval.
    """
    try:
        start_time = time.time()
        ctx_synthesis_strategy = get_ctx_synthesis_strategy(settings.SYNTHESIS_STRATEGY, llm=llm_client)

        retrieval_response = ""
        full_response = ""

        refined_user_input = await refine_question(
            llm_client, query.text, chat_history=chat_history, max_new_tokens=settings.MAX_NEW_TOKENS
        )
        retrieved_contents, sources = index.similarity_search_with_threshold(
            query=refined_user_input, k=settings.NUM_RETRIEVALS
        )
        if retrieved_contents:
            retrieval_response += "Here are the retrieved text chunks with a content preview: \n\n"

            for source in sources:
                retrieval_response += prettify_source(source)
                retrieval_response += "\n\n"
        else:
            retrieval_response += "I did not detect any pertinent chunk of text from the documents. \n\n"
            await websocket.send_text(retrieval_response)

        await websocket.send_text(retrieval_response)
        await websocket.send_text("-" * 20 + "\n\n")
        await websocket.send_text("**Answer:** \n\n")

        streamer, _ = await answer_with_context(
            llm_client,
            ctx_synthesis_strategy,
            query.text,
            chat_history,
            retrieved_contents,
            settings.MAX_NEW_TOKENS,
        )

        for output in streamer:
            token = llm_client.parse_token(output)
            if token:
                full_response += token
                await websocket.send_text(token)

        if llm_client.model_settings.reasoning:
            final_answer = extract_content_after_reasoning(full_response, llm_client.model_settings.reasoning_stop_tag)
            if final_answer == "":
                final_answer = "I wasn't able to provide the answer; Do you want me to try again?"
        else:
            final_answer = full_response

        chat_history.append(f"question: {query.text}, answer: {final_answer}")

        took = time.time() - start_time
        logger.info(f"\n--- Took {took:.2f} seconds ---")

    except Exception as exc:
        logger.exception("Error during RAG streaming: %s", exc)
        await websocket.send_text("Error during RAG streaming.")
