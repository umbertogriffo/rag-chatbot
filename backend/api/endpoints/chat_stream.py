import time

from bot.conversation.conversation_handler import answer
from core.config import settings
from fastapi import APIRouter, Response, WebSocket, WebSocketDisconnect
from helpers.log import get_logger
from schemas.chat import ChatRequest

from api.deps import ChatHistoryDep, LamaCppClientDep, VectorDatabaseDep

logger = get_logger(__name__)

router = APIRouter()


@router.delete(
    path="/chat/history",
    status_code=204,
)
async def clear_chat_history(chat_history: ChatHistoryDep):
    """Clear the server-side chat history."""
    chat_history.clear()
    return Response(status_code=204)


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
            max_new_tokens=settings.DEFAULT_MAX_NEW_TOKENS,
        )
        for output in stream:
            token = llm_client.parse_token(output)
            if token:
                full_response += token
                await websocket.send_text(token)
        chat_history.append(f"question: {query.text}, answer: {full_response}")
        logger.debug(f"Updated chat history: {chat_history}")

        took = time.time() - start_time
        logger.info(f"\n--- Took {took:.2f} seconds ---")
    except Exception as exc:
        logger.exception("Error during streaming: %s", exc)
        await websocket.send_text("Error during streaming.")


@router.websocket(
    path="/chat/stream",
)
async def chat_stream(
    websocket: WebSocket, llm_client: LamaCppClientDep, chat_history: ChatHistoryDep, index: VectorDatabaseDep
):
    """WebSocket endpoint for streaming chat responses token by token."""
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    try:
        while True:
            data = await websocket.receive_json()
            logger.info(f"Received data: {data}")
            query = ChatRequest(**data)
            if query.rag:
                logger.info("RAG enabled, but RAG functionality is not implemented yet.")
                await websocket.send_text("RAG functionality is not implemented yet.")
            else:
                await stream_chat_response(websocket, llm_client, query, chat_history)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception(f"Unexpected error in WebSocket handler: {e}")
        raise
