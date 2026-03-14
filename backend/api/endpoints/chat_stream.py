from fastapi import APIRouter, Response, WebSocket, WebSocketDisconnect
from helpers.log import get_logger
from schemas.chat import ChatRequest

from api.deps import ChatHistoryDep, LamaCppClientDep, VectorDatabaseDep
from api.services.chat_stream import stream_chat_response, stream_rag_response

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
                await stream_rag_response(websocket, llm_client, query, chat_history, index)
            else:
                await stream_chat_response(websocket, llm_client, query, chat_history)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception(f"Unexpected error in WebSocket handler: {e}")
        raise
