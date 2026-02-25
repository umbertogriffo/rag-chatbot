from core.config import settings
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from helpers.log import get_logger
from schemas.chat import ChatRequest

from api.deps import LamaCppClientDep

logger = get_logger(__name__)

router = APIRouter()


@router.websocket(
    path="/chat/stream",
)
async def chat_stream(websocket: WebSocket, llm_client: LamaCppClientDep):
    """WebSocket endpoint for streaming chat responses token by token."""
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    try:
        while True:
            data = await websocket.receive_json()
            logger.info(f"Received data: {data}")
            query = ChatRequest(**data)
            try:
                stream = await llm_client.async_start_answer_iterator_streamer(
                    query.text,
                    max_new_tokens=settings.DEFAULT_MAX_NEW_TOKENS,
                )
                for output in stream:
                    token = llm_client.parse_token(output)
                    if token:
                        await websocket.send_json({"token": token, "done": False})
                await websocket.send_json({"token": "", "done": True})
            except Exception as exc:
                logger.exception("Error during streaming: %s", exc)
                await websocket.send_json({"error": str(exc), "done": True})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception(f"Unexpected error in WebSocket handler: {e}")
        raise
