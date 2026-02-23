from core.config import settings
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from helpers.log import get_logger
from schemas.chat import ChatRequest

logger = get_logger(__name__)

router = APIRouter()


def _get_llm_client():
    """Lazy import of llm_client to allow mocking in tests."""
    from llm_client import llm_client  # noqa: PLC0415

    return llm_client


@router.websocket("/chat/stream")
async def chat_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming chat responses token by token."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            query = ChatRequest(**data)

            try:
                client = _get_llm_client()
                stream = await client.async_start_answer_iterator_streamer(
                    query.text,
                    max_new_tokens=settings.DEFAULT_MAX_NEW_TOKENS,
                )
                for output in stream:
                    token = client.parse_token(output)
                    if token:
                        await websocket.send_json({"token": token, "done": False})
                await websocket.send_json({"token": "", "done": True})
            except Exception as exc:
                logger.exception("Error during streaming: %s", exc)
                await websocket.send_json({"error": str(exc), "done": True})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
