from config import settings
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from helpers.log import get_logger
from schemas.chat import ChatRequest

from api.deps import LlamaCppClientDep

logger = get_logger(__name__)

router = APIRouter()


@router.post("/chat/")
async def chat(query: ChatRequest, llm_client: LlamaCppClientDep):
    logger.info(query)

    try:
        answer = await llm_client.async_generate_answer(query.text, max_new_tokens=settings.MAX_NEW_TOKENS)
        return JSONResponse({"response": answer})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to generate response: {str(e)}"})
