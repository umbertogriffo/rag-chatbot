from core.config import settings
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from helpers.log import get_logger
from schemas.chat import ChatRequest

from api.deps import LamaCppClientDep

logger = get_logger(__name__)

router = APIRouter()


@router.post("/chat/")
async def chat(query: ChatRequest, llm_client: LamaCppClientDep):
    logger.info(query)

    try:
        # Your existing LLM logic here
        # response = MARKDOWN_RESPONSE
        answer = llm_client.generate_answer(query.text, max_new_tokens=settings.DEFAULT_MAX_NEW_TOKENS)
        return JSONResponse({"response": answer})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to generate response: {str(e)}"})
