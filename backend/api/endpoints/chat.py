from core.config import settings
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from helpers.log import get_logger
from schemas.chat import ChatRequest

from api.deps import LlamaCppClientDep
from api.exceptions import LLMTimeoutError, LLMConnectionError, LLMError

logger = get_logger(__name__)

router = APIRouter()


@router.post("/chat/")
async def chat(query: ChatRequest, llm_client: LlamaCppClientDep):
    logger.info(query)

    try:
        answer = await llm_client.async_generate_answer(query.text, max_new_tokens=settings.MAX_NEW_TOKENS)
        return JSONResponse({"response": answer})
    except LLMTimeoutError as e:
        logger.error(f"LLM timeout for query: {query.text[:50]}... - {e}")
        raise HTTPException(
            status_code=504,
            detail="The request took too long to process. Please try again with a shorter query."
        )
    except LLMConnectionError as e:
        logger.error(f"LLM connection failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to the LLM service. Please try again later."
        )
    except LLMError as e:
        logger.error(f"LLM error for query: {query.text[:50]}... - {e}")
        raise HTTPException(
            status_code=e.status_code,
            detail=f"Failed to generate response: {e.message}"
        )
    except Exception as e:
        # Catch-all for unexpected errors
        logger.exception(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later."
        )
