from fastapi import APIRouter, Depends

from backend.app.core.config import settings
from backend.app.core.security import get_current_user
from backend.app.schemas.models import ModelInfo, ModelListResponse, SynthesisStrategyListResponse

router = APIRouter()


@router.get("/", response_model=ModelListResponse)
async def list_models(_current_user: str = Depends(get_current_user)):
    from chatbot.bot.model.model_registry import get_models

    model_names = get_models()
    models = [ModelInfo(name=name) for name in model_names]
    return ModelListResponse(models=models, default_model=settings.DEFAULT_MODEL)


@router.get("/strategies", response_model=SynthesisStrategyListResponse)
async def list_synthesis_strategies(_current_user: str = Depends(get_current_user)):
    from chatbot.bot.conversation.ctx_strategy import get_ctx_synthesis_strategies

    strategies = get_ctx_synthesis_strategies()
    return SynthesisStrategyListResponse(
        strategies=strategies,
        default_strategy=settings.DEFAULT_SYNTHESIS_STRATEGY,
    )
