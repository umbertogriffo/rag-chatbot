from pydantic import BaseModel


class ModelInfo(BaseModel):
    name: str
    available: bool = True


class ModelListResponse(BaseModel):
    models: list[ModelInfo]
    default_model: str


class SynthesisStrategyListResponse(BaseModel):
    strategies: list[str]
    default_strategy: str
