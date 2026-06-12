from llm_providers.prompt import SYSTEM_TEMPLATE
from pydantic import BaseModel


class ModelSettings(BaseModel):
    url: str
    name: str
    file_name: str
    reasoning_start_tag: str | None
    reasoning_stop_tag: str | None
    system_template: str = SYSTEM_TEMPLATE
    reasoning: bool = False
