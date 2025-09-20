from abc import ABC
from typing import Any

from bot.client.prompt import SYSTEM_TEMPLATE


class ModelSettings(ABC):
    url: str
    file_name: str
    system_template: str = SYSTEM_TEMPLATE
    config: dict[str, Any]
    config_answer: dict[str, Any] | None
    reasoning: bool = False
    reasoning_start_tag: str | None
    reasoning_stop_tag: str | None
