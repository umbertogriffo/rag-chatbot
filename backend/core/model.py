from dataclasses import dataclass

from bot.client.prompt import SYSTEM_TEMPLATE


@dataclass
class ModelSettings:
    url: str
    name: str
    file_name: str
    reasoning_start_tag: str | None
    reasoning_stop_tag: str | None
    system_template: str = SYSTEM_TEMPLATE
    reasoning: bool = False
