from abc import ABC
from typing import Any


class ModelSettings(ABC):
    url: str
    file_name: str
    config: dict[str, Any]
    config_answer: dict[str, Any] | None
