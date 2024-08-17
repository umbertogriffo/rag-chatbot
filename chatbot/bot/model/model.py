from abc import ABC
from typing import Any, Dict, Optional


class Model(ABC):
    url: str
    file_name: str
    clients: list[str]
    config: Dict[str, Any]
    config_answer: Optional[Dict[str, Any]]
    type: Optional[str]
    system_template: str
