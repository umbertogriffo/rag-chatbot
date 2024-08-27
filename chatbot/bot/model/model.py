from abc import ABC
from typing import Any, Dict, Optional


class Model(ABC):
    url: str
    file_name: str
    config: Dict[str, Any]
    config_answer: Optional[Dict[str, Any]]
