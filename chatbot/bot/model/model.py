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
    qa_prompt_template: str
    ctx_prompt_template: str
    refined_ctx_prompt_template: str
    refined_question_conversation_awareness_prompt_template: str
    refined_answer_conversation_awareness_prompt_template: str
