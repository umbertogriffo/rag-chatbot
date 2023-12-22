from abc import ABC
from typing import Any


class Model(ABC):
    url: str
    file_name: str
    clients: list[str]
    type: str
    system_template: str
    qa_prompt_template: str
    ctx_prompt_template: str
    refined_ctx_prompt_template: str
    refined_question_conversation_awareness_prompt_template: str
    refined_answer_conversation_awareness_prompt_template: str
    config: Any
