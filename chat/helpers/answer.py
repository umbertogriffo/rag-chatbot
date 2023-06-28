import re
from typing import Optional

from helpers.log import get_logger

logger = get_logger(__name__)

answer_pattern = r"Answer:\s*(.*)"


def extract_answer(answer: str) -> Optional[str]:
    # Extract the generated answer
    match = re.search(answer_pattern, answer, re.MULTILINE | re.DOTALL)
    if match:
        helpful_answer = match.group(1)
        return helpful_answer
    else:
        return None
