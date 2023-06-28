import re
from typing import Optional

from helpers.log import get_logger

logger = get_logger(__name__)

answer_pattern = r"Answer:\s*(.*)"
summary_pattern = r"Concise Summary:\s*(.*)"


def extract_answer(answer: str) -> Optional[str]:
    # Extract the generated answer
    match = re.search(answer_pattern, answer, re.MULTILINE | re.DOTALL)
    if match:
        helpful_answer = match.group(1)
        return helpful_answer
    else:
        return None


def extract_summary(summary: str) -> Optional[str]:
    # Extract the generated summary
    match = re.search(summary_pattern, summary, re.MULTILINE | re.DOTALL)
    if match:
        concise_summary = match.group(1)
        return concise_summary
    else:
        return None
