import re
from typing import Optional

from helpers.log import get_logger

logger = get_logger(__name__)

answer_pattern = r"Answer:\s*(.*)"
summary_pattern = r"CONCISE SUMMARY:\s*(.*)"


def extract_answer(answer: str) -> Optional[str]:
    """
    Extracts the helpful answer from the generated answer string.

    Parameters:
    -----------
    answer : str
        The generated answer string.

    Returns:
    -------
    Optional[str]
        The extracted helpful answer, or None if no match is found.

    """
    # Extract the generated answer
    match = re.search(answer_pattern, answer, re.MULTILINE | re.DOTALL)
    if match:
        helpful_answer = match.group(1)
        return helpful_answer
    else:
        return None


def extract_summary(summary: str) -> Optional[str]:
    """
    Extracts the concise summary from the generated summary string.

    Parameters:
    -----------
    summary : str
        The generated summary string.

    Returns:
    -------
    Optional[str]
        The extracted concise summary, or None if no match is found.

    """
    # Extract the generated summary
    match = re.search(summary_pattern, summary, re.MULTILINE | re.DOTALL)
    if match:
        concise_summary = match.group(1)
        return concise_summary
    else:
        return None
