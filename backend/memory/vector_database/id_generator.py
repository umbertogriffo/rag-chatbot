import hashlib
import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalize text for deterministic ID generation.

    Steps:
    1. Convert to lowercase
    2. Normalize unicode characters (NFD form)
    3. Collapse multiple whitespaces to single space
    4. Strip leading/trailing whitespace

    Args:
        text (str): Input text to normalize

    Returns:
        str: Normalized text
    """
    # Convert to lowercase
    normalized = text.lower()

    # Normalize unicode to NFD (canonical decomposition)
    normalized = unicodedata.normalize("NFD", normalized)

    # Collapse multiple whitespaces (including newlines, tabs) to single space
    normalized = re.sub(r"\s+", " ", normalized)

    # Strip leading/trailing whitespace
    normalized = normalized.strip()

    return normalized


def generate_id(text: str) -> str:
    """
    Generate a deterministic SHA-256 fingerprint for a text.

    Args:
        text (str): The text content to hash

    Returns:
        str: SHA-256 hex digest representing the document version.

    Example:
        >>> generate_id("Hello World")
        'd2a84f4b8b650937ec8f73cd8be2c74add5a911ba64df27458ed8229da804a26'
    """
    # Normalize the text for consistent hashing
    normalized = normalize_text(text)
    # Generate SHA-256 hash
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
