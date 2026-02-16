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


def generate_deterministic_id(text: str) -> str:
    """
    Generate a deterministic ID from text content using SHA-256 hash.

    This ensures that identical content always produces the same ID,
    enabling deduplication when using upsert() operations.

    Args:
        text (str): The text content to hash

    Returns:
        str: SHA-256 hex digest as deterministic ID

    Example:
        >>> generate_deterministic_id("Hello World")
        'd2a84f4b8b650937ec8f73cd8be2c74add5a911ba64df27458ed8229da804a26'
    """

    # Normalize the text for consistent hashing
    content_to_hash = normalize_text(text)

    # Generate SHA-256 hash
    hash_object = hashlib.sha256(content_to_hash.encode("utf-8"))
    deterministic_id = hash_object.hexdigest()

    return deterministic_id


def generate_deterministic_ids(texts: list[str]) -> list[str]:
    """
    Generate deterministic IDs for a list of texts.

    Args:
        texts (list[str]): List of text contents

    Returns:
        list[str]: List of deterministic IDs
    """
    ids = []

    for text in texts:
        deterministic_id = generate_deterministic_id(text)
        ids.append(deterministic_id)

    return ids
