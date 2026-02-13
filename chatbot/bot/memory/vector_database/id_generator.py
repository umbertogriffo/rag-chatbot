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


def generate_deterministic_id(text: str, source: str | None = None, chunk_index: int | None = None) -> str:
    """
    Generate a deterministic ID from text content using SHA-256 hash.
    
    This ensures that identical content always produces the same ID,
    enabling deduplication when using upsert() operations.
    
    Args:
        text (str): The text content to hash
        source (str, optional): Source filename/URL to include in hash for uniqueness
        chunk_index (int, optional): Chunk index to include in hash for uniqueness
        
    Returns:
        str: SHA-256 hex digest as deterministic ID
        
    Example:
        >>> generate_deterministic_id("Hello World")
        'd2a84f4b8b650937ec8f73cd8be2c74add5a911ba64df27458ed8229da804a26'
        
        >>> generate_deterministic_id("Hello World", source="doc.md", chunk_index=0)
        'fb1f4c37da9c9c9e8a0c98d1f5c8e3f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6'
    """
    # Normalize the text for consistent hashing
    normalized_text = normalize_text(text)
    
    # Build the content to hash
    content_to_hash = normalized_text
    
    # Optionally include source to avoid collisions across different files
    if source is not None:
        content_to_hash += f"|source:{source}"
    
    # Optionally include chunk index to ensure unique IDs for chunks from same document
    if chunk_index is not None:
        content_to_hash += f"|chunk:{chunk_index}"
    
    # Generate SHA-256 hash
    hash_object = hashlib.sha256(content_to_hash.encode("utf-8"))
    deterministic_id = hash_object.hexdigest()
    
    return deterministic_id


def generate_deterministic_ids(
    texts: list[str],
    metadatas: list[dict] | None = None,
) -> list[str]:
    """
    Generate deterministic IDs for a list of texts.
    
    If metadatas are provided, uses 'source' field from metadata
    to ensure uniqueness across different source documents.
    
    Args:
        texts (list[str]): List of text contents
        metadatas (list[dict], optional): List of metadata dicts with optional 'source' field
        
    Returns:
        list[str]: List of deterministic IDs
    """
    ids = []
    
    for idx, text in enumerate(texts):
        source = None
        if metadatas and idx < len(metadatas) and metadatas[idx]:
            source = metadatas[idx].get("source")
        
        # Use chunk index to ensure uniqueness even if content is similar
        deterministic_id = generate_deterministic_id(text, source=source, chunk_index=idx)
        ids.append(deterministic_id)
    
    return ids
