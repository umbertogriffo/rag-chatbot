from pathlib import Path

from bot.memory.document_registry import DocumentRegistry
from bot.memory.embedder import Embedder
from bot.memory.vector_database.chroma import Chroma
from core.config import settings


def init_index(vector_store_path: Path) -> Chroma:
    """
    Loads a Vector Database index based on the specified vector store path.

    Args:
        vector_store_path (Path): The path to the vector store.

    Returns:
        Chroma: An instance of the Vector Database.
    """
    embedding = Embedder()
    index = Chroma(is_persistent=True, persist_directory=str(vector_store_path), embedding=embedding)

    return index


def init_registry(db_path: Path) -> DocumentRegistry:
    """
    Create or open the SQLite-backed document registry.

    Args:
        db_path (Path): Path to the SQLite database file.

    Returns:
        DocumentRegistry: A registry instance.
    """
    return DocumentRegistry(db_path)


index = init_index(settings.VECTOR_STORE_PATH)
registry = init_registry(settings.DATABASE_PATH)
