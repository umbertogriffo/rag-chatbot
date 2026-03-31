from pathlib import Path

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
    embedding = Embedder(model_name=settings.EMBEDDING_MODEL)
    index = Chroma(is_persistent=True, persist_directory=str(vector_store_path), embedding=embedding)

    return index
