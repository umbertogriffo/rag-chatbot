from typing import Any, List

from cleantext import clean
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from helpers.log import get_logger

logger = get_logger(__name__)


class VectorMemory:
    """
    Class for managing vector memory operations.

    Parameters:
    -----------
    embedding : Any
        The embedding object used for vectorization.

    verbose : bool, optional
        Whether to enable verbose mode (default is False).

    """
    def __init__(self, embedding: Any, verbose=False) -> None:
        self.embedding = embedding
        self.verbose = verbose

        if self.embedding is None:
            logger.error("No embedder passed to VectorMemory")
            raise Exception("No embedder passed to VectorMemory")

    def create_memory_index(self, chunks: List, vector_store_path: str) -> Chroma:
        """
        Creates a Chroma memory index from the given chunks.

        Parameters:
        -----------
        chunks : List
            The list of document chunks.

        vector_store_path : str
            The path to store the vector store.

        Returns:
        -------
        Chroma
            The created Chroma memory index.

        """
        texts = [clean(doc.page_content, no_emoji=True) for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]
        memory_index = Chroma.from_texts(
            texts,
            self.embedding,
            metadatas=metadatas,
            persist_directory=vector_store_path,
        )
        memory_index.persist()
        return memory_index

    def load_memory_index(self, vector_store_path: str) -> Chroma:
        """
        Loads the Chroma memory index from the given vector store path.

        Parameters:
        -----------
        vector_store_path : str
            The path to the vector store.

        Returns:
        -------
        Chroma
            The loaded Chroma memory index.

        """
        # Load the vector store to use as the index
        index = Chroma(
            persist_directory=str(vector_store_path), embedding_function=self.embedding
        )
        return index


def initialize_embedding() -> HuggingFaceEmbeddings:
    """
    Initializes the HuggingFaceEmbeddings object with the specified model.

    Returns:
    -------
    HuggingFaceEmbeddings
        The initialized HuggingFaceEmbeddings object.

    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def similarity_search(query: str, index: Chroma, k: int = 4):
    """
    Performs similarity search on the given query using the specified index.

    Parameters:
    -----------
    query : str
        The query string.

    index : Chroma
        The Chroma index to perform the search on.

    k : int, optional
        The number of retrievals to consider (default is 4).

    Returns:
    -------
    Tuple[List[Document]], List[Dict[str, Any]]
        A tuple containing the list of matched documents and a list of their sources.

    """
    matched_docs = index.similarity_search(query, k=k)
    sources = []
    for doc in matched_docs:
        sources.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    return matched_docs, sources


def search_most_similar_doc(query: str, index: Chroma, k: int = 4):
    """
    Searches for the most similar document to the given query using the specified index.

    Parameters:
    -----------
    query : str
        The query string.

    index : Chroma
        The Chroma index to perform the search on.

    k : int, optional
        The number of retrievals to consider (default is 4).

    Returns:
    -------
    Tuple[Document, float]
        A tuple containing the most similar document and its similarity score.

    """
    matched_docs = index.similarity_search_with_score(query, k=k)
    matched_doc = max(matched_docs, key=lambda x: x[1])

    return matched_doc
