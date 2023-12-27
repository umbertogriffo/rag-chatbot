from typing import Any, Dict, List, Tuple

from cleantext import clean
from helpers.log import get_logger
from langchain.vectorstores import Chroma
from langchain_core.documents import Document

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

    def __init__(self, vector_store_path: str, embedding: Any, verbose=False) -> None:
        self.embedding = embedding
        self.verbose = verbose

        if self.embedding is None:
            logger.error("No embedder passed to VectorMemory")
            raise Exception("No embedder passed to VectorMemory")

        self.index = self.load_memory_index(vector_store_path)

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
        index = Chroma(persist_directory=str(vector_store_path), embedding_function=self.embedding)
        return index

    def similarity_search(
        self, query: str, k: int = 4, threshold: float = 0.2
    ) -> Tuple[List[Document], List[Dict[str, Any]]]:
        """
        Performs similarity search on the given query.

        Parameters:
        -----------
        query : str
            The query string.

        index : Chroma
            The Chroma index to perform the search on.

        k : int, optional
            The number of retrievals to consider (default is 4).

        threshold : float, optional
            The threshold for considering similarity scores (default is 0.2).


        Returns:
        -------
        Tuple[List[Document], List[Dict[str, Any]]]
            A tuple containing the list of matched documents and a list of their sources.

        """
        # `similarity_search_with_relevance_scores` return docs and relevance scores in the range [0, 1].
        # 0 is dissimilar, 1 is most similar.
        matched_docs = self.index.similarity_search_with_relevance_scores(query, k=k)
        filtered_docs_by_threshold = [doc for doc in matched_docs if doc[1] > threshold]
        sorted_matched_docs_by_relevance_score = sorted(filtered_docs_by_threshold, key=lambda x: x[1], reverse=True)
        retrieved_contents = [doc[0] for doc in sorted_matched_docs_by_relevance_score]
        sources = []
        for doc, score in sorted_matched_docs_by_relevance_score:
            sources.append(
                {
                    "score": round(score, 3),
                    "document": doc.metadata.get("source"),
                    "content_preview": f"{doc.page_content[0:150]}...",
                }
            )

        return retrieved_contents, sources

    @staticmethod
    def create_memory_index(embedding: Any, chunks: List, vector_store_path: str):
        texts = [clean(doc.page_content, no_emoji=True) for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]
        memory_index = Chroma.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            persist_directory=vector_store_path,
        )
        memory_index.persist()
