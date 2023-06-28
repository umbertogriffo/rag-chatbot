from typing import Any, List

from cleantext import clean
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from helpers.log import get_logger

logger = get_logger(__name__)


class VectorMemory:
    def __init__(self, embedding: Any, verbose=False) -> None:
        self.embedding = embedding
        self.verbose = verbose

        if self.embedding is None:
            logger.error("No embedder passed to VectorMemory")
            raise Exception("No embedder passed to VectorMemory")

    def create_memory_index(self, chunks: List, vector_store_path: str) -> Chroma:
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
        # Load the vector store to use as the index
        index = Chroma(
            persist_directory=str(vector_store_path), embedding_function=self.embedding
        )
        return index


def initialize_embedding() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def similarity_search(query: str, index: Chroma, k: int = 4):
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
    matched_docs = index.similarity_search_with_score(query, k=k)
    matched_doc = max(matched_docs, key=lambda x: x[1])

    return matched_doc
