"""
MIT License

Copyright (c) LangChain, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging
import uuid
from typing import Any, Callable, Iterable

import chromadb
import chromadb.config
from bot.memory.embedder import Embedder
from chromadb.utils.batch_utils import create_batches
from entities.document import Document
from vector_database.distance_metric import DistanceMetric, get_relevance_score_fn

logger = logging.getLogger(__name__)


class Chroma:
    """
    Chroma classes have been extracted and refactored from LangChain's project.
    https://github.com/langchain-ai/langchain/blob/907c758d67764385828c8abad14a3e64cf44d05b/libs/partners/chroma/langchain_chroma/vectorstores.py#L133
    """

    def __init__(
        self,
        embedding_function: Embedder | None = None,
        persist_directory: str | None = None,
        client_settings: chromadb.config.Settings | None = None,
        collection_name: str = "default",
        collection_metadata: dict | None = None,
        client: chromadb.Client = None,
    ) -> None:
        """Initialize with a Chroma client."""

        if client is not None:
            self._client_settings = client_settings
            self._client = client
            self._persist_directory = persist_directory
        else:
            if client_settings:
                # If client_settings is provided with persist_directory specified,
                # then it is "in-memory and persisting to disk" mode.
                client_settings.persist_directory = persist_directory or client_settings.persist_directory
                _client_settings = client_settings
            elif persist_directory:
                # Maintain backwards compatibility with chromadb < 0.4.0
                _client_settings = chromadb.config.Settings(is_persistent=True)
                _client_settings.persist_directory = persist_directory
            else:
                _client_settings = chromadb.config.Settings()
            self._client_settings = _client_settings
            self._client = chromadb.Client(_client_settings)
            self._persist_directory = _client_settings.persist_directory or persist_directory

        self._embedding_function = embedding_function
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata=collection_metadata,
        )

    @property
    def embeddings(self) -> Embedder | None:
        return self._embedding_function

    def __query_collection(
        self,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 4,
        where: dict[str, str] | None = None,
        where_document: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        """Query the chroma collection.

        Args:
            query_texts: List of query texts.
            query_embeddings: List of query embeddings.
            n_results: Number of results to return. Defaults to 4.
            where: dict used to filter results by
                    e.g. {"color" : "red", "price": 4.20}.
            where_document: dict used to filter by the documents.
                    E.g. {$contains: {"text": "hello"}}.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns:
            List of `n_results` nearest neighbor embeddings for provided
            query_embeddings or query_texts.

        See more: https://docs.trychroma.com/reference/py-collection#query
        """

        return self._collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            **kwargs,
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        embeddings = None
        texts = list(texts)
        if self._embedding_function is not None:
            embeddings = self._embedding_function.embed_documents(texts)
        if metadatas:
            # fill metadatas with empty dicts if somebody
            # did not specify metadata for all texts
            length_diff = len(texts) - len(metadatas)
            if length_diff:
                metadatas = metadatas + [{}] * length_diff
            empty_ids = []
            non_empty_ids = []
            for idx, m in enumerate(metadatas):
                if m:
                    non_empty_ids.append(idx)
                else:
                    empty_ids.append(idx)
            if non_empty_ids:
                metadatas = [metadatas[idx] for idx in non_empty_ids]
                texts_with_metadatas = [texts[idx] for idx in non_empty_ids]
                embeddings_with_metadatas = [embeddings[idx] for idx in non_empty_ids] if embeddings else None
                ids_with_metadata = [ids[idx] for idx in non_empty_ids]
                try:
                    self._collection.upsert(
                        metadatas=metadatas,
                        embeddings=embeddings_with_metadatas,
                        documents=texts_with_metadatas,
                        ids=ids_with_metadata,
                    )
                except ValueError as e:
                    if "Expected metadata value to be" in str(e):
                        msg = "Try filtering complex metadata from the document."
                        raise ValueError(e.args[0] + "\n\n" + msg)
                    else:
                        raise e
            if empty_ids:
                texts_without_metadatas = [texts[j] for j in empty_ids]
                embeddings_without_metadatas = [embeddings[j] for j in empty_ids] if embeddings else None
                ids_without_metadatas = [ids[j] for j in empty_ids]
                self._collection.upsert(
                    embeddings=embeddings_without_metadatas,
                    documents=texts_without_metadatas,
                    ids=ids_without_metadatas,
                )
        else:
            self._collection.upsert(
                embeddings=embeddings,
                documents=texts,
                ids=ids,
            )
        return ids

    @classmethod
    def from_texts(
        self,
        texts: list[str],
        embedding: Embedder | None = None,
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        collection_name: str = "default",
        persist_directory: str | None = None,
        client_settings: chromadb.config.Settings | None = None,
        client=None,
        collection_metadata: dict | None = None,
    ):
        """Create a Chroma vectorstore from a raw documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            texts (List[str]): List of texts to add to the collection.
            collection_name (str): Name of the collection to create.
            persist_directory (Optional[str]): Directory to persist the collection.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            metadatas (Optional[List[dict]]): List of metadatas. Defaults to None.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.
            client_settings (Optional[chromadb.config.Settings]): Chroma client settings
            collection_metadata (Optional[Dict]): Collection configurations.
                                                  Defaults to None.

        Returns:
            Chroma: Chroma vectorstore.
        """
        chroma_collection = self(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
        )
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        for batch in create_batches(
            api=chroma_collection._client,
            ids=ids,
            metadatas=metadatas,
            documents=texts,
        ):
            chroma_collection.add_texts(
                texts=batch[3] if batch[3] else [],
                metadatas=batch[2] if batch[2] else None,
                ids=batch[0],
            )
        return chroma_collection

    def similarity_search(self, query: str, k: int = 4, filter: dict[str, str] | None = None) -> list[Document]:
        """Run similarity search with Chroma.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Document]: List of documents most similar to the query text.
        """
        docs_and_scores = self.similarity_search_with_score(query, k, filter=filter)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, str] | None = None,
        where_document: dict[str, str] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Run similarity search with Chroma with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
            where_document (Optional[Dict[str, str]]): Filter by document content. Defaults to None.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            list[tuple[Document, float]]: List of documents most similar to
            the query text and cosine distance in float for each.
            Lower score represents more similarity.
        """
        if self._embedding_function is None:
            results = self.__query_collection(
                query_texts=[query],
                n_results=k,
                where=filter,
                where_document=where_document,
            )
        else:
            query_embedding = self._embedding_function.embed_query(query)
            results = self.__query_collection(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter,
                where_document=where_document,
            )
        return [
            (Document(page_content=result[0], metadata=result[1] or {}), result[2])
            for result in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def __select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function may differ depending on the distance/similarity metric used by the VectorStore.
        """

        distance = DistanceMetric.L2
        distance_key = "hnsw:space"
        metadata = self._collection.metadata

        if metadata and distance_key in metadata:
            distance = metadata[distance_key]
        return get_relevance_score_fn(distance)

    def similarity_search_with_relevance_scores(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """
        Return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Args:
            query: input text
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Tuples of (doc, similarity_score)
        """
        # relevance_score_fn is a function to calculate relevance score from distance.
        relevance_score_fn = self.__select_relevance_score_fn()

        docs_and_scores = self.similarity_search_with_score(query, k)
        docs_and_similarities = [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]
        if any(similarity < 0.0 or similarity > 1.0 for _, similarity in docs_and_similarities):
            logger.warning("Relevance scores must be between" f" 0 and 1, got {docs_and_similarities}")
        return docs_and_similarities
