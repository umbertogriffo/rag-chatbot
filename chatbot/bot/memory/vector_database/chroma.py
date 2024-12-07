import logging
import uuid
from typing import Any, Callable, Iterable

import chromadb
import chromadb.config
from bot.memory.embedder import Embedder
from bot.memory.vector_database.distance_metric import DistanceMetric, get_relevance_score_fn
from chromadb.utils.batch_utils import create_batches
from cleantext import clean
from entities.document import Document

logger = logging.getLogger(__name__)


class Chroma:
    def __init__(
        self,
        client: chromadb.Client = None,
        embedding: Embedder | None = None,
        persist_directory: str | None = None,
        collection_name: str = "default",
        collection_metadata: dict | None = None,
        is_persistent: bool = True,
    ) -> None:
        client_settings = chromadb.config.Settings(is_persistent=is_persistent)
        client_settings.persist_directory = persist_directory

        if client is not None:
            self.client = client
        else:
            self.client = chromadb.Client(client_settings)

        self.embedding = embedding

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata=collection_metadata,
        )

    @property
    def embeddings(self) -> Embedder | None:
        return self.embedding

    def __query_collection(
        self,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 4,
        where: dict[str, str] | None = None,
        where_document: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        """
        Query the chroma collection.

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

        return self.collection.query(
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
        """
        Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (list[dict] | None): Optional list of metadatas.
            ids (list[dict] | None): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        embeddings = None
        texts = list(texts)
        if self.embedding is not None:
            embeddings = self.embedding.embed_documents(texts)
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
                    self.collection.upsert(
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
                self.collection.upsert(
                    embeddings=embeddings_without_metadatas,
                    documents=texts_without_metadatas,
                    ids=ids_without_metadatas,
                )
        else:
            self.collection.upsert(
                embeddings=embeddings,
                documents=texts,
                ids=ids,
            )
        return ids

    def from_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """
        Adds a batch of texts to the Chroma collection, optionally with metadata and IDs.

        Args:
            texts (list[str]): List of texts to add to the collection.
            metadatas (list[dict], optional): List of metadata dictionaries corresponding to the texts.
                Defaults to None.
            ids (list[str], optional): List of IDs for the texts. If not provided, UUIDs will be generated.
                Defaults to None.

        Returns:
            None
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        for batch in create_batches(
            api=self.client,
            ids=ids,
            metadatas=metadatas,
            documents=texts,
        ):
            self.add_texts(
                texts=batch[3] if batch[3] else [],
                metadatas=batch[2] if batch[2] else None,
                ids=batch[0],
            )

    def from_chunks(self, chunks: list) -> None:
        """
        Adds a batch of documents to the Chroma collection.

        Args:
            chunks (list): List of Document objects to add to the collection.
        """
        texts = [clean(doc.page_content, no_emoji=True) for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]
        self.from_texts(
            texts=texts,
            metadatas=metadatas,
        )

    def similarity_search_with_threshold(
        self,
        query: str,
        k: int = 4,
        threshold: float | None = 0.2,
    ) -> tuple[list[Document], list[dict[str, Any]]]:
        """
        Performs similarity search on the given query.

        Parameters:
        -----------
        query : str
            The query string.

        k : int, optional
            The number of retrievals to consider (default is 4).

        threshold : float, optional
            The threshold for considering similarity scores (default is 0.2).

        Returns:
        -------
        tuple[list[Document], list[dict[str, Any]]]
            A tuple containing the list of matched documents and a list of their sources.

        """
        # `similarity_search_with_relevance_scores` return docs and relevance scores in the range [0, 1].
        # 0 is dissimilar, 1 is most similar.
        docs_and_scores = self.similarity_search_with_relevance_scores(query, k)

        if threshold is not None:
            docs_and_scores = [doc for doc in docs_and_scores if doc[1] > threshold]
            if len(docs_and_scores) == 0:
                logger.warning("No relevant docs were retrieved using the relevance score" f" threshold {threshold}")

            docs_and_scores = sorted(docs_and_scores, key=lambda x: x[1], reverse=True)

        retrieved_contents = [doc[0] for doc in docs_and_scores]
        sources = []
        for doc, score in docs_and_scores:
            sources.append(
                {
                    "score": round(score, 3),
                    "document": doc.metadata.get("source"),
                    "content_preview": f"{doc.page_content[0:256]}...",
                }
            )

        return retrieved_contents, sources

    def similarity_search(self, query: str, k: int = 4, filter: dict[str, str] | None = None) -> list[Document]:
        """
        Run similarity search with Chroma.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (dict[str, str]|None): Filter by metadata. Defaults to None.

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
            filter (dict[str, str]|None): Filter by metadata. Defaults to None.
            where_document (dict[str, str]|None): Filter by document content. Defaults to None.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            list[tuple[Document, float]]: List of documents most similar to
            the query text and cosine distance in float for each.
            Lower score represents more similarity.
        """
        if self.embedding is None:
            results = self.__query_collection(
                query_texts=[query],
                n_results=k,
                where=filter,
                where_document=where_document,
            )
        else:
            query_embedding = self.embedding.embed_query(query)
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
        metadata = self.collection.metadata

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
