import logging
from typing import Any, Iterable

import chromadb
import chromadb.config
from bot.memory.embedder import Embedder
from bot.memory.vector_database.distance_metric import DistanceMetric, get_relevance_score_fn
from bot.memory.vector_database.id_generator import generate_deterministic_ids
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
        is_persistent: bool = False,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> None:
        """
        Initializes a Chroma vector database instance.

        Args:
            client (chromadb.Client, optional): An existing Chroma client instance. If not provided, a new client will
                be created. Defaults to None.
            embedding (Embedder | None, optional): An instance of the Embedder class to generate embeddings for the
                texts. If not provided, Chroma will use sentence transformer embedding function as a default.
                Defaults to None.
            persist_directory (str | None, optional): Directory path to persist the Chroma collection. If not provided,
                the collection will be stored in memory. Defaults to None.
            collection_name (str, optional): Name of the Chroma collection to use or create. Defaults to "default".
            collection_metadata (dict | None, optional): Optional metadata to associate with the Chroma collection.
                Defaults to None.
            is_persistent (bool, optional): Whether to persist the Chroma collection to disk. If True, the collection
                will be saved to the specified persist_directory. If False, the collection will be stored in memory.
                    Defaults to True.
            distance_metric (DistanceMetric, optional): The distance metric to use for similarity search.
                Defaults to DistanceMetric.COSINE.
        """
        if is_persistent:
            client_settings = chromadb.config.Settings(is_persistent=is_persistent, persist_directory=persist_directory)
        else:
            client_settings = chromadb.config.Settings(is_persistent=is_persistent)

        if client is not None:
            self.client = client
        else:
            self.client = chromadb.Client(client_settings)

        self.embedding = embedding
        self.distance_metric = distance_metric

        # If embedding_function is None, Chroma will use Sentence Transformer all-MiniLM-L6-v2 embedding
        # function as a default.
        # We provide embeddings directly when adding data to a collection.
        # In this case, the collection will not have an embedding function set, and we are responsible for providing
        # embeddings directly when adding data and querying.
        # https://docs.trychroma.com/docs/collections/manage-collections#embedding-functions

        # Chroma’s default metric when creating a collection is L2 distance (squared L2 norm), configurable
        # to cosine or inner product (ip). Hence, Lower scores = better matches.
        # We set the Cosine by default. For Chroma whatever score represent distance;
        # So the get the right similarity we have to apply 1 - score in similarity_search_with_relevance_scores method.
        # https://docs.trychroma.com/cloud/search-api/ranking#understanding-scores
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": self.distance_metric.value}},
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

    def __dedupe(self, ids: list[str], texts: list[str]) -> tuple[list[int], list[str], list[str]]:
        """
        Deduplicates the input texts based on their IDs.

        Args:
            ids (list[str]): List of IDs corresponding to the texts.
            texts (list[str]): List of texts to be deduplicated.

        Returns:
            Tuple containing deduplicated lists of IDs, texts, and metadata (if provided).
        """

        # Keep only the first occurrence of duplicated IDs
        seen = set()
        deduped_indices = []
        for id, value in enumerate(ids):
            if value not in seen:
                deduped_indices.append(id)
                seen.add(value)

        deduped_ids = list(seen)
        deduped_texts = [texts[i] for i in deduped_indices]

        return deduped_indices, deduped_ids, deduped_texts

    def add_texts(
        self,
        texts: Iterable[str],
        metadata: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        Adds a batch of texts to the Chroma collection.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadata (list[dict] | None): Optional list of metadata.
            ids (list[str] | None): Optional list of IDs. If not provided,
                deterministic IDs will be generated from normalized text content
                to enable deduplication.

        Returns:
            List[str]: List of IDs of the added texts.
        """

        if self.embedding is None:
            raise ValueError("Embedding function is not defined for this Chroma instance.")

        texts = list(texts)

        if ids is None:
            ids = generate_deterministic_ids(texts)

        deduped_indices, deduped_ids, deduped_texts = self.__dedupe(ids, texts)
        embeddings = self.embedding.embed_documents(deduped_texts)

        try:
            if metadata:
                # fill metadata with empty dicts if somebody
                # did not specify metadata for all texts
                length_diff = len(texts) - len(metadata)

                if length_diff:
                    metadata = metadata + [{}] * length_diff

                deduped_metadata = [metadata[i] for i in deduped_indices]

                non_empty_ids = [idx for idx, m in enumerate(deduped_metadata) if m]
                empty_ids = [idx for idx, m in enumerate(deduped_metadata) if not m]

                if non_empty_ids:
                    # Upsert texts with metadata
                    metadata = [deduped_metadata[idx] for idx in non_empty_ids]
                    texts_with_metadata = [deduped_texts[idx] for idx in non_empty_ids]
                    embeddings_with_metadata = [embeddings[idx] for idx in non_empty_ids] if embeddings else None
                    ids_with_metadata = [deduped_ids[idx] for idx in non_empty_ids]
                    self.collection.upsert(
                        metadatas=metadata,
                        embeddings=embeddings_with_metadata,
                        documents=texts_with_metadata,
                        ids=ids_with_metadata,
                    )

                if empty_ids:
                    # Upsert texts without metadata
                    texts_without_metadata = [deduped_texts[j] for j in empty_ids]
                    embeddings_without_metadata = [embeddings[j] for j in empty_ids] if embeddings else None
                    ids_without_metadata = [deduped_ids[j] for j in empty_ids]
                    self.collection.upsert(
                        embeddings=embeddings_without_metadata,
                        documents=texts_without_metadata,
                        ids=ids_without_metadata,
                    )

            else:
                self.collection.upsert(
                    embeddings=embeddings,
                    documents=deduped_texts,
                    ids=deduped_ids,
                )
        except Exception as e:
            logger.error(f"Error adding texts to Chroma collection: {e}")
            raise
        return ids

    def from_texts(
        self,
        texts: list[str],
        metadata: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """
        Adds a batch of texts to the Chroma collection, optionally with metadata and IDs.

        Args:
            texts (list[str]): List of texts to add to the collection.
            metadata (list[dict], optional): List of metadata dictionaries corresponding to the texts.
                Defaults to None.
            ids (list[str], optional): List of IDs for the texts. If not provided,
                deterministic IDs will be generated from normalized text content
                to enable deduplication.
                Defaults to None.

        Returns:
            None
        """
        # Generate deterministic IDs if not provided
        if ids is None:
            ids = generate_deterministic_ids(texts)

        for batch in create_batches(
            api=self.client,
            ids=ids,
            metadatas=metadata,
            documents=texts,
        ):
            self.add_texts(
                texts=batch[3] if batch[3] else [],
                metadata=batch[2] if batch[2] else None,
                ids=batch[0],
            )

    def from_chunks(self, chunks: list[Document]) -> None:
        """
        Add document chunks to the vector database index.

        Args:
            chunks (list): List of Document chunks to add to the collection.
        """
        texts = [clean(doc.page_content, no_emoji=True) for doc in chunks]
        metadata = [doc.metadata for doc in chunks]
        self.from_texts(
            texts=texts,
            metadata=metadata,
        )

    def get_indexed_documents(self) -> list[str]:
        """
        Get list of unique document sources in the index.

        Args:
            index: Chroma vector database instance

        Returns:
            List of unique source document names
        """
        try:
            # Get all items from collection
            results = self.collection.get()
            if results and "metadatas" in results:
                sources = set()
                for metadatas in results["metadatas"]:
                    if metadatas and "source" in metadatas:
                        sources.add(metadatas["source"])
                return sorted(sources)
        except Exception as e:
            logger.warning(f"Could not retrieve indexed documents: {e}")
        return []

    def delete_collection(self, collection_name: str = "default") -> None:
        """
        Deletes the entire Chroma collection, removing all indexed data.

        Args:
            collection_name (str): The name of the Chroma collection to delete. Defaults to "default".
        """
        try:
            self.client.delete_collection(name=collection_name)
            logger.info("Chroma collection deleted successfully.")
        except Exception as e:
            logger.error(f"Error deleting Chroma collection: {e}", exc_info=True, stack_info=True)
            raise

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
                logger.warning(f"No relevant docs were retrieved using the relevance score threshold {threshold}")

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
        relevance_score_fn = get_relevance_score_fn(self.distance_metric)

        docs_and_scores = self.similarity_search_with_score(query, k)
        docs_and_similarities = [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]
        if any(similarity < 0.0 or similarity > 1.0 for _, similarity in docs_and_similarities):
            logger.warning(f"Relevance scores must be between 0 and 1, got {docs_and_similarities}")
        return docs_and_similarities
