import logging
import math
import uuid
from typing import Any, Callable, Iterable

import chromadb
import chromadb.config
from bot.memory.embedder import HuggingFaceEmbedder
from chromadb.api.types import ID, OneOrMany, Where, WhereDocument
from entities.document import Document

logger = logging.getLogger(__name__)


def _results_to_docs(results: Any) -> list[Document]:
    return [doc for doc, _ in _results_to_docs_and_scores(results)]


def _results_to_docs_and_scores(results: Any) -> list[tuple[Document, float]]:
    return [
        # TODO: Chroma can do batch querying,
        # we shouldn't hard code to the 1st result
        (Document(page_content=result[0], metadata=result[1] or {}), result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


class Chroma:
    def __init__(
        self,
        embedding_function: HuggingFaceEmbedder | None = None,
        persist_directory: str | None = None,
        client_settings: chromadb.config.Settings | None = None,
        collection_name: str = "default",
        collection_metadata: dict | None = None,
        client: chromadb.Client = None,
        relevance_score_fn: Callable[[float], float] | None = None,
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
        self.override_relevance_score_fn = relevance_score_fn

    @property
    def embeddings(self) -> HuggingFaceEmbedder | None:
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
        """Query the chroma collection."""

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
        # TODO: Handle the case where the user doesn't provide ids on the Collection
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]
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

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
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
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Run similarity search with Chroma with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to
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

        return _results_to_docs_and_scores(results)

    @staticmethod
    def _cosine_relevance_score_fn(distance: float) -> float:
        """Normalize the distance to a score on a scale [0, 1]."""

        return 1.0 - distance

    @staticmethod
    def _euclidean_relevance_score_fn(distance: float) -> float:
        """Return a similarity score on a scale [0, 1]."""
        # The 'correct' relevance function
        # may differ depending on a few things, including:
        # - the distance / similarity metric used by the VectorStore
        # - the scale of your embeddings (OpenAI's are unit normed. Many
        #  others are not!)
        # - embedding dimensionality
        # - etc.
        # This function converts the euclidean norm of normalized embeddings
        # (0 is most similar, sqrt(2) most dissimilar)
        # to a similarity function (0 to 1)
        return 1.0 - distance / math.sqrt(2)

    @staticmethod
    def _max_inner_product_relevance_score_fn(distance: float) -> float:
        """Normalize the distance to a score on a scale [0, 1]."""
        if distance > 0:
            return 1.0 - distance

        return -1.0 * distance

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn:
            return self.override_relevance_score_fn

        distance = "l2"
        distance_key = "hnsw:space"
        metadata = self._collection.metadata

        if metadata and distance_key in metadata:
            distance = metadata[distance_key]

        if distance == "cosine":
            return self._cosine_relevance_score_fn
        elif distance == "l2":
            return self._euclidean_relevance_score_fn
        elif distance == "ip":
            return self._max_inner_product_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance metric of type: {distance}."
                "Consider providing relevance_score_fn to Chroma constructor."
            )

    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """
        Default similarity search with relevance scores. Modify if necessary
        in subclass.
        Return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Args:
            query: input text
            k: Number of Documents to return. Defaults to 4.
            **kwargs: kwargs to be passed to similarity search. Should include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of Tuples of (doc, similarity_score)
        """
        relevance_score_fn = self._select_relevance_score_fn()
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Args:
            query: input text
            k: Number of Documents to return. Defaults to 4.
            **kwargs: kwargs to be passed to similarity search. Should include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of Tuples of (doc, similarity_score)
        """
        score_threshold = kwargs.pop("score_threshold", None)

        docs_and_similarities = self._similarity_search_with_relevance_scores(query, k=k, **kwargs)
        if any(similarity < 0.0 or similarity > 1.0 for _, similarity in docs_and_similarities):
            logger.warning("Relevance scores must be between" f" 0 and 1, got {docs_and_similarities}")

        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity) for doc, similarity in docs_and_similarities if similarity >= score_threshold
            ]
            if len(docs_and_similarities) == 0:
                logger.warning(
                    "No relevant docs were retrieved using the relevance score" f" threshold {score_threshold}"
                )
        return docs_and_similarities

    def delete_collection(self) -> None:
        """Delete the collection."""
        self._client.delete_collection(self._collection.name)

    def get(
        self,
        ids: OneOrMany[ID] | None = None,
        where: Where | None = None,
        limit: int | None = None,
        offset: int | None = None,
        where_document: WhereDocument | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Gets the collection.

        Args:
            ids: The ids of the embeddings to get. Optional.
            where: A Where type dict used to filter results by.
                   E.g. `{"color" : "red", "price": 4.20}`. Optional.
            limit: The number of documents to return. Optional.
            offset: The offset to start returning results from.
                    Useful for paging results with limit. Optional.
            where_document: A WhereDocument type dict used to filter by the documents.
                            E.g. `{$contains: "hello"}`. Optional.
            include: A list of what to include in the results.
                     Can contain `"embeddings"`, `"metadatas"`, `"documents"`.
                     Ids are always included.
                     Defaults to `["metadatas", "documents"]`. Optional.
        """
        kwargs = {
            "ids": ids,
            "where": where,
            "limit": limit,
            "offset": offset,
            "where_document": where_document,
        }

        if include is not None:
            kwargs["include"] = include

        return self._collection.get(**kwargs)

    def update_document(self, document_id: str, document: Document) -> None:
        """Update a document in the collection.

        Args:
            document_id (str): ID of the document to update.
            document (Document): Document to update.
        """
        return self.update_documents([document_id], [document])

    def update_documents(self, ids: list[str], documents: list[Document]) -> None:
        """Update a document in the collection.

        Args:
            ids (List[str]): List of ids of the document to update.
            documents (List[Document]): List of documents to update.
        """
        text = [document.page_content for document in documents]
        metadata = [document.metadata for document in documents]
        if self._embedding_function is None:
            raise ValueError("For update, you must specify an embedding function on creation.")
        embeddings = self._embedding_function.embed_documents(text)

        if hasattr(self._collection._client, "max_batch_size"):  # for Chroma 0.4.10 and above
            from chromadb.utils.batch_utils import create_batches

            for batch in create_batches(
                api=self._collection._client,
                ids=ids,
                metadatas=metadata,
                documents=text,
                embeddings=embeddings,
            ):
                self._collection.update(
                    ids=batch[0],
                    embeddings=batch[1],
                    documents=batch[3],
                    metadatas=batch[2],
                )
        else:
            self._collection.update(
                ids=ids,
                embeddings=embeddings,
                documents=text,
                metadatas=metadata,
            )

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: HuggingFaceEmbedder | None = None,
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        collection_name: str = "default",
        persist_directory: str | None = None,
        client_settings: chromadb.config.Settings | None = None,
        client=None,
        collection_metadata: dict | None = None,
        **kwargs: Any,
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
        chroma_collection = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            client_settings=client_settings,
            client=client,
            collection_metadata=collection_metadata,
            **kwargs,
        )
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]
        if hasattr(chroma_collection._client, "max_batch_size"):  # for Chroma 0.4.10 and above
            from chromadb.utils.batch_utils import create_batches

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
        else:
            chroma_collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return chroma_collection

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> None:
        """Delete by vector IDs.

        Args:
            ids: List of ids to delete.
        """
        self._collection.delete(ids=ids)
