from typing import Any

import sentence_transformers


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_folder: str | None = None, **kwargs: Any):
        """
        Initialize the Embedder class with the specified parameters.

        Args:
            **kwargs (Any): Additional keyword arguments to pass to the SentenceTransformer model.
        """
        self.client = sentence_transformers.SentenceTransformer(model_name, cache_folder=cache_folder, **kwargs)

    def embed_documents(self, texts: list[str], multi_process: bool = False, **encode_kwargs: Any) -> list[list[float]]:
        """
        Compute document embeddings using a transformer model.

        Args:
            texts (list[str]): The list of texts to embed.
            multi_process (bool): If True, use multiple processes to compute embeddings.
            **encode_kwargs (Any): Additional keyword arguments to pass when calling the `encode` method of the model.

        Returns:
            list[list[float]]: A list of embeddings, one for each text.
        """

        texts = list(map(lambda x: x.replace("\n", " "), texts))
        if multi_process:
            pool = self.client.start_multi_process_pool()
            embeddings = self.client.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self.client.encode(texts, show_progress_bar=True, **encode_kwargs)

        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Compute query embeddings using a transformer model.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: Embeddings for the text.
        """
        return self.embed_documents([text])[0]
