from typing import Any

import sentence_transformers
import torch

EMBEDDER_ARGS = {
    "jinaai/jina-embeddings-v5-text-nano-retrieval": {
        "trust_remote_code": True,
        "model_kwargs": {"dtype": torch.bfloat16},  # Recommended for GPUs
        "config_kwargs": {},
    },
    "jinaai/jina-embeddings-v5-text-small-retrieval": {
        "trust_remote_code": True,
        "model_kwargs": {"dtype": torch.bfloat16},  # Recommended for GPUs
        "config_kwargs": {},
    },
}


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_folder: str | None = None, **kwargs: Any):
        """
        Initialize the Embedder class with the specified parameters.

        Args:
            model_name (str): The name of the SentenceTransformer model to use for embedding.
            cache_folder (str | None): The directory where the model will be cached.
            **kwargs (Any): Additional keyword arguments to pass to the SentenceTransformer model.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        args = EMBEDDER_ARGS.get(model_name, None)
        if args is not None:
            kwargs.update(args)

        self.client = sentence_transformers.SentenceTransformer(
            model_name_or_path=model_name, device=device, cache_folder=cache_folder, **kwargs
        )

    @staticmethod
    def _clean_texts(texts: list[str]) -> list[str]:
        """
        Clean the input texts by replacing newline characters with spaces.

        Args:
            texts (list[str]): The list of texts to clean.

        Returns:
            list[str]: The cleaned list of texts.
        """
        return [x.replace("\n", " ") for x in texts]

    def embed_documents(self, texts: list[str], multi_process: bool = False, **encode_kwargs: Any) -> list[list[float]]:
        """
        Compute document embeddings using a transformer model.

        Notes:
            The more general `SentenceTransformer.encode` method differs in two ways from
            `SentenceTransformer.encode_query` and `SentenceTransformer.encode_document`:
            - If no prompt_name or prompt is provided, it uses a predefined “query” or “document” prompt, if specified
              in the model’s prompts dictionary.
            - It sets the task to “document”. If the model has a Router module, it will use the “query” or “document”
              task type to route the input through the appropriate submodules.

        Args:
            texts (list[str]): The list of texts to embed.
            multi_process (bool): If True, use multiple processes to compute embeddings.
            **encode_kwargs (Any): Additional keyword arguments to pass when calling the `encode` method of the model.

        Returns:
            list[list[float]]: A list of embeddings, one for each text.
        """

        texts = self._clean_texts(texts)
        if multi_process:
            pool = self.client.start_multi_process_pool()
            embeddings = self.client.encode(sentences=texts, pool=pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self.client.encode(
                sentences=texts, normalize_embeddings=False, show_progress_bar=True, **encode_kwargs
            )

        return embeddings.tolist()

    def embed_query(self, text: str, **encode_kwargs) -> list[float]:
        """
        Compute query embeddings using a transformer model.

        Notes:
            The more general `SentenceTransformer.encode` method differs in two ways from
            `SentenceTransformer.encode_query` and `SentenceTransformer.encode_document`:
            - If no prompt_name or prompt is provided, it uses a predefined “query” or “document” prompt, if specified
              in the model’s prompts dictionary.
            - It sets the task to “document”. If the model has a Router module, it will use the “query” or “document”
              task type to route the input through the appropriate submodules.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: Embeddings for the text.
        """
        text = self._clean_texts([text])[0]

        embeddings = self.client.encode(
            sentences=text, normalize_embeddings=False, show_progress_bar=True, **encode_kwargs
        )

        return embeddings.tolist()
