from abc import ABC
from typing import Any

from langchain.embeddings import HuggingFaceEmbeddings


class Embedder(ABC):
    embedder: Any

    def get_embedding(self):
        return self.embedder


class EmbedderHuggingFace(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)
