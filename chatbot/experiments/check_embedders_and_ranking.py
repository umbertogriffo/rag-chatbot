import time
from pathlib import Path

import torch
from bot.memory.embedder import Embedder
from bot.memory.vector_database.chroma import Chroma

ROOT_FOLDER = Path(__file__).resolve().parents[2]
EXPERIMENTS_FOLDER = Path(__file__).parent


def load_texts(file_path: Path) -> list[str]:
    # Example from https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html#manual-implementation
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def run_experiment(texts: list[str], query: str, model_name: str = "all-MiniLM-L6-v2") -> None:
    print(f"---{model_name}---")

    start_time = time.time()

    embedding = Embedder(model_name=model_name)
    index = Chroma(embedding=embedding)
    index.from_texts(texts=texts)

    # results = index.similarity_search_with_score(query)
    #
    # documents = [doc for doc, _ in results]
    # similarity_scores = [score for _, score in results]
    # scores, indices = torch.topk(torch.tensor(similarity_scores), k=4)
    # for score, idx in zip(scores, indices):
    #     print(f"(Score: {score:.4f})", documents[idx])
    #     pass
    #
    # print("---")

    relevant_results = index.similarity_search_with_relevance_scores(query)

    relevant_documents = [doc for doc, _ in relevant_results]
    relevance_scores = [score for _, score in relevant_results]
    scores, indices = torch.topk(torch.tensor(relevance_scores), k=4)
    for score, idx in zip(scores, indices):
        print(f"(Score: {score:.4f})", relevant_documents[idx])

    took = time.time() - start_time
    print(f"--- Took {took:.2f} seconds ---")

    index.delete_collection()


if __name__ == "__main__":
    texts_file_path = EXPERIMENTS_FOLDER / "sample_texts.txt"

    texts = load_texts(texts_file_path)

    query = "How do artificial neural networks work?"

    # Original Models: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#original-models
    run_experiment(texts, query, model_name="all-MiniLM-L6-v2")
    run_experiment(texts, query, model_name="all-MiniLM-L12-v2")
    run_experiment(texts, query, model_name="all-mpnet-base-v2")
