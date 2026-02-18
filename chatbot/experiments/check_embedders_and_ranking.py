from pathlib import Path

import torch
from bot.memory.embedder import Embedder
from bot.memory.vector_database.chroma import Chroma

if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parents[2]
    # Contains an extract of documents uploaded to the RAG bot;
    declarative_vector_store_path = root_folder / "vector_store" / "exp_docs_index"

    embedding = Embedder(model_name="all-MiniLM-L6-v2")
    index = Chroma(persist_directory=str(declarative_vector_store_path), embedding=embedding)

    # Example from https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html#manual-implementation
    index.from_texts(
        texts=[
            "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
            "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
            "Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains.",
            "Mars rovers are robotic vehicles designed to travel on the surface of Mars to collect data and perform experiments.",
            "The James Webb Space Telescope is the largest optical telescope in space, designed to conduct infrared astronomy.",
            "SpaceX's Starship is designed to be a fully reusable transportation system capable of carrying humans to Mars and beyond.",
            "Global warming is the long-term heating of Earth's climate system observed since the pre-industrial period due to human activities.",
            "Renewable energy sources include solar, wind, hydro, and geothermal power that naturally replenish over time.",
            "Carbon capture technologies aim to collect CO2 emissions before they enter the atmosphere and store them underground.",
        ]
    )

    query = "How do artificial neural networks work?"

    results = index.similarity_search_with_score(query)

    documents = [doc for doc, _ in results]
    similarity_scores = [score for _, score in results]
    scores, indices = torch.topk(torch.tensor(similarity_scores), k=4)
    for score, idx in zip(scores, indices):
        print(f"(Score: {score:.4f})", documents[idx])

    print("---")

    relevant_results = index.similarity_search_with_relevance_scores(query)

    relevant_documents = [doc for doc, _ in relevant_results]
    relevance_scores = [score for _, score in relevant_results]
    scores, indices = torch.topk(torch.tensor(relevance_scores), k=4)
    for score, idx in zip(scores, indices):
        print(f"(Score: {score:.4f})", documents[idx])
