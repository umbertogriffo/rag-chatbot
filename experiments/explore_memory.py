from pathlib import Path

import chromadb
from langchain_community.vectorstores.chroma import Chroma

from bot.memory.embedder import EmbedderHuggingFace
from bot.memory.vector_memory import VectorMemory

if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent.parent
    # Contains an extract of documents uploaded to the RAG bot;
    declarative_vector_store_path = root_folder / "vector_store" / "docs_index"
    # Contains an extract of things the user said in the past;
    episodic_vector_store_path = root_folder / "vector_store" / "episodic_index"

    embedding = EmbedderHuggingFace().get_embedding()
    index = VectorMemory(vector_store_path=str(declarative_vector_store_path), embedding=embedding)

    # query = "<write_your_query_here>"
    query = "tell me a joke about ClearML"

    matched_docs, sources = index.similarity_search(query)
    matched_doc = index.search_most_similar_doc(query)

    for doc, score in matched_docs:
        print("-- PAGE CONTENT --")
        print(doc.page_content)
        print("-- METADATA --")
        print(doc.metadata)

    persistent_client = chromadb.PersistentClient(path=str(episodic_vector_store_path))
    collection = persistent_client.get_or_create_collection("episodic_memory")
    collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])
    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name="episodic_memory",
        embedding_function=embedding,
    )
    docs = langchain_chroma.similarity_search("a")
    docs_with_score = langchain_chroma.similarity_search_with_score("a")
    docs_with_relevance_score = langchain_chroma.similarity_search_with_relevance_scores("a")
    matched_doc = max(docs_with_relevance_score, key=lambda x: x[1])

    # The returned distance score is cosine distance. Therefore, a lower score is better.
    results = collection.query(
        query_texts=["a"],
        n_results=2,
        # where={"metadata_field": "is_equal_to_this"}, # optional filter
        # where_document={"$contains":"search_string"}  # optional filter
    )
    print(results)