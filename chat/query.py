from pathlib import Path

from memory.vector_memory import VectorMemory, initialize_embedding

if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent.parent
    vector_store_path = root_folder / "vector_store" / "docs_index"

    embedding = initialize_embedding()
    memory = VectorMemory(embedding=embedding)
    index = memory.load_memory_index(str(vector_store_path))

    query = "<write_your_query_here>"

    docs = index.similarity_search(query=query, k=4)

    for doc in docs:
        print("-- PAGE CONTENT --")
        print(doc.page_content)
        print("-- METADATA --")
        print(doc.metadata)
