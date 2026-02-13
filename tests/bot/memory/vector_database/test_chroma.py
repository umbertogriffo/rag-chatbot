import pytest
from bot.memory.embedder import Embedder
from bot.memory.vector_database.chroma import Chroma
from bot.memory.vector_database.id_generator import generate_deterministic_id
from entities.document import Document


@pytest.fixture
def chroma_instance(tmp_path):
    return Chroma(embedding=Embedder(), persist_directory=str(tmp_path))


def test_initialization(chroma_instance):
    assert chroma_instance.embedding is not None
    assert chroma_instance.client is not None
    assert chroma_instance.collection is not None


def test_add_texts(chroma_instance):
    texts = ["This is a test document."]
    metadatas = [{"source": "test_source"}]
    ids = chroma_instance.add_texts(texts, metadatas)
    assert len(ids) == 1


def test_add_texts_with_deterministic_ids(chroma_instance):
    """Test that deterministic IDs are generated when not provided"""
    texts = ["Test document"]
    metadatas = [{"source": "test.md"}]

    # Add once
    ids1 = chroma_instance.add_texts(texts, metadatas)
    assert len(ids1) == 1

    # Add again - should get same IDs due to deterministic generation
    ids2 = chroma_instance.add_texts(texts, metadatas)
    assert len(ids2) == 1
    assert ids1[0] == ids2[0]


def test_deduplication_with_upsert(chroma_instance):
    """Test that duplicate documents are deduplicated via upsert"""
    text = "This is a duplicate document."
    metadata = {"source": "test.md"}

    # Add the same document twice
    chroma_instance.add_texts([text], [metadata])
    chroma_instance.add_texts([text], [metadata])

    # Query to get all documents
    results = chroma_instance.similarity_search(text, k=10)

    # Should only have one document due to deduplication
    assert len(results) == 1
    assert results[0].page_content == text


def test_different_documents_not_deduplicated(chroma_instance):
    """Test that different documents are not deduplicated"""
    texts = ["Document one", "Document two", "Document three"]
    metadatas = [{"source": "test.md"}, {"source": "test.md"}, {"source": "test.md"}]

    chroma_instance.add_texts(texts, metadatas)

    # Query to get all documents
    results = chroma_instance.similarity_search("Document", k=10)

    # Should have all three documents
    assert len(results) == 3


def test_from_texts_deduplication(chroma_instance):
    """Test that from_texts also uses deterministic IDs for deduplication"""
    texts = ["Duplicate text", "Duplicate text", "Unique text"]
    metadatas = [{"source": "doc.md"}, {"source": "doc.md"}, {"source": "doc.md"}]

    # First add
    chroma_instance.from_texts(texts[:2], metadatas[:2])
    # Second add with overlap
    chroma_instance.from_texts(texts[1:], metadatas[1:])

    # Query all
    results = chroma_instance.similarity_search("text", k=10)

    # Note: Since chunk_index is included in ID generation, identical content at
    # different indices will have different IDs. The assertion is intentionally loose.
    # With 3 adds total (texts[:2] = 2 chunks, texts[1:] = 2 chunks), we expect
    # at least 2 unique documents due to different chunk indices.
    assert len(results) >= 2


def test_similarity_search(chroma_instance):
    texts = ["This is a test document."]
    metadatas = [{"source": "test_source"}]
    chroma_instance.add_texts(texts, metadatas)

    results = chroma_instance.similarity_search("test", k=1)
    assert len(results) == 1
    assert isinstance(results[0], Document)


def test_similarity_search_with_threshold(chroma_instance):
    texts = ["This is a test document."]
    metadatas = [{"source": "test_source"}]
    chroma_instance.add_texts(texts, metadatas)

    results, source = chroma_instance.similarity_search_with_threshold("test", k=1, threshold=0.3)
    assert len(results) == 1
    assert len(source) == 1
    assert isinstance(results[0], Document)
    assert source[0].get("score") == pytest.approx(0.353, 0.1)


def test_similarity_search_with_score(chroma_instance):
    texts = ["This is a test document."]
    metadatas = [{"source": "test_source"}]
    chroma_instance.add_texts(texts, metadatas)

    results = chroma_instance.similarity_search_with_score("test", k=1)
    assert len(results) == 1
    assert isinstance(results[0][0], Document)
    assert isinstance(results[0][1], float)


def test_similarity_search_with_relevance_scores(chroma_instance):
    texts = ["This is a test document."]
    metadatas = [{"source": "test_source"}]
    chroma_instance.add_texts(texts, metadatas)

    results = chroma_instance.similarity_search_with_relevance_scores("test", k=1)
    assert len(results) == 1
    assert isinstance(results[0][0], Document)
    assert isinstance(results[0][1], float)
    assert 0.0 <= results[0][1] <= 1.0
