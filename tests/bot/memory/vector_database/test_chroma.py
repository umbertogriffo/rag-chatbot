import pytest
from bot.memory.embedder import Embedder
from bot.memory.vector_database.chroma import Chroma
from entities.document import Document


@pytest.fixture
def chroma_instance(tmp_path):
    return Chroma(embedding=Embedder(), persist_directory=str(tmp_path), is_persistent=True)


def test_initialization(chroma_instance):
    """Test that the Chroma instance initializes correctly with the provided embedding and persist directory"""
    assert chroma_instance.embedding is not None
    assert chroma_instance.client is not None
    assert chroma_instance.collection is not None


def test_add_texts(chroma_instance):
    """Test that texts can be added to the Chroma collection and that IDs are returned"""
    texts = ["This is a test document."]
    metadata = [{"source": "test_source"}]
    ids = chroma_instance.add_texts(texts, metadata)
    assert len(ids) == 1


def test_add_texts_with_missing_metadata(chroma_instance):
    """Test that texts can be added even if metadata is missing"""
    texts = ["Test document 1", "Test document 2"]
    metadata = [{"source": "test.md"}]

    ids = chroma_instance.add_texts(texts, metadata)
    assert len(ids) == 2


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
    metadata = [{"source": "test.md"}, {"source": "test.md"}, {"source": "test.md"}]

    chroma_instance.add_texts(texts, metadata)

    results = chroma_instance.similarity_search("Document", k=10)

    assert len(results) == 3


@pytest.mark.parametrize(
    "texts,metadata,expected_count",
    [
        (["Duplicate text", "Duplicate text", "Unique text"], [], 2),
        (["Duplicate text", "Duplicate text", "Unique text"], [{"source": "doc.md"}], 2),
        (
            ["Duplicate text", "Duplicate text", "Unique text"],
            [{"source": "doc.md"}, {"source": "doc.md"}, {"source": "unique.md"}],
            2,
        ),
        (["Duplicate text", "Duplicate text", "Unique text"], [{"source": "doc.md"}, {"source": "doc.md"}], 2),
        (["Duplicate text", "Duplicate text", "Unique text"], [{"source": "doc.md"}, {}, {"source": "unique.md"}], 2),
    ],
    ids=[
        "no_metadata",
        "single_metadata_for_all",
        "full_metadata_with_duplicates",
        "partial_metadata_missing_last",
        "partial_metadata_with_empty_dict",
    ],
)
def test_from_texts_deduplication(chroma_instance, texts, metadata, expected_count):
    """Test that from_texts method deduplicates duplicate documents"""
    chroma_instance.from_texts(texts, metadata)

    results = chroma_instance.similarity_search("text", k=10)

    assert len(results) == expected_count
    assert results[1].page_content == "Duplicate text"
    assert results[1].metadata.get("source") == "doc.md" or results[1].metadata.get("source") is None

    assert results[0].page_content == "Unique text"
    assert results[0].metadata.get("source") == "unique.md" or results[0].metadata.get("source") is None


def test_similarity_search(chroma_instance):
    """Test that similarity search returns relevant documents based on the query"""
    texts = ["This is a test document."]
    metadatas = [{"source": "test_source"}]
    chroma_instance.add_texts(texts, metadatas)

    results = chroma_instance.similarity_search("test", k=1)
    assert len(results) == 1
    assert isinstance(results[0], Document)


def test_similarity_search_with_threshold(chroma_instance):
    """Test that similarity search with threshold returns documents that meet the relevance threshold"""
    texts = ["This is a test document."]
    metadatas = [{"source": "test_source"}]
    chroma_instance.add_texts(texts, metadatas)

    results, source = chroma_instance.similarity_search_with_threshold("test", k=1, threshold=0.3)
    assert len(results) == 1
    assert len(source) == 1
    assert isinstance(results[0], Document)
    assert source[0].get("score") == pytest.approx(0.543, 0.1)


def test_similarity_search_with_score(chroma_instance):
    """Test that similarity search with score returns documents along with their relevance scores"""
    texts = ["This is a test document."]
    metadatas = [{"source": "test_source"}]
    chroma_instance.add_texts(texts, metadatas)

    results = chroma_instance.similarity_search_with_score("test", k=1)
    assert len(results) == 1
    assert isinstance(results[0][0], Document)
    assert isinstance(results[0][1], float)


def test_similarity_search_with_relevance_scores(chroma_instance):
    """Test that similarity search with relevance scores returns documents along with normalized relevance scores"""
    texts = ["This is a test document."]
    metadatas = [{"source": "test_source"}]
    chroma_instance.add_texts(texts, metadatas)

    results = chroma_instance.similarity_search_with_relevance_scores("test", k=1)
    assert len(results) == 1
    assert isinstance(results[0][0], Document)
    assert isinstance(results[0][1], float)
    assert 0.0 <= results[0][1] <= 1.0
