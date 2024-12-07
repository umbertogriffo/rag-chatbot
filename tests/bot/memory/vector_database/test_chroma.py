import pytest
from bot.memory.embedder import Embedder
from bot.memory.vector_database.chroma import Chroma
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
