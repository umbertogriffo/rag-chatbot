import pytest
from bot.memory.vector_database.id_generator import (
    generate_deterministic_id,
    generate_deterministic_ids,
    normalize_text,
)


class TestNormalizeText:
    def test_lowercase_conversion(self):
        assert normalize_text("HELLO WORLD") == "hello world"
        assert normalize_text("HeLLo WoRLd") == "hello world"

    def test_whitespace_collapse(self):
        assert normalize_text("hello  world") == "hello world"
        assert normalize_text("hello\n\nworld") == "hello world"
        assert normalize_text("hello\t\tworld") == "hello world"
        assert normalize_text("  hello   world  ") == "hello world"

    def test_unicode_normalization(self):
        # Test unicode normalization
        text1 = "café"  # é as single character
        text2 = "café"  # é as combining character
        assert normalize_text(text1) == normalize_text(text2)

    def test_strip_leading_trailing(self):
        assert normalize_text("  hello  ") == "hello"
        assert normalize_text("\nhello\n") == "hello"


class TestGenerateDeterministicId:
    def test_same_content_same_id(self):
        """Same content should always produce the same ID"""
        text = "This is a test document."
        id1 = generate_deterministic_id(text)
        id2 = generate_deterministic_id(text)
        assert id1 == id2

    def test_different_content_different_id(self):
        """Different content should produce different IDs"""
        text1 = "This is document one."
        text2 = "This is document two."
        id1 = generate_deterministic_id(text1)
        id2 = generate_deterministic_id(text2)
        assert id1 != id2

    def test_normalization_consistency(self):
        """Variations in whitespace and case should produce same ID"""
        text1 = "Hello World"
        text2 = "hello world"
        text3 = "HELLO  WORLD"
        text4 = "  hello   world  "

        id1 = generate_deterministic_id(text1)
        id2 = generate_deterministic_id(text2)
        id3 = generate_deterministic_id(text3)
        id4 = generate_deterministic_id(text4)

        assert id1 == id2 == id3 == id4

    def test_with_source(self):
        """Same content from different sources should produce different IDs"""
        text = "This is a test."
        id1 = generate_deterministic_id(text, source="doc1.md")
        id2 = generate_deterministic_id(text, source="doc2.md")
        assert id1 != id2

    def test_with_chunk_index(self):
        """Same content with different chunk indices should produce different IDs"""
        text = "This is a test."
        id1 = generate_deterministic_id(text, chunk_index=0)
        id2 = generate_deterministic_id(text, chunk_index=1)
        assert id1 != id2

    def test_with_source_and_chunk_index(self):
        """Test with both source and chunk index"""
        text = "This is a test."
        id1 = generate_deterministic_id(text, source="doc.md", chunk_index=0)
        id2 = generate_deterministic_id(text, source="doc.md", chunk_index=1)
        id3 = generate_deterministic_id(text, source="other.md", chunk_index=0)

        # Same source, different chunk
        assert id1 != id2
        # Different source, same chunk
        assert id1 != id3
        # Different in both
        assert id2 != id3

    def test_returns_sha256_hex(self):
        """ID should be a valid SHA-256 hex digest (64 characters)"""
        text = "Test"
        id_ = generate_deterministic_id(text)
        assert len(id_) == 64
        assert all(c in "0123456789abcdef" for c in id_)


class TestGenerateDeterministicIds:
    def test_multiple_texts_without_metadata(self):
        """Generate IDs for multiple texts without metadata"""
        texts = ["First document", "Second document", "Third document"]
        ids = generate_deterministic_ids(texts)

        assert len(ids) == 3
        # All IDs should be unique due to chunk_index
        assert len(set(ids)) == 3
        # Should be valid SHA-256 hashes
        assert all(len(id_) == 64 for id_ in ids)

    def test_multiple_texts_with_metadata(self):
        """Generate IDs for multiple texts with metadata"""
        texts = ["First chunk", "Second chunk"]
        metadatas = [{"source": "doc1.md"}, {"source": "doc2.md"}]
        ids = generate_deterministic_ids(texts, metadatas)

        assert len(ids) == 2
        assert ids[0] != ids[1]

    def test_same_text_different_sources(self):
        """Same text from different sources should have different IDs"""
        texts = ["Same content", "Same content"]
        metadatas = [{"source": "doc1.md"}, {"source": "doc2.md"}]
        ids = generate_deterministic_ids(texts, metadatas)

        # IDs should be different due to different sources
        assert ids[0] != ids[1]

    def test_metadata_without_source(self):
        """Handle metadata without source field"""
        texts = ["Text one", "Text two"]
        metadatas = [{"other_field": "value"}, {"another_field": "value"}]
        ids = generate_deterministic_ids(texts, metadatas)

        assert len(ids) == 2
        # Should still generate valid IDs using chunk_index
        assert ids[0] != ids[1]

    def test_empty_metadata(self):
        """Handle empty metadata dictionaries"""
        texts = ["Text one", "Text two"]
        metadatas = [{}, {}]
        ids = generate_deterministic_ids(texts, metadatas)

        assert len(ids) == 2
        # Should generate different IDs due to chunk_index
        assert ids[0] != ids[1]

    def test_partial_metadata(self):
        """Handle case where metadata list is shorter than texts list"""
        texts = ["Text one", "Text two", "Text three"]
        metadatas = [{"source": "doc.md"}]
        ids = generate_deterministic_ids(texts, metadatas)

        assert len(ids) == 3
        # All should be unique
        assert len(set(ids)) == 3


class TestDeduplication:
    def test_duplicate_detection(self):
        """Test that duplicate content produces the same ID for deduplication"""
        # Same content, same source
        text = "This is a duplicate document."
        source = "test.md"
        chunk_idx = 0

        id1 = generate_deterministic_id(text, source=source, chunk_index=chunk_idx)
        id2 = generate_deterministic_id(text, source=source, chunk_index=chunk_idx)

        # Should be identical for deduplication via upsert
        assert id1 == id2

    def test_content_variation_detection(self):
        """Test that minor content variations are NOT treated as duplicates"""
        text1 = "This is version 1."
        text2 = "This is version 2."

        id1 = generate_deterministic_id(text1, source="doc.md", chunk_index=0)
        id2 = generate_deterministic_id(text2, source="doc.md", chunk_index=0)

        # Different content should have different IDs
        assert id1 != id2
