from bot.memory.vector_database.id_generator import (
    generate_deterministic_id,
    generate_deterministic_ids,
    normalize_text,
)


class TestNormalizeText:
    def test_lowercase_conversion(self):
        """Test that text is converted to lowercase"""
        assert normalize_text("HELLO WORLD") == "hello world"
        assert normalize_text("HeLLo WoRLd") == "hello world"

    def test_whitespace_collapse(self):
        """Test that multiple whitespaces are collapsed to a single space"""
        assert normalize_text("hello  world") == "hello world"
        assert normalize_text("hello\n\nworld") == "hello world"
        assert normalize_text("hello\t\tworld") == "hello world"
        assert normalize_text("  hello   world  ") == "hello world"

    def test_unicode_normalization(self):
        """Test that unicode characters are normalized to NFD form"""

        text1 = "caf\u00e9"  # é as single character code
        text2 = "cafe\u0301"  # é as two characters code that combine visually

        assert text1 != text2
        assert normalize_text(text1) == normalize_text(text2)

    def test_strip_leading_trailing(self):
        """Test that leading and trailing whitespace is stripped"""
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

    def test_returns_sha256_hex(self):
        """ID should be a valid SHA-256 hex digest (64 characters)"""
        text = "Test"
        id_ = generate_deterministic_id(text)
        assert len(id_) == 64
        assert all(c in "0123456789abcdef" for c in id_)


class TestGenerateDeterministicIds:
    def test_multiple_texts(self):
        """Generate IDs for multiple texts without metadata"""
        texts = ["First document", "Second document", "Third document"]
        ids = generate_deterministic_ids(texts)

        assert len(ids) == 3
        # All IDs should be unique due to chunk_index
        assert len(set(ids)) == 3
        # Should be valid SHA-256 hashes
        assert all(len(id_) == 64 for id_ in ids)


class TestDeduplication:
    def test_duplicate_detection(self):
        """Test that duplicate content produces the same ID for deduplication"""
        # Same content, same source
        text = "This is a duplicate document."

        id1 = generate_deterministic_id(text)
        id2 = generate_deterministic_id(text)

        # Should be identical for deduplication via upsert
        assert id1 == id2

    def test_content_variation_detection(self):
        """Test that minor content variations are NOT treated as duplicates"""
        text1 = "This is version 1."
        text2 = "This is version 2."

        id1 = generate_deterministic_id(text1)
        id2 = generate_deterministic_id(text2)

        # Different content should have different IDs
        assert id1 != id2
