import pytest
from bot.memory.document_registry import DocumentRecord, DocumentRegistry


@pytest.fixture
def registry(session):
    """Provide a fresh DocumentRegistry backed by a temp SQLite DB."""
    reg = DocumentRegistry(session)
    yield reg


class TestUpsertAndGet:
    def test_upsert_and_get(self, registry):
        registry.upsert(
            "doc-1",
            source="/docs/file.md",
            filename="file.md",
            size=100,
            content_type="text/markdown",
            version_hash="abc123",
            chunk_ids=["c1", "c2"],
        )
        rec = registry.get("doc-1")
        assert rec is not None
        assert rec.document_id == "doc-1"
        assert rec.source == "/docs/file.md"
        assert rec.filename == "file.md"
        assert rec.size == 100
        assert rec.content_type == "text/markdown"
        assert rec.version_hash == "abc123"
        assert rec.chunk_ids == ["c1", "c2"]

    def test_upsert_overwrites(self, registry):
        registry.upsert("doc-1", version_hash="v1", chunk_ids=["a"])
        registry.upsert("doc-1", version_hash="v2", chunk_ids=["b", "c"])
        rec = registry.get("doc-1")
        assert rec.version_hash == "v2"
        assert rec.chunk_ids == ["b", "c"]

    def test_get_missing_returns_none(self, registry):
        assert registry.get("nonexistent") is None


class TestGetAll:
    def test_empty(self, registry):
        assert registry.get_all() == []

    def test_multiple_records(self, registry):
        registry.upsert("doc-1", filename="a.md")
        registry.upsert("doc-2", filename="b.md")
        records = registry.get_all()
        assert len(records) == 2
        ids = {r.document_id for r in records}
        assert ids == {"doc-1", "doc-2"}


class TestRemove:
    def test_remove_existing(self, registry):
        registry.upsert("doc-1", filename="a.md")
        registry.remove("doc-1")
        assert registry.get("doc-1") is None

    def test_remove_nonexistent(self, registry):
        # Should not raise
        registry.remove("nonexistent")


class TestGetByFilename:
    def test_found(self, registry):
        registry.upsert("doc-1", filename="readme.md")
        rec = registry.get_by_filename("readme.md")
        assert rec is not None
        assert rec.document_id == "doc-1"

    def test_not_found(self, registry):
        assert registry.get_by_filename("missing.md") is None


class TestGetStaleDocuments:
    def test_all_new(self, registry):
        current = {"d1": "hash1", "d2": "hash2"}
        new, changed, deleted = registry.get_stale_documents(current)
        assert new == {"d1", "d2"}
        assert changed == set()
        assert deleted == set()

    def test_all_deleted(self, registry):
        registry.upsert("d1", version_hash="hash1")
        registry.upsert("d2", version_hash="hash2")
        new, changed, deleted = registry.get_stale_documents({})
        assert new == set()
        assert changed == set()
        assert deleted == {"d1", "d2"}

    def test_changed(self, registry):
        registry.upsert("d1", version_hash="old_hash")
        current = {"d1": "new_hash"}
        new, changed, deleted = registry.get_stale_documents(current)
        assert new == set()
        assert changed == {"d1"}
        assert deleted == set()

    def test_mixed(self, registry):
        registry.upsert("existing", version_hash="same")
        registry.upsert("changed", version_hash="old")
        registry.upsert("removed", version_hash="gone")
        current = {"existing": "same", "changed": "new", "brand_new": "fresh"}
        new, changed, deleted = registry.get_stale_documents(current)
        assert new == {"brand_new"}
        assert changed == {"changed"}
        assert deleted == {"removed"}

    def test_unchanged(self, registry):
        registry.upsert("d1", version_hash="h1")
        new, changed, deleted = registry.get_stale_documents({"d1": "h1"})
        assert new == set()
        assert changed == set()
        assert deleted == set()


class TestDocumentRecord:
    def test_defaults(self):
        rec = DocumentRecord(document_id="x")
        assert rec.source == ""
        assert rec.filename == ""
        assert rec.size == 0
        assert rec.chunk_ids == []
