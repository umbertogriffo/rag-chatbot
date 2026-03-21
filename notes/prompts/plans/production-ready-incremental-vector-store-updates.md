# Plan: Production-Ready Incremental Vector Store Updates

Replace the full-rebuild ingestion and in-memory _documents dict with a SQLite-backed DocumentRegistry,
document-level version tracking on every chunk, targeted Chroma deletions, an incremental diff pipeline,
and an admin reindex endpoint with concurrency guard and status polling.

Steps

1. Add compute_version_hash to `id_generator.py`. Add a function that SHA-256 hashes the full document content (reusing normalize_text) to produce a version fingerprint. This represents the whole source document, not individual chunks. Add corresponding tests in `test_id_generator.py`.
2. Create SQLite-backed `DocumentRegistry` in a new file `chatbot/bot/memory/document_registry.py`. Single table `documents(document_id TEXT PK, source TEXT, filename TEXT, size INTEGER, content_type TEXT, version_hash TEXT, chunk_ids TEXT)` where `chunk_ids` is a JSON-serialized list.
   1. Methods: `get_all()`, `get(doc_id)`, `upsert(doc_id, ...)`, `remove(doc_id)`, `get_by_filename(filename)`, `get_stale_documents(current_docs: dict[str, str]) → (new, changed, deleted)` — the diff helper comparing a {doc_id: version_hash} snapshot against what's stored. Add tests in a new `tests/bot/memory/test_document_registry.py`.
3. Add `delete_chunks_by_document_id` to `Chroma` and make `from_chunks` / `from_texts` return `list[str]`. The delete method accepts `document_id` and optional `chunk_ids`: `list[str]`; when `chunk_ids` is provided uses `self.collection.delete(ids=chunk_ids) (fast, precise)`, otherwise falls back to `self.collection.delete(where={"document_id": document_id})`. Changing from_chunks and from_texts from returning None to list[str] lets callers capture generated chunk IDs for the registry. Add tests in `test_chroma.py`.
4. Enrich chunk metadata with `document_id` and `version_hash`. In `memory_builder.py`, after loading each source doc, derive document_id (SHA-256 of source path) and compute version_hash from content, then inject both into every chunk's metadata after split_chunks. In the upload endpoint, compute version_hash from file content and ensure it (alongside the existing document_id) propagates into every chunk's metadata.
5. Refactor `build_memory_index` into an incremental pipeline in `memory_builder.py`.
   1. New flow: (a) load source docs, compute each doc's document_id + version_hash
   2. (b) call registry.get_stale_documents(...) to find new/changed/deleted sets
   3. (c) for changed/deleted docs call chroma.delete_chunks_by_document_id(doc_id, chunk_ids) using IDs from the registry
   4. (d) chunk & ingest only new/changed docs
   5. (e) upsert results (version_hash + chunk_ids) into registry.
   6. Add a full_rebuild: bool parameter that wipes the collection + registry first. Keep the --full-rebuild CLI flag in get_args for direct script invocation.
6. Adapt `documents.py` to use `DocumentRegistry`.
   1. Remove `_documents dict`.
   2. `upload_document`: compute version_hash, call index.from_chunks → capture returned chunk_ids, call registry.upsert(...).
   3. list_documents: call registry.get_all().
   4. delete_document: look up in registry, call index.
   5. delete_chunks_by_document_id(doc_id, chunk_ids), call registry.remove(doc_id), delete file from disk.
   6. Add version_hash: str field to DocumentInfo in documents schema.


--- Key Benefits

- **Document-level metadata tracking**: every chunk gets tagged with a source doc ID + version hash. When a doc changes, we regenerate chunks for that doc only, delete the old ones by metadata filter, and insert new ones. way cheaper than rebuilding the whole index.
- **Incremental ingestion pipeline**: we run a job that diffs source docs against what's already indexed (using those version hashes). Only changed/new docs get processed. Keeps compute costs reasonable as the corpus grows.
- **Handling deletions**:We keep a separate mapping table (doc_id → chunk_ids) so we can precisely target what to remove without scanning the whole store.

> One thing to watch out for — if you ever swap embedding models, you must rebuild it from scratch since the vector spaces won’t be compatible. Plan for that early.
