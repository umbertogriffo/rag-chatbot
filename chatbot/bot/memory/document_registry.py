"""
SQLModel-backed document registry for tracking ingested documents and their chunks.
"""

import json
import logging

from sqlmodel import Field, Session, SQLModel, select

logger = logging.getLogger(__name__)


class DocumentRecord(SQLModel, table=True):
    """A single row in the document registry."""

    __tablename__ = "documents"

    document_id: str = Field(primary_key=True)
    source: str = Field(default="")
    filename: str = Field(default="")
    size: int = Field(default=0)
    content_type: str = Field(default="")
    version_hash: str = Field(default="")
    chunk_ids_json: str = Field(default="[]", sa_column_kwargs={"name": "chunk_ids"})

    @property
    def chunk_ids(self) -> list[str]:
        """Deserialize the stored JSON string into a Python list."""
        return json.loads(self.chunk_ids_json)

    @chunk_ids.setter
    def chunk_ids(self, value: list[str]) -> None:
        """Serialize a Python list into a JSON string for storage."""
        self.chunk_ids_json = json.dumps(value)


class DocumentRegistry:
    """
    Persistent registry backed by a SQLite database via SQLModel.

    Stores metadata about every ingested document so that the ingestion
    pipeline can compute incremental diffs (new / changed / deleted).
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    @property
    def session(self):
        return self._session

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def get_all(self) -> list[DocumentRecord]:
        """Return every document record."""
        return list(self._session.exec(select(DocumentRecord)).all())

    def get(self, document_id: str) -> DocumentRecord | None:
        """Return a single record by its *document_id*, or ``None``."""
        return self._session.get(DocumentRecord, document_id)

    def upsert(
        self,
        document_id: str,
        *,
        source: str = "",
        filename: str = "",
        size: int = 0,
        content_type: str = "",
        version_hash: str = "",
        chunk_ids: list[str] | None = None,
    ) -> None:
        """Insert or replace a document record."""

        existing = self._session.get(DocumentRecord, document_id)
        if existing:
            existing.source = source
            existing.filename = filename
            existing.size = size
            existing.content_type = content_type
            existing.version_hash = version_hash
            existing.chunk_ids = chunk_ids or []
            self._session.add(existing)
        else:
            record = DocumentRecord(
                document_id=document_id,
                source=source,
                filename=filename,
                size=size,
                content_type=content_type,
                version_hash=version_hash,
                chunk_ids_json=json.dumps(chunk_ids or []),
            )
            self._session.add(record)
        self._session.commit()

    def remove(self, document_id: str) -> None:
        """Delete a document record."""
        record = self._session.get(DocumentRecord, document_id)
        if record:
            self._session.delete(record)
            self._session.commit()

    def get_by_filename(self, filename: str) -> DocumentRecord | None:
        """Look up a document by its filename."""
        return self._session.exec(select(DocumentRecord).where(DocumentRecord.filename == filename)).first()

    def get_stale_documents(
        self,
        current_docs: dict[str, str],
    ) -> tuple[set[str], set[str], set[str]]:
        """
        Compare a snapshot of ``{document_id: version_hash}`` against the
        registry and return ``(new, changed, deleted)`` document-ID sets.

        * **new** — IDs present in *current_docs* but absent from the registry.
        * **changed** — IDs present in both but with a different *version_hash*.
        * **deleted** — IDs in the registry but absent from *current_docs*.
        """
        stored = {r.document_id: r.version_hash for r in self.get_all()}

        current_ids = set(current_docs.keys())
        stored_ids = set(stored.keys())

        new_ids = current_ids - stored_ids
        deleted_ids = stored_ids - current_ids
        changed_ids = {doc_id for doc_id in current_ids & stored_ids if current_docs[doc_id] != stored[doc_id]}

        return new_ids, changed_ids, deleted_ids
