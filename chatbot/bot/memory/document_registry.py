"""
SQLModel-backed document registry for tracking ingested documents and their chunks.
"""

import json
import logging
from pathlib import Path

from sqlmodel import Field, Session, SQLModel, create_engine, select

logger = logging.getLogger(__name__)


def _create_sqlite_engine(db_path: Path):
    """Create a SQLite engine with WAL journal mode for concurrent-read safety."""
    from sqlalchemy import event

    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.close()

    return engine


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

    def __init__(self, db_path: str | Path) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = _create_sqlite_engine(db_path)
        SQLModel.metadata.create_all(self._engine)

    @property
    def engine(self):
        return self._engine

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def get_all(self) -> list[DocumentRecord]:
        """Return every document record."""
        with Session(self._engine) as session:
            return list(session.exec(select(DocumentRecord)).all())

    def get(self, document_id: str) -> DocumentRecord | None:
        """Return a single record by its *document_id*, or ``None``."""
        with Session(self._engine) as session:
            return session.get(DocumentRecord, document_id)

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
        with Session(self._engine) as session:
            existing = session.get(DocumentRecord, document_id)
            if existing:
                existing.source = source
                existing.filename = filename
                existing.size = size
                existing.content_type = content_type
                existing.version_hash = version_hash
                existing.chunk_ids = chunk_ids or []
                session.add(existing)
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
                session.add(record)
            session.commit()

    def remove(self, document_id: str) -> None:
        """Delete a document record."""
        with Session(self._engine) as session:
            record = session.get(DocumentRecord, document_id)
            if record:
                session.delete(record)
                session.commit()

    def get_by_filename(self, filename: str) -> DocumentRecord | None:
        """Look up a document by its filename."""
        with Session(self._engine) as session:
            return session.exec(select(DocumentRecord).where(DocumentRecord.filename == filename)).first()

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

    def close(self) -> None:
        """Dispose of the underlying engine."""
        self._engine.dispose()
