"""
SQLite-backed document registry for tracking ingested documents and their chunks.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """A single row in the document registry."""

    document_id: str
    source: str = ""
    filename: str = ""
    size: int = 0
    content_type: str = ""
    version_hash: str = ""
    chunk_ids: list[str] = field(default_factory=list)


class DocumentRegistry:
    """
    Persistent registry backed by a SQLite database.

    Stores metadata about every ingested document so that the ingestion
    pipeline can compute incremental diffs (new / changed / deleted).

    The database is opened in WAL mode for concurrent-read safety.
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS documents (
        document_id TEXT PRIMARY KEY,
        source      TEXT NOT NULL DEFAULT '',
        filename    TEXT NOT NULL DEFAULT '',
        size        INTEGER NOT NULL DEFAULT 0,
        content_type TEXT NOT NULL DEFAULT '',
        version_hash TEXT NOT NULL DEFAULT '',
        chunk_ids   TEXT NOT NULL DEFAULT '[]'
    );
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(self._DDL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            source=row["source"],
            filename=row["filename"],
            size=row["size"],
            content_type=row["content_type"],
            version_hash=row["version_hash"],
            chunk_ids=json.loads(row["chunk_ids"]),
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def get_all(self) -> list[DocumentRecord]:
        """Return every document record."""
        rows = self._conn.execute("SELECT * FROM documents").fetchall()
        return [self._row_to_record(r) for r in rows]

    def get(self, document_id: str) -> DocumentRecord | None:
        """Return a single record by its *document_id*, or ``None``."""
        row = self._conn.execute(
            "SELECT * FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return self._row_to_record(row) if row else None

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
        self._conn.execute(
            """
            INSERT INTO documents (document_id, source, filename, size, content_type, version_hash, chunk_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                source       = excluded.source,
                filename     = excluded.filename,
                size         = excluded.size,
                content_type = excluded.content_type,
                version_hash = excluded.version_hash,
                chunk_ids    = excluded.chunk_ids
            """,
            (
                document_id,
                source,
                filename,
                size,
                content_type,
                version_hash,
                json.dumps(chunk_ids or []),
            ),
        )
        self._conn.commit()

    def remove(self, document_id: str) -> None:
        """Delete a document record."""
        self._conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        self._conn.commit()

    def get_by_filename(self, filename: str) -> DocumentRecord | None:
        """Look up a document by its filename."""
        row = self._conn.execute(
            "SELECT * FROM documents WHERE filename = ?",
            (filename,),
        ).fetchone()
        return self._row_to_record(row) if row else None

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
        changed_ids = {
            doc_id
            for doc_id in current_ids & stored_ids
            if current_docs[doc_id] != stored[doc_id]
        }

        return new_ids, changed_ids, deleted_ids

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
