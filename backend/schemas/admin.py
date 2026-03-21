from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ReindexResponse(BaseModel):
    """Returned by POST /admin/reindex."""

    message: str
    status: str


class ReindexStatusResponse(BaseModel):
    """Returned by GET /admin/reindex/status."""

    status: Literal["idle", "running", "failed", "completed"]
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    docs_processed: int = 0
    docs_deleted: int = 0
    docs_skipped: int = 0
