"""
Admin endpoints for triggering and monitoring background re-index operations.
"""

import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from core.config import settings
from fastapi import APIRouter, BackgroundTasks, HTTPException
from helpers.log import get_logger
from memory_builder import build_memory_index
from schemas.admin import ReindexResponse, ReindexStatusResponse

logger = get_logger(__name__)

router = APIRouter()

# ── concurrency guard and shared state ───────────────────────────────────
_reindex_lock = threading.Lock()


@dataclass
class _ReindexState:
    status: str = "idle"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    docs_processed: int = 0
    docs_deleted: int = 0
    docs_skipped: int = 0


_state = _ReindexState()


def _run_reindex(full_rebuild: bool, vector_store_path: str, docs_path: str, chunk_size: int, chunk_overlap: int):
    """
    Synchronous function executed in a background thread via BackgroundTasks.

    Calls the incremental (or full-rebuild) memory builder and updates the
    shared ``_state`` accordingly.
    """
    global _state
    try:
        stats = build_memory_index(
            docs_path=docs_path,
            vector_store_path=vector_store_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            full_rebuild=full_rebuild,
        )
        _state.docs_processed = stats.get("processed", 0)
        _state.docs_deleted = stats.get("deleted", 0)
        _state.docs_skipped = stats.get("skipped", 0)
        _state.status = "completed"
        _state.finished_at = datetime.now(timezone.utc)
        logger.info("Reindex completed: %s", stats)
    except Exception as exc:
        _state.status = "failed"
        _state.error = str(exc)
        _state.finished_at = datetime.now(timezone.utc)
        logger.error("Reindex failed: %s", exc, exc_info=True)
    finally:
        _reindex_lock.release()


@router.post(
    "/admin/reindex",
    response_model=ReindexResponse,
    status_code=202,
    responses={409: {"description": "Conflict - A reindex operation is already running."}},
)
async def reindex(
    background_tasks: BackgroundTasks,
    full_rebuild: bool = False,
):
    """
    Trigger a background re-index of all source documents.

    * If *full_rebuild* is ``True`` the vector store and registry are wiped
      first.
    * Returns **202 Accepted** if the task was scheduled, or **409 Conflict**
      if one is already running.  Poll ``GET /admin/reindex/status`` for
      progress.
    """
    global _state

    if not _reindex_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A reindex operation is already in progress.")

    _state = _ReindexState(status="running", started_at=datetime.now(timezone.utc))

    background_tasks.add_task(
        _run_reindex,
        full_rebuild=full_rebuild,
        vector_store_path=str(settings.VECTOR_STORE_PATH),
        docs_path=str(settings.DOCS_PATH),
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    return ReindexResponse(message="Reindex started.", status="running")


@router.get("/admin/reindex/status", response_model=ReindexStatusResponse)
async def reindex_status():
    """Return the current state of the last reindex operation."""
    return ReindexStatusResponse(
        status=_state.status,
        started_at=_state.started_at,
        finished_at=_state.finished_at,
        error=_state.error,
        docs_processed=_state.docs_processed,
        docs_deleted=_state.docs_deleted,
        docs_skipped=_state.docs_skipped,
    )
