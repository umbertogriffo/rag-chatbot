import shutil
import uuid
from pathlib import Path
from typing import Annotated

from core.config import settings
from fastapi import APIRouter, File, HTTPException, UploadFile
from schemas.documents import DocumentInfo, DocumentListResponse, DocumentUploadResponse

router = APIRouter()

# In-memory store of document metadata; keyed by document_id.
_documents: dict[str, DocumentInfo] = {}


def get_documents_store() -> dict[str, DocumentInfo]:
    return _documents


@router.post("/documents", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(file: Annotated[UploadFile, File(...)]):
    """Upload a document to the knowledge base."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{suffix}' not supported. Allowed: {sorted(settings.ALLOWED_UPLOAD_EXTENSIONS)}",
        )

    for doc in _documents.values():
        if doc.filename == file.filename:
            raise HTTPException(
                status_code=409,
                detail=f"Document '{file.filename}' already exists.",
            )

    document_id = str(uuid.uuid4())
    dest_dir = settings.DOCS_PATH / document_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / (file.filename or document_id)

    content = await file.read()
    file_path.write_bytes(content)

    doc_info = DocumentInfo(
        document_id=document_id,
        filename=file.filename or document_id,
        size=len(content),
        content_type=file.content_type or "application/octet-stream",
    )
    _documents[document_id] = doc_info

    return DocumentUploadResponse(document_id=document_id, filename=doc_info.filename)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all uploaded documents."""
    return DocumentListResponse(documents=list(_documents.values()))


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(document_id: str):
    """Delete a document from the knowledge base."""
    if document_id not in _documents:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    dest_dir = settings.DOCS_PATH / document_id
    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    del _documents[document_id]
