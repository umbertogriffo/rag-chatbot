import shutil
import uuid
from pathlib import Path
from typing import Annotated

from core.config import settings
from entities.document import Document
from fastapi import APIRouter, File, HTTPException, UploadFile
from helpers.log import get_logger
from memory_builder import split_chunks
from schemas.documents import DocumentInfo, DocumentListResponse, DocumentUploadResponse

from api.deps import VectorDatabaseDep

logger = get_logger(__name__)

router = APIRouter()

# In-memory store of document metadata; keyed by document_id.
_documents: dict[str, DocumentInfo] = {}


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=201,
    responses={
        400: {"description": "Bad Request - Invalid file type."},
        409: {"description": "Conflict - Document with the same filename already exists."},
    },
)
async def upload_document(file: Annotated[UploadFile, File(...)], index: VectorDatabaseDep):
    """
    Upload a document to the knowledge base.

    Args:
        file: The file to upload. Must have an allowed extension.
        index: Vector database dependency for storing document chunks.

    Returns:
        DocumentUploadResponse containing the generated document_id and filename.

    Raises:
        HTTPException: 400 if file type is not supported.
        HTTPException: 409 if a document with the same filename already exists.
    """
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

    try:
        page_content = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        logger.warning(
            "Failed to decode uploaded file '%s' as UTF-8: %s",
            file.filename,
            exc,
        )
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not valid UTF-8 text and cannot be processed.",
        )

    document = Document(
        page_content=page_content,
        metadata={
            "source": str(file_path),
            "document_id": document_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
        },
    )
    chunks = split_chunks([document], chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    num_chunks = len(chunks)

    logger.info(f"Number of generated chunks: {num_chunks}")
    logger.info("Adding document chunks to the vector database index...")

    index.from_chunks(chunks)

    logger.info("Memory Index has been updated successfully!")

    return DocumentUploadResponse(document_id=document_id, filename=doc_info.filename)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    List all uploaded documents.

    Returns:
        DocumentListResponse containing a list of all document metadata.
    """
    return DocumentListResponse(documents=list(_documents.values()))


@router.delete(
    "/documents/{document_id}",
    status_code=204,
    responses={404: {"description": "Not Found - Document with the given ID does not exist."}},
)
async def delete_document(document_id: str, index: VectorDatabaseDep):
    """
    Delete a document from the knowledge base.

    Removes the document's metadata, associated file from disk, and should remove
    chunks from the vector database index (currently not fully implemented).

    Args:
        document_id: The unique identifier of the document to delete.
        index: Vector database dependency for removing document chunks.

    Raises:
        HTTPException: 404 if the document with the given ID is not found.
    """
    if document_id not in _documents:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    dest_dir = settings.DOCS_PATH / document_id
    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    # TODO: implement an efficient way to remove all chunks associated with this document from the vector database index
    # https://github.com/umbertogriffo/rag-chatbot/pull/10#discussion_r2936567674

    del _documents[document_id]
