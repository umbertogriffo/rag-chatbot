import shutil
import uuid
from pathlib import Path
from typing import Annotated

from bot.memory.document_registry import DocumentRegistry
from bot.memory.vector_database.id_generator import compute_version_hash
from core.config import settings
from entities.document import Document
from fastapi import APIRouter, File, HTTPException, UploadFile
from helpers.log import get_logger
from memory_builder import split_chunks
from schemas.documents import DocumentInfo, DocumentListResponse, DocumentUploadResponse

from api.deps import SessionDep, VectorDatabaseDep

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=201,
    responses={
        400: {"description": "Bad Request - Invalid file type."},
        409: {"description": "Conflict - Document with the same filename already exists."},
    },
)
async def upload_document(
    file: Annotated[UploadFile, File(...)],
    index: VectorDatabaseDep,
    session: SessionDep,
):
    """
    Upload a document to the knowledge base.

    Args:
        file: The file to upload. Must have an allowed extension.
        index: Vector database dependency for storing document chunks.
        session: Database session dependency for the document registry.

    Returns:
        DocumentUploadResponse containing the generated document_id and filename.

    Raises:
        HTTPException: 400 if file type is not supported.
        HTTPException: 409 if a document with the same filename already exists.
    """
    registry = DocumentRegistry(session)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{suffix}' not supported. Allowed: {sorted(settings.ALLOWED_UPLOAD_EXTENSIONS)}",
        )

    existing = registry.get_by_filename(file.filename or "")
    if existing is not None:
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

    version_hash = compute_version_hash(page_content)

    document = Document(
        page_content=page_content,
        metadata={
            "source": str(file_path),
            "document_id": document_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "version_hash": version_hash,
        },
    )
    chunks = split_chunks([document], chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)

    # Inject document_id + version_hash into every chunk's metadata
    for chunk in chunks:
        chunk.metadata["document_id"] = document_id
        chunk.metadata["version_hash"] = version_hash

    num_chunks = len(chunks)
    logger.info(f"Number of generated chunks: {num_chunks}")
    logger.info("Adding document chunks to the vector database index...")

    chunk_ids = index.from_chunks(chunks)

    registry.upsert(
        document_id,
        source=str(file_path),
        filename=file.filename or document_id,
        size=len(content),
        content_type=file.content_type or "application/octet-stream",
        version_hash=version_hash,
        chunk_ids=chunk_ids,
    )

    logger.info("Memory Index has been updated successfully!")

    return DocumentUploadResponse(document_id=document_id, filename=file.filename or document_id)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(session: SessionDep):
    """
    List all uploaded documents.

    Returns:
        DocumentListResponse containing a list of all document metadata.
    """
    registry = DocumentRegistry(session)
    records = registry.get_all()
    documents = [
        DocumentInfo(
            document_id=rec.document_id,
            filename=rec.filename,
            size=rec.size,
            content_type=rec.content_type,
            version_hash=rec.version_hash,
        )
        for rec in records
    ]
    return DocumentListResponse(documents=documents)


@router.delete(
    "/documents/{document_id}",
    status_code=204,
    responses={404: {"description": "Not Found - Document with the given ID does not exist."}},
)
async def delete_document(document_id: str, index: VectorDatabaseDep, session: SessionDep):
    """
    Delete a document from the knowledge base.

    Removes the document's metadata, associated file from disk, and its
    chunks from the vector database index.

    Args:
        document_id: The unique identifier of the document to delete.
        index: Vector database dependency for removing document chunks.
        session: Database session dependency for the document registry.

    Raises:
        HTTPException: 404 if the document with the given ID is not found.
    """
    registry = DocumentRegistry(session)
    rec = registry.get(document_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    index.delete_chunks_by_document_id(document_id, chunk_ids=rec.chunk_ids or None)
    registry.remove(document_id)

    dest_dir = settings.DOCS_PATH / document_id
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
