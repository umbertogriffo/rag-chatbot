from pathlib import Path
from typing import Annotated

from bot.memory.document_registry import DocumentRegistry
from bot.memory.vector_database.id_generator import generate_id
from core.config import settings
from document_loader.loader import DirectoryLoader
from fastapi import APIRouter, File, HTTPException, UploadFile
from helpers.log import get_logger
from memory_builder import split_chunks
from schemas.documents import DocumentInfo, DocumentListResponse, DocumentUploadResponse

from api.deps import SessionDep, VectorDatabaseDep

logger = get_logger(__name__)

router = APIRouter()

# In-memory store of the uploaded document metadata, keyed by document_id.
_uploaded_documents: dict[str, DocumentInfo] = {}


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

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{suffix}' not supported. Allowed: {sorted(settings.ALLOWED_UPLOAD_EXTENSIONS)}",
        )

    registry = DocumentRegistry(session)
    existing = registry.get_by_filename(file.filename or "")
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Document '{file.filename}' already exists.",
        )

    dest_dir = settings.DOCS_PATH
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / file.filename
    document_id = generate_id(str(file_path))

    content = await file.read()
    file_path.write_bytes(content)

    # Use DirectoryLoader to load the file content (same as build_memory_index)
    # This ensures consistent content processing and version hashing
    try:
        loader = DirectoryLoader(
            path=file_path.parent,
            glob=file_path.name,
            show_progress=False,
        )
        loaded_docs = loader.load()

        if not loaded_docs:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load document '{file.filename}'. The file may be corrupted or in an"
                f" unsupported format.",
            )

        # Extract the loaded document (should be exactly one)
        document = loaded_docs[0]
        page_content = document.page_content

    except Exception as exc:
        logger.warning(
            f"Failed to load uploaded file '{file.filename}': {exc}",
        )
        # Clean up the saved file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process document '{file.filename}': {str(exc)}",
        )

    version_hash = generate_id(page_content)

    # Update document metadata with our tracking fields
    document.metadata.update(
        {
            "source": str(file_path),
            "document_id": document_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "version_hash": version_hash,
        }
    )

    # Store the document metadata in the in-memory registry for listing purposes
    doc_info = DocumentInfo(
        document_id=document_id,
        filename=file.filename or document_id,
        size=len(content),
        content_type=file.content_type or "application/octet-stream",
    )
    _uploaded_documents[document_id] = doc_info

    # Split the document into chunks for vector indexing
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
    return DocumentListResponse(documents=list(_uploaded_documents.values()))


@router.delete(
    "/documents/{document_id}",
    status_code=204,
    responses={404: {"description": "Not Found - Document with the given ID does not exist."}},
)
async def delete_document(document_id: str, index: VectorDatabaseDep, session: SessionDep):
    """
    Delete the uploaded document from the knowledge base.

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
    entry = registry.get(document_id)

    if entry is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    index.delete_chunks_by_document_id(document_id, chunk_ids=entry.chunk_ids or None)
    registry.remove(document_id)

    file_path = settings.DOCS_PATH / entry.filename
    if file_path.exists():
        file_path.unlink()
