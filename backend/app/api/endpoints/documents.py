import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from backend.app.core.config import settings
from backend.app.core.security import get_current_user
from backend.app.schemas.documents import DocumentInfo, DocumentListResponse, DocumentUploadResponse

router = APIRouter()


@router.get("/", response_model=DocumentListResponse)
async def list_documents(_current_user: str = Depends(get_current_user)):
    """List all indexed documents in the vector store."""
    try:
        from chatbot.bot.memory.embedder import Embedder
        from chatbot.bot.memory.vector_database.chroma import Chroma

        embedder = Embedder()
        vector_db = Chroma(
            embedding=embedder,
            persist_directory=str(settings.VECTOR_STORE_PATH),
            is_persistent=True,
        )
        indexed_docs = vector_db.get_indexed_documents()
        documents = [DocumentInfo(source=doc) for doc in indexed_docs]
        return DocumentListResponse(documents=documents, total=len(documents))
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Failed to list documents: %s", e)
        return DocumentListResponse(documents=[], total=0)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile,
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
    _current_user: str = Depends(get_current_user),
):
    """Upload a document and index it in the vector store."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    allowed_extensions = {".md", ".txt", ".pdf"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_extensions)}",
        )

    # Save uploaded file to docs directory
    docs_path = settings.DOCS_PATH
    os.makedirs(docs_path, exist_ok=True)
    file_path = docs_path / file.filename

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        from chatbot.document_loader.text_splitter import RecursiveCharacterTextSplitter
        from chatbot.entities.document import Document

        text_content = content.decode("utf-8")
        doc = Document(page_content=text_content, metadata={"source": file.filename})

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=True,
        )
        chunks = splitter.split_documents([doc])

        from chatbot.bot.memory.embedder import Embedder
        from chatbot.bot.memory.vector_database.chroma import Chroma

        embedder = Embedder()
        vector_db = Chroma(
            embedding=embedder,
            persist_directory=str(settings.VECTOR_STORE_PATH),
            is_persistent=True,
        )
        vector_db.from_chunks(chunks)

        return DocumentUploadResponse(
            message="Document uploaded and indexed successfully",
            source=file.filename,
            chunks_created=len(chunks),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
