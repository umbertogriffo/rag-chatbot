from pydantic import BaseModel


class DocumentInfo(BaseModel):
    source: str
    chunk_count: int | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class DocumentUploadResponse(BaseModel):
    message: str
    source: str
    chunks_created: int
