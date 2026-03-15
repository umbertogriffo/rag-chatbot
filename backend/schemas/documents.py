from pydantic import BaseModel


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    size: int
    content_type: str


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
