import json

from sqlmodel import Field, SQLModel


class DocumentRecord(SQLModel, table=True):
    """A single row in the document registry."""

    __tablename__ = "documents"

    document_id: str = Field(primary_key=True)
    source: str = Field(default="")
    filename: str = Field(default="")
    size: int = Field(default=0)
    content_type: str = Field(default="")
    version_hash: str = Field(default="")
    chunk_ids_json: str = Field(default="[]", sa_column_kwargs={"name": "chunk_ids"})

    @property
    def chunk_ids(self) -> list[str]:
        """Deserialize the stored JSON string into a Python list."""
        return json.loads(self.chunk_ids_json)

    @chunk_ids.setter
    def chunk_ids(self, value: list[str]) -> None:
        """Serialize a Python list into a JSON string for storage."""
        self.chunk_ids_json = json.dumps(value)
