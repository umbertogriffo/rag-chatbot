from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field


@dataclass
class Document:
    """Class for storing a piece of text and associated metadata."""

    page_content: str
    """String text."""
    metadata: dict = Field(default_factory=dict)
    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """
    type: Literal["Document"] = "Document"
