from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Document:
    """Class for storing a piece of text and associated metadata."""

    page_content: str
    """String text."""
    metadata: dict = field(default_factory=dict)
    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """
    type: Literal["Document"] = "Document"
