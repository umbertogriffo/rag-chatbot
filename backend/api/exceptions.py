"""Custom exceptions for the RAG Chatbot API.

This module defines a hierarchy of application-specific exceptions
to provide more precise error handling and better debugging capabilities.
"""


class AppException(Exception):
    """Base exception for all application-specific errors.

    Attributes:
        message: Human-readable error message
        status_code: Suggested HTTP status code for API responses
    """

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class LLMError(AppException):
    """Base exception for LLM-related errors."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, status_code)


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out."""

    def __init__(self, message: str = "LLM request timed out"):
        super().__init__(message, status_code=504)


class LLMConnectionError(LLMError):
    """Raised when unable to connect to LLM service."""

    def __init__(self, message: str = "Failed to connect to LLM service"):
        super().__init__(message, status_code=503)


class DocumentError(AppException):
    """Base exception for document processing errors."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message, status_code)


class DocumentLoadError(DocumentError):
    """Raised when document cannot be loaded or parsed."""

    def __init__(self, filename: str, reason: str = ""):
        message = f"Failed to load document '{filename}'"
        if reason:
            message += f": {reason}"
        super().__init__(message, status_code=400)


class DocumentNotFoundError(DocumentError):
    """Raised when requested document does not exist."""

    def __init__(self, document_id: str):
        super().__init__(
            f"Document '{document_id}' not found",
            status_code=404
        )


class VectorDBError(AppException):
    """Base exception for vector database errors."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, status_code)


class VectorDBConnectionError(VectorDBError):
    """Raised when unable to connect to vector database."""

    def __init__(self, message: str = "Failed to connect to vector database"):
        super().__init__(message, status_code=503)
