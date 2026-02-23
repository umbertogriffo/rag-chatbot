"""Tests for the document management API endpoints."""

from __future__ import annotations

import io

import pytest

pytestmark = pytest.mark.asyncio


class TestUploadDocument:
    async def test_upload_valid_markdown(self, async_client, tmp_path):
        content = b"# Hello\nThis is a test document."
        response = await async_client.post(
            "/api/documents",
            files={"file": ("readme.md", io.BytesIO(content), "text/markdown")},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["filename"] == "readme.md"
        assert "document_id" in body

    async def test_upload_valid_txt(self, async_client):
        content = b"Plain text content."
        response = await async_client.post(
            "/api/documents",
            files={"file": ("notes.txt", io.BytesIO(content), "text/plain")},
        )
        assert response.status_code == 201

    async def test_upload_unsupported_extension_returns_400(self, async_client):
        content = b"some data"
        response = await async_client.post(
            "/api/documents",
            files={"file": ("script.py", io.BytesIO(content), "text/plain")},
        )
        assert response.status_code == 400
        assert "not supported" in response.json()["detail"].lower()

    async def test_upload_duplicate_filename_returns_409(self, async_client):
        content = b"# Doc"
        files = {"file": ("dup.md", io.BytesIO(content), "text/markdown")}
        await async_client.post("/api/documents", files=files)

        files = {"file": ("dup.md", io.BytesIO(content), "text/markdown")}
        response = await async_client.post("/api/documents", files=files)
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"].lower()


class TestListDocuments:
    async def test_list_returns_empty_initially(self, async_client):
        response = await async_client.get("/api/documents")
        assert response.status_code == 200
        assert response.json() == {"documents": []}

    async def test_list_returns_uploaded_documents(self, async_client):
        for name in ("a.md", "b.txt"):
            await async_client.post(
                "/api/documents",
                files={"file": (name, io.BytesIO(b"content"), "text/plain")},
            )

        response = await async_client.get("/api/documents")
        assert response.status_code == 200
        docs = response.json()["documents"]
        assert len(docs) == 2
        filenames = {d["filename"] for d in docs}
        assert filenames == {"a.md", "b.txt"}

    async def test_list_document_has_expected_fields(self, async_client):
        content = b"Hello"
        await async_client.post(
            "/api/documents",
            files={"file": ("check.md", io.BytesIO(content), "text/markdown")},
        )
        response = await async_client.get("/api/documents")
        doc = response.json()["documents"][0]
        assert "document_id" in doc
        assert doc["filename"] == "check.md"
        assert doc["size"] == len(content)
        assert "content_type" in doc


class TestDeleteDocument:
    async def test_delete_existing_document_returns_204(self, async_client):
        upload = await async_client.post(
            "/api/documents",
            files={"file": ("to_delete.md", io.BytesIO(b"bye"), "text/markdown")},
        )
        doc_id = upload.json()["document_id"]

        response = await async_client.delete(f"/api/documents/{doc_id}")
        assert response.status_code == 204

    async def test_delete_removes_document_from_list(self, async_client):
        upload = await async_client.post(
            "/api/documents",
            files={"file": ("gone.md", io.BytesIO(b"bye"), "text/markdown")},
        )
        doc_id = upload.json()["document_id"]

        await async_client.delete(f"/api/documents/{doc_id}")

        response = await async_client.get("/api/documents")
        assert response.json() == {"documents": []}

    async def test_delete_nonexistent_document_returns_404(self, async_client):
        response = await async_client.delete("/api/documents/nonexistent-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
