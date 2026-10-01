"""Integration tests for Document Management API endpoints and Multi-Tenant Isolation (Phase 6)."""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from backend.app.main import app
from backend.app.db.session import get_db
from backend.app.core.config import get_settings


@pytest.fixture
def client(db_session: Session) -> TestClient:
    """Provides a TestClient with database session overridden to advanced_rag_test."""
    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def create_user_and_get_token(client: TestClient, email: str, password: str = "SecurePass123!") -> str:
    """Helper to register and login a user, returning their JWT access token."""
    client.post("/api/v1/auth/register", json={
        "email": email,
        "password": password,
        "full_name": f"User {email.split('@')[0]}",
    })
    login_res = client.post("/api/v1/auth/login", json={
        "email": email,
        "password": password,
    })
    return login_res.json()["access_token"]


class TestDocumentUploadAPI:
    """Test POST /api/v1/documents/upload endpoint."""

    def test_upload_valid_pdf_success(self, client: TestClient):
        token = create_user_and_get_token(client, "pdf_uploader@example.com")
        headers = {"Authorization": f"Bearer {token}"}

        pdf_content = b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF"
        files = {
            "file": ("machine_learning_paper.pdf", io.BytesIO(pdf_content), "application/pdf")
        }

        response = client.post("/api/v1/documents/upload", headers=headers, files=files)
        assert response.status_code == 201
        data = response.json()

        assert data["filename"] == "machine_learning_paper.pdf"
        assert data["file_type"] == ".pdf"
        assert data["file_size_bytes"] == len(pdf_content)
        assert data["status"] == "PENDING"
        assert data["page_count"] is None
        assert data["total_chunks"] == 0
        assert "id" in data
        
        # Security assertion: storage_path must NOT be exposed
        assert "storage_path" not in data

        # Verify physical file existence
        settings = get_settings()
        doc_id = data["id"]
        # Find directory matching doc_id under storage_dir
        matches = list(Path(settings.storage.upload_storage_dir).glob(f"*/{doc_id}/document.pdf"))
        assert len(matches) == 1
        assert matches[0].exists()
        assert matches[0].read_bytes() == pdf_content

    def test_upload_valid_docx_success(self, client: TestClient):
        token = create_user_and_get_token(client, "docx_uploader@example.com")
        headers = {"Authorization": f"Bearer {token}"}

        docx_content = b"PK\x03\x04\x14\x00\x06\x00Dummy DOCX Content Stream"
        files = {
            "file": ("specifications.docx", io.BytesIO(docx_content), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        }

        response = client.post("/api/v1/documents/upload", headers=headers, files=files)
        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "specifications.docx"
        assert data["file_type"] == ".docx"
        assert data["status"] == "PENDING"

    def test_upload_valid_txt_success(self, client: TestClient):
        token = create_user_and_get_token(client, "txt_uploader@example.com")
        headers = {"Authorization": f"Bearer {token}"}

        txt_content = "This is a clean plaintext document for RAG ingestion.".encode("utf-8")
        files = {
            "file": ("readme.txt", io.BytesIO(txt_content), "text/plain")
        }

        response = client.post("/api/v1/documents/upload", headers=headers, files=files)
        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "readme.txt"
        assert data["file_type"] == ".txt"
        assert data["status"] == "PENDING"

    def test_unauthenticated_upload_rejected(self, client: TestClient):
        files = {
            "file": ("notes.txt", io.BytesIO(b"Hello world"), "text/plain")
        }
        response = client.post("/api/v1/documents/upload", files=files)
        assert response.status_code == 401

    def test_upload_unsupported_extension_rejected(self, client: TestClient):
        token = create_user_and_get_token(client, "unsupported_ext@example.com")
        headers = {"Authorization": f"Bearer {token}"}

        files = {
            "file": ("malicious_script.py", io.BytesIO(b"import os; os.system('echo hi')"), "text/x-python")
        }
        response = client.post("/api/v1/documents/upload", headers=headers, files=files)
        assert response.status_code == 400
        assert "unsupported" in response.json()["detail"].lower()

    def test_upload_header_content_mismatch_rejected(self, client: TestClient):
        token = create_user_and_get_token(client, "mismatch@example.com")
        headers = {"Authorization": f"Bearer {token}"}

        # Named .pdf but contains plaintext header
        fake_pdf = io.BytesIO(b"I AM NOT A REAL PDF FILE HEADER")
        files = {
            "file": ("fake.pdf", fake_pdf, "application/pdf")
        }
        response = client.post("/api/v1/documents/upload", headers=headers, files=files)
        assert response.status_code == 400
        assert "missing standard %pdf-" in response.json()["detail"].lower()

    def test_upload_path_traversal_filename_sanitized(self, client: TestClient):
        token = create_user_and_get_token(client, "traversal@example.com")
        headers = {"Authorization": f"Bearer {token}"}

        pdf_content = b"%PDF-1.7\nSample PDF"
        files = {
            "file": ("../../../../etc/passwd.pdf", io.BytesIO(pdf_content), "application/pdf")
        }
        response = client.post("/api/v1/documents/upload", headers=headers, files=files)
        assert response.status_code == 201
        data = response.json()
        # Filename should be sanitized to passwd.pdf without path traversal prefixes
        assert data["filename"] == "passwd.pdf"
        assert "/" not in data["filename"]
        assert "\\" not in data["filename"]


class TestMultiTenantDocumentIsolation:
    """Mandatory Multi-Tenant Isolation tests verifying strict user data partitioning."""

    def test_multi_tenant_isolation_lifecycle(self, client: TestClient):
        # 1. Setup User A and User B
        token_a = create_user_and_get_token(client, "tenant_a@example.com")
        token_b = create_user_and_get_token(client, "tenant_b@example.com")

        headers_a = {"Authorization": f"Bearer {token_a}"}
        headers_b = {"Authorization": f"Bearer {token_b}"}

        # 2. User A uploads Document A
        pdf_a = b"%PDF-1.7\nTenant A confidential document."
        res_a = client.post(
            "/api/v1/documents/upload",
            headers=headers_a,
            files={"file": ("tenant_a_doc.pdf", io.BytesIO(pdf_a), "application/pdf")},
        )
        assert res_a.status_code == 201
        doc_a_id = res_a.json()["id"]

        # 3. User B uploads Document B
        pdf_b = b"%PDF-1.7\nTenant B confidential document."
        res_b = client.post(
            "/api/v1/documents/upload",
            headers=headers_b,
            files={"file": ("tenant_b_doc.pdf", io.BytesIO(pdf_b), "application/pdf")},
        )
        assert res_b.status_code == 201
        doc_b_id = res_b.json()["id"]

        # 4. User A lists documents -> sees only Document A
        list_a = client.get("/api/v1/documents", headers=headers_a)
        assert list_a.status_code == 200
        items_a = list_a.json()["items"]
        assert len(items_a) == 1
        assert items_a[0]["id"] == doc_a_id

        # 5. User B lists documents -> sees only Document B
        list_b = client.get("/api/v1/documents", headers=headers_b)
        assert list_b.status_code == 200
        items_b = list_b.json()["items"]
        assert len(items_b) == 1
        assert items_b[0]["id"] == doc_b_id

        # 6. User A attempts to access Document B -> returns 404 (does not disclose existence)
        get_b_by_a = client.get(f"/api/v1/documents/{doc_b_id}", headers=headers_a)
        assert get_b_by_a.status_code == 404

        # 7. User B attempts to access Document A -> returns 404
        get_a_by_b = client.get(f"/api/v1/documents/{doc_a_id}", headers=headers_b)
        assert get_a_by_b.status_code == 404

        # 8. User A attempts to delete Document B -> returns 404
        del_b_by_a = client.delete(f"/api/v1/documents/{doc_b_id}", headers=headers_a)
        assert del_b_by_a.status_code == 404

        # Verify Document B still exists for User B
        get_b_by_b = client.get(f"/api/v1/documents/{doc_b_id}", headers=headers_b)
        assert get_b_by_b.status_code == 200

        # 9. User A deletes Document A -> returns 204
        del_a = client.delete(f"/api/v1/documents/{doc_a_id}", headers=headers_a)
        assert del_a.status_code == 204

        # Verify Document A is gone for User A
        get_a_after_del = client.get(f"/api/v1/documents/{doc_a_id}", headers=headers_a)
        assert get_a_after_del.status_code == 404

        # Verify physical file for Document A was removed from disk
        settings = get_settings()
        matches_a = list(Path(settings.storage.upload_storage_dir).glob(f"*/{doc_a_id}/*"))
        assert len(matches_a) == 0


class TestUploadFailureCleanup:
    """Verify that failed uploads clean up any partially created filesystem artifacts."""

    def test_failed_upload_cleans_up_filesystem(self, client: TestClient):
        token = create_user_and_get_token(client, "cleanup_user@example.com")
        headers = {"Authorization": f"Bearer {token}"}

        # Attempt upload with invalid PDF header
        files = {
            "file": ("corrupted.pdf", io.BytesIO(b"INVALID_HEADER_CONTENT"), "application/pdf")
        }
        response = client.post("/api/v1/documents/upload", headers=headers, files=files)
        assert response.status_code == 400

        # Verify no empty or corrupted directories were left orphaned
        settings = get_settings()
        corrupted_matches = list(Path(settings.storage.upload_storage_dir).glob("*/corrupted*"))
        assert len(corrupted_matches) == 0
