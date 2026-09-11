"""V1 multi-user: auth, tenancy payload, device sync (no live infra)."""
from __future__ import annotations

import time


def test_password_hash_roundtrip():
    from shared.auth import verify_password, hash_password

    stored = hash_password("correct-horse-8")
    assert verify_password("correct-horse-8", stored)
    assert not verify_password("wrong-pass-1", stored)


def test_jwt_issue_verify_refresh_window():
    from shared.auth import issue_token, verify_token, needs_refresh

    secret = "test-secret"
    token = issue_token(user_id="u123", email="a@b.c", secret=secret, ttl_seconds=3600)
    payload = verify_token(token, secret=secret)
    assert payload["sub"] == "u123"
    assert needs_refresh(payload)  # 1h < 48h window → refresh advised

    long_token = issue_token(user_id="u123", email="a@b.c", secret=secret)
    assert not needs_refresh(verify_token(long_token, secret=secret))

    expired = issue_token(user_id="u123", email="a@b.c", secret=secret, ttl_seconds=-1)
    try:
        verify_token(expired, secret=secret)
        raise AssertionError("expired token must raise")
    except Exception:
        pass


def test_bearer_parsing_rejects_missing():
    from shared.auth import bearer_user

    try:
        bearer_user(None, secret="s")
        raise AssertionError("must raise")
    except Exception:
        pass


def test_file_auth_store_register_authenticate(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from orchestrator_service.infrastructure.auth_store import FileAuthStore

    store = FileAuthStore(str(tmp_path / "users"))
    user = store.register("User@Example.com", "password-123")
    assert user["email"] == "user@example.com"
    authed = store.authenticate("user@example.com", "password-123")
    assert authed["user_id"] == user["user_id"]
    try:
        store.authenticate("user@example.com", "bad-password-1")
        raise AssertionError("must raise")
    except Exception:
        pass


def test_device_sync_push_pull_version_wins(tmp_path):
    from shared.device_sync import push_document, pull_manifest, read_blob, remove_document

    root = tmp_path / "global"
    push_document(user_id="u1", document_id="d1", version=1, blob=b"hello", root=root)
    push_document(user_id="u1", document_id="d1", version=2, blob=b"hello v2", root=root)
    stale = push_document(user_id="u1", document_id="d1", version=1, blob=b"old", root=root)
    assert stale["pushed"] is False

    manifest = pull_manifest("u1", root=root)
    assert manifest["documents"]["d1"]["version"] == 2
    assert read_blob(user_id="u1", blob_name=manifest["documents"]["d1"]["blob"], root=root) == b"hello v2"

    remove_document(user_id="u1", document_id="d1", root=root)
    assert pull_manifest("u1", root=root)["documents"] == {}


def test_migration_006_has_tenancy():
    from document_mgmt_service.schemas import MIGRATION_006, MIGRATIONS

    assert MIGRATION_006.version == 6
    assert "CREATE TABLE IF NOT EXISTS users" in MIGRATION_006.sql
    assert "ADD COLUMN IF NOT EXISTS user_id" in MIGRATION_006.sql
    assert MIGRATIONS[-1].version == 6


def test_qdrant_payload_carries_user_id():
    from document_mgmt_service.application.search import SemanticChunker
    from document_mgmt_service.domain.models import (
        DocumentFileKind, DocumentPrivacyClassification, DocumentProcessingStatus,
        DocumentVersionRecord, SemanticIndexStatus,
    )

    record = DocumentVersionRecord(
        document_id="d1", version=1, original_filename="a.jpg", content_type="image/jpeg",
        file_kind=DocumentFileKind.IMAGE, storage_key="k", file_size_bytes=3, sha256="s",
        privacy=DocumentPrivacyClassification.PRIVATE,
        processing_status=DocumentProcessingStatus.COMPLETED,
        extracted_text="hello world", user_id="u-tenant-1",
    )
    chunks = SemanticChunker().chunk(record)
    assert chunks and all(c.user_id == "u-tenant-1" for c in chunks)

    from document_mgmt_service.infrastructure.qdrant import QdrantSemanticChunkStoreAdapter
    payload = QdrantSemanticChunkStoreAdapter.__new__(QdrantSemanticChunkStoreAdapter)._build_payload(chunks[0])
    assert payload["user_id"] == "u-tenant-1"
