"""V2 multi-user slices: per-user keys, mirrors, restore contract."""
from __future__ import annotations


def test_per_user_keys_isolate_and_legacy_keeps_working():
    from shared.auth import hash_password  # noqa: F401 (import sanity)
    from document_mgmt_service.infrastructure.encryption import (
        decrypt_payload_with_date, derive_user_master_key, encrypt_payload,
    )

    master = "master-secret"
    assert derive_user_master_key(master, None) == master
    assert derive_user_master_key(master, "__local__") == master
    assert derive_user_master_key(master, "u1") != derive_user_master_key(master, "u2")

    enc = encrypt_payload(
        {"chunk_id": "c", "document_id": "d", "version": 1, "text": "s3cret", "user_id": "u1"},
        document_id="d", version=1, upload_date="2026-01-01", master_key=master,
    )
    assert enc["text"].startswith("enc:")
    assert decrypt_payload_with_date(dict(enc), master)["text"] == "s3cret"

    # Legacy payload without user_id still decrypts with the global master.
    legacy = encrypt_payload(
        {"chunk_id": "c", "document_id": "d", "version": 1, "text": "old"},
        document_id="d", version=1, upload_date="2026-01-01", master_key=master,
    )
    assert decrypt_payload_with_date(dict(legacy), master)["text"] == "old"


def test_mirrors_are_safe_noops_without_env():
    from shared import device_sync

    assert device_sync.GLOBAL_POSTGRES_DSN == ""
    assert device_sync.mirror_manifest_pg(user_id="u", document_id="d", entry={"version": 1}) is False
    assert device_sync.mirror_qdrant_user(user_id="u")["mirrored"] is False


def test_migration_and_ui_artifacts_present():
    from pathlib import Path
    from document_mgmt_service.schemas import MIGRATION_006

    assert "users" in MIGRATION_006.sql and "user_id" in MIGRATION_006.sql
    root = Path(__file__).resolve().parents[1]
    index = (root / "ui" / "index.html").read_text(encoding="utf-8")
    app = (root / "ui" / "app.js").read_text(encoding="utf-8")
    assert "auth-modal" in index and "btn-logout" in index
    assert "setupAuth" in app and "/api/auth/" in app
    import json
    en = json.loads((root / "ui" / "i18n" / "en.json").read_text(encoding="utf-8"))
    hi = json.loads((root / "ui" / "i18n" / "hi.json").read_text(encoding="utf-8"))
    for key in ("auth_title", "auth_login", "auth_register"):
        assert key in en and key in hi
