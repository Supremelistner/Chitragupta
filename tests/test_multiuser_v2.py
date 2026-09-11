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


class _FakeRepo:
    """Minimal get_version stub with two owners + one legacy row."""

    def __init__(self, records):
        self._records = records

    def get_version(self, document_id, version, user_id=None):
        return self._records.get((document_id, version))


def _owned_record(user_id, version=1):
    from document_mgmt_service.domain.models import (
        DocumentFileKind, DocumentPrivacyClassification, DocumentProcessingStatus,
        DocumentVersionRecord, SemanticIndexStatus,
    )
    return DocumentVersionRecord(
        document_id="d1", version=version, original_filename="a.jpg",
        content_type="image/jpeg", file_kind=DocumentFileKind.IMAGE,
        storage_key="k", file_size_bytes=3, sha256="s",
        privacy=DocumentPrivacyClassification.PRIVATE,
        processing_status=DocumentProcessingStatus.COMPLETED,
        user_id=user_id,
    )


def _access_service(repo):
    from unittest.mock import MagicMock
    from document_mgmt_service.application.access import DocumentAccessService
    return DocumentAccessService(
        repository=repo, search=MagicMock(), storage=MagicMock(),
        policy=MagicMock(), auditor=MagicMock(),
    )


def test_load_record_enforces_ownership():
    repo = _FakeRepo({
        ("d1", 1): _owned_record("u-alice"),
        ("d1", 2): _owned_record("__local__"),
    })
    svc = _access_service(repo)
    assert svc._load_record("d1", 1, user_id="u-alice").user_id == "u-alice"
    # Legacy rows stay visible during transition.
    assert svc._load_record("d1", 2, user_id="u-bob").user_id == "__local__"
    # Cross-user reads 404 (no oracle).
    try:
        svc._load_record("d1", 1, user_id="u-bob")
        raise AssertionError("cross-user read must raise")
    except KeyError:
        pass
    # Unscoped callers (pre-auth paths) keep working.
    assert svc._load_record("d1", 1).user_id == "u-alice"


def test_field_value_enforces_ownership():
    from document_mgmt_service.application.fields import FieldAccessError, get_field_value
    rec = _owned_record("u-alice")
    repo = _FakeRepo({("d1", 1): rec})
    try:
        get_field_value(repo, document_id="d1", version=1, field_name="x", user_id="u-bob")
        raise AssertionError("cross-user field read must raise")
    except FieldAccessError:
        pass


def _scoped_engine():
    import tempfile
    from unittest.mock import MagicMock
    from orchestrator_service.application.orchestrator import OrchestrationEngine
    from orchestrator_service.domain.models import ServiceTarget
    from orchestrator_service.domain.ports import LLMResponse
    from orchestrator_service.infrastructure.confirmation_store import InMemoryConfirmationStore
    from orchestrator_service.infrastructure.service_clients import ServiceClientRouter
    from orchestrator_service.infrastructure.session_store import FileSessionStore
    from orchestrator_service.infrastructure.tool_registry import DefaultToolRegistry
    from model_service.infrastructure.language_preferences import LanguagePreferenceStore
    from model_service.infrastructure.translation_provider import TranslationResult
    from pathlib import Path as _Path

    class _NoOp:
        def __init__(self):
            self._store = LanguagePreferenceStore(_Path(tempfile.mkdtemp()) / "prefs.json")

        def translate_user_input(self, text, session_id):
            return TranslationResult(text=text, source="en", target="en", cached=True, latency_ms=0)

        def translate_bot_reply(self, text, session_id):
            return TranslationResult(text=text, source="en", target="en", cached=True, latency_ms=0)

        def get_preference(self, session_id):
            return self._store.get(session_id)

        def set_preference(self, session_id, **kw):
            return self._store.set(session_id, **kw)

    llm = MagicMock()
    llm.chat.return_value = LLMResponse(content="done", tool_calls=[])
    mock_client = MagicMock()
    mock_client.call_tool.return_value = {"results": []}
    tmp = tempfile.mkdtemp()
    engine = OrchestrationEngine(
        llm=llm, sessions=FileSessionStore(base_dir=f"{tmp}/s"),
        confirmations=InMemoryConfirmationStore(),
        tool_registry=DefaultToolRegistry(),
        service_router=ServiceClientRouter({ServiceTarget.DOCUMENT: mock_client}),
        translation_service=_NoOp(),
    )
    return engine, mock_client


def test_scoped_args_only_touches_document_tools():
    from orchestrator_service.domain.models import ServiceTarget, Session
    engine, _ = _scoped_engine()
    session = Session(session_id="s1", user_id="u-alice", title="t")
    doc_args = engine._scoped_args(session, ServiceTarget.DOCUMENT, {"query": "x"})
    assert doc_args["user_id"] == "u-alice"
    # Explicit caller value wins.
    assert engine._scoped_args(session, ServiceTarget.DOCUMENT, {"user_id": "u-x"})["user_id"] == "u-x"
    # Other services untouched (strict servers would reject unknown params).
    assert engine._scoped_args(session, ServiceTarget.WEB_SEARCH, {"query": "x"}) == {"query": "x"}
    # No user anywhere → passthrough (legacy single-user).
    bare = Session(session_id="s2", user_id="", title="t")
    assert engine._scoped_args(bare, ServiceTarget.DOCUMENT, {"query": "x"}) == {"query": "x"}


def test_process_message_binds_and_refuses_session():
    engine, _ = _scoped_engine()
    session = engine.create_session(user_id="", title="t")
    resp = engine.process_message(session.session_id, "hello", user_id="u-alice")
    assert resp.session_id == session.session_id
    assert engine.get_session(session.session_id).user_id == "u-alice"
    # Another account touching the same chat is refused.
    resp2 = engine.process_message(session.session_id, "hello", user_id="u-bob")
    assert "another account" in resp2.message
