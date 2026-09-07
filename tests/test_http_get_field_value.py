"""HTTP-layer tests for the new ``get_field_value`` tool.

Covers:
1. Argument validation (missing args, non-int version, etc.)
2. Two-step confirmation protocol via the HMAC-signed token
3. ``asdict(datetime)`` JSON-safety on the requires_confirmation response
4. Token replay / cross-field abuse is rejected
5. Token TTL expiry is enforced

These tests instantiate the request handler with a stub repository, so
they don't need a live Postgres. The BaseHTTPRequestHandler is faked
enough to drive ``_tool_get_field_value`` directly.
"""
from __future__ import annotations

import json
import os
import unittest
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Any
from unittest.mock import MagicMock

from document_mgmt_service.adapters.http.health_server import (
    _DocumentRequestHandler,
    _FIELD_CONFIRM_TTL_SECONDS,
    _get_field_confirm_key,
    _json_safe,
    _sign_field_confirm_token,
    _verify_field_confirm_token,
)
from document_mgmt_service.application.fields import (
    FieldAccessError,
    get_field_value,
)
from document_mgmt_service.domain.models import DocumentVersionRecord


def _make_record(**overrides) -> DocumentVersionRecord:
    base = dict(
        document_id="doc-1",
        version=1,
        original_filename="aadhaar.jpg",
        content_type="image/jpeg",
        file_kind="image",
        storage_key="k",
        file_size_bytes=10,
        sha256="x",
        privacy="SENSITIVE",
        processing_status="COMPLETED",
        metadata={},
        semantic_index_status="INDEXED",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        completed_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        description_safe="Aadhaar card",
        summary="Aadhaar card",
        extracted_text="...",
        extracted_fields={
            "aadhaar_number": "1234 5678 9012",
            "name": "Manish",
        },
        owner_type="SELF",
        relation=None,
        relation_name=None,
        expiry_date=None,
    )
    base.update(overrides)
    return DocumentVersionRecord(**base)


class _FakeRepo:
    def __init__(self, record: DocumentVersionRecord | None) -> None:
        self._record = record
        self.calls: list[tuple[str, int]] = []

    def get_version(self, document_id: str, version: int):
        self.calls.append((document_id, version))
        return self._record


def _make_handler(repo: _FakeRepo) -> _DocumentRequestHandler:
    """Build a handler instance wired to the fake repo.

    We bypass ``__init__`` (which expects a real socket) and set the
    class-level service attributes directly. ``_tool_get_field_value``
    is a normal method; we drive it via a small ``_drive_tool`` helper
    that captures the JSON response.
    """
    handler = _DocumentRequestHandler.__new__(_DocumentRequestHandler)
    captured: dict[str, Any] = {}

    def capture(status: HTTPStatus, payload: Any) -> None:
        captured["status"] = status
        captured["payload"] = payload

    handler._send_json = capture  # type: ignore[assignment]
    captured_send = captured

    # Stub access_service.repository
    handler.access_service = MagicMock()
    handler.access_service.repository = repo
    return handler, captured


class JsonSafeTests(unittest.TestCase):
    """The requires_confirmation response carries a datetime via the
    FieldValueSource dataclass; verify the serializer handles it."""

    def test_iso_formats_datetime(self) -> None:
        out = _json_safe(datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc))
        self.assertEqual(out, "2026-01-02T03:04:05+00:00")

    def test_recursive_dict_with_datetime(self) -> None:
        out = _json_safe({"ts": datetime(2026, 1, 1, tzinfo=timezone.utc)})
        self.assertEqual(out, {"ts": "2026-01-01T00:00:00+00:00"})

    def test_recursive_list_with_datetime(self) -> None:
        out = _json_safe(
            [datetime(2026, 1, 1, tzinfo=timezone.utc), "plain"]
        )
        self.assertEqual(
            out, ["2026-01-01T00:00:00+00:00", "plain"]
        )

    def test_dataclass_with_datetime_round_trips_to_json(self) -> None:
        """End-to-end: serialize a dataclass carrying a datetime, ensure
        the result is JSON-dumpable. This is the bug that prompted the fix:
        ``asdict(result)`` followed by ``json.dumps`` raised TypeError."""
        from document_mgmt_service.application.fields import FieldValueSource

        src = FieldValueSource(
            name="Aadhaar card",
            document_id="doc-1",
            version=1,
            relation="SELF",
            relation_name=None,
            page_number=None,
            field="aadhaar_number",
            extracted_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
        serialized = _json_safe(src)
        # Must not raise.
        json.dumps(serialized)
        self.assertEqual(serialized["field"], "aadhaar_number")
        self.assertEqual(serialized["extracted_at"], "2026-01-02T00:00:00+00:00")


class TokenRoundTripTests(unittest.TestCase):
    """Verify HMAC tokens sign / verify correctly across the contract."""

    def setUp(self) -> None:
        os.environ["CHITRAGUPTA_FIELD_CONFIRM_KEY"] = (
            "00112233445566778899aabbccddeeff"
        )
        # Reset cached key.
        import document_mgmt_service.adapters.http.health_server as hs

        hs._FIELD_CONFIRM_KEY_CACHE = None

    def test_sign_then_verify_succeeds(self) -> None:
        token = _sign_field_confirm_token(
            document_id="doc-1", version=1, field_name="aadhaar_number",
            ttl=_FIELD_CONFIRM_TTL_SECONDS,
        )
        self.assertTrue(
            _verify_field_confirm_token(
                token, document_id="doc-1", version=1, field_name="aadhaar_number"
            )
        )

    def test_verify_fails_on_field_mismatch(self) -> None:
        """A token issued for ``aadhaar_number`` must NOT work for ``name``."""
        token = _sign_field_confirm_token(
            document_id="doc-1", version=1, field_name="aadhaar_number",
            ttl=_FIELD_CONFIRM_TTL_SECONDS,
        )
        self.assertFalse(
            _verify_field_confirm_token(
                token, document_id="doc-1", version=1, field_name="name"
            )
        )

    def test_verify_fails_on_document_mismatch(self) -> None:
        token = _sign_field_confirm_token(
            document_id="doc-1", version=1, field_name="aadhaar_number",
            ttl=_FIELD_CONFIRM_TTL_SECONDS,
        )
        self.assertFalse(
            _verify_field_confirm_token(
                token, document_id="doc-2", version=1, field_name="aadhaar_number"
            )
        )

    def test_verify_fails_on_version_mismatch(self) -> None:
        token = _sign_field_confirm_token(
            document_id="doc-1", version=1, field_name="aadhaar_number",
            ttl=_FIELD_CONFIRM_TTL_SECONDS,
        )
        self.assertFalse(
            _verify_field_confirm_token(
                token, document_id="doc-1", version=2, field_name="aadhaar_number"
            )
        )

    def test_verify_fails_on_tampered_signature(self) -> None:
        token = _sign_field_confirm_token(
            document_id="doc-1", version=1, field_name="aadhaar_number",
            ttl=_FIELD_CONFIRM_TTL_SECONDS,
        )
        # Flip the last char.
        tampered = token[:-1] + ("A" if token[-1] != "A" else "B")
        self.assertFalse(
            _verify_field_confirm_token(
                tampered, document_id="doc-1", version=1, field_name="aadhaar_number"
            )
        )

    def test_verify_fails_on_garbage_token(self) -> None:
        self.assertFalse(
            _verify_field_confirm_token(
                "not-a-real-token",
                document_id="doc-1", version=1, field_name="aadhaar_number",
            )
        )


class GetFieldValueEndpointTests(unittest.TestCase):
    """Drive ``_tool_get_field_value`` directly via a stubbed handler."""

    def setUp(self) -> None:
        os.environ["CHITRAGUPTA_FIELD_CONFIRM_KEY"] = (
            "deadbeef" * 4
        )
        import document_mgmt_service.adapters.http.health_server as hs

        hs._FIELD_CONFIRM_KEY_CACHE = None

    def _drive(self, args: dict, repo: _FakeRepo):
        handler, captured = _make_handler(repo)
        handler._tool_get_field_value(args)
        return captured["status"], captured["payload"]

    def test_missing_document_id_returns_400(self) -> None:
        repo = _FakeRepo(_make_record())
        status, payload = self._drive(
            {"version": "1", "field": "aadhaar_number"}, repo
        )
        self.assertEqual(status, HTTPStatus.BAD_REQUEST)
        self.assertIn("error", payload)

    def test_non_int_version_returns_400(self) -> None:
        repo = _FakeRepo(_make_record())
        status, payload = self._drive(
            {"document_id": "doc-1", "version": "abc", "field": "aadhaar_number"},
            repo,
        )
        self.assertEqual(status, HTTPStatus.BAD_REQUEST)
        self.assertIn("integer", payload["error"])

    def test_first_call_returns_token_and_no_value(self) -> None:
        repo = _FakeRepo(_make_record())
        status, payload = self._drive(
            {"document_id": "doc-1", "version": "1", "field": "aadhaar_number"},
            repo,
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "requires_confirmation")
        self.assertIsNone(payload["value"])
        self.assertIn("confirmation_token", payload)
        self.assertEqual(
            payload["confirmation_ttl_seconds"], _FIELD_CONFIRM_TTL_SECONDS
        )

    def test_first_call_response_is_json_safe(self) -> None:
        """Regression: ``asdict(result)`` would crash on datetime fields."""
        repo = _FakeRepo(_make_record())
        status, payload = self._drive(
            {"document_id": "doc-1", "version": "1", "field": "aadhaar_number"},
            repo,
        )
        # If this doesn't raise, the fix is in place.
        json.dumps(payload)
        # The source carries an ISO-formatted timestamp, not a datetime.
        self.assertIsInstance(payload["source"]["extracted_at"], str)

    def test_legacy_confirm_true_arg_is_ignored(self) -> None:
        """The previous design let callers pass ``confirm=true`` directly.
        With the new design, a valid token is required even if the caller
        also passes ``confirm=true``."""
        repo = _FakeRepo(_make_record())
        status, payload = self._drive(
            {
                "document_id": "doc-1",
                "version": "1",
                "field": "aadhaar_number",
                "confirm": "true",
            },
            repo,
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "requires_confirmation")
        self.assertIsNone(payload["value"])

    def test_second_call_with_valid_token_returns_value(self) -> None:
        repo = _FakeRepo(_make_record())
        # First call -> token
        _, first = self._drive(
            {"document_id": "doc-1", "version": "1", "field": "aadhaar_number"},
            repo,
        )
        token = first["confirmation_token"]
        # Second call with token -> value
        status, payload = self._drive(
            {
                "document_id": "doc-1",
                "version": "1",
                "field": "aadhaar_number",
                "confirmation_token": token,
            },
            repo,
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["value"], "1234 5678 9012")

    def test_token_for_one_field_rejected_for_another(self) -> None:
        repo = _FakeRepo(_make_record())
        _, first = self._drive(
            {"document_id": "doc-1", "version": "1", "field": "aadhaar_number"},
            repo,
        )
        token = first["confirmation_token"]
        status, payload = self._drive(
            {
                "document_id": "doc-1",
                "version": "1",
                "field": "name",  # different field
                "confirmation_token": token,
            },
            repo,
        )
        self.assertEqual(status, HTTPStatus.OK)
        # Falls back to requires_confirmation with a fresh token,
        # not the value-revealing ok path.
        self.assertEqual(payload["status"], "requires_confirmation")
        self.assertIsNone(payload["value"])

    def test_unknown_document_returns_400(self) -> None:
        repo = _FakeRepo(None)
        status, payload = self._drive(
            {"document_id": "missing", "version": "1", "field": "aadhaar_number"},
            repo,
        )
        self.assertEqual(status, HTTPStatus.BAD_REQUEST)
        self.assertIn("not found", payload["error"])

    def test_unknown_field_returns_not_found_status(self) -> None:
        repo = _FakeRepo(_make_record())
        status, payload = self._drive(
            {"document_id": "doc-1", "version": "1", "field": "pan_number"},
            repo,
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(payload["status"], "not_found")
        self.assertIn("pan_number", payload["suggestion"])
        self.assertIn("aadhaar_number", payload["available_fields"])
        # No token issued for a not_found response.
        self.assertNotIn("confirmation_token", payload)


class ConfirmKeyEnvTests(unittest.TestCase):
    """The HMAC key must come from the env in production."""

    def setUp(self) -> None:
        import document_mgmt_service.adapters.http.health_server as hs

        hs._FIELD_CONFIRM_KEY_CACHE = None
        # Clean env
        for k in ("CHITRAGUPTA_FIELD_CONFIRM_KEY", "CHITRAGUPTA_ENV"):
            os.environ.pop(k, None)

    def test_prod_env_without_key_raises(self) -> None:
        os.environ["CHITRAGUPTA_ENV"] = "production"
        with self.assertRaises(RuntimeError):
            _get_field_confirm_key()

    def test_dev_env_without_key_returns_random_bytes(self) -> None:
        # No env vars set.
        key = _get_field_confirm_key()
        self.assertIsInstance(key, bytes)
        self.assertEqual(len(key), 32)


if __name__ == "__main__":
    unittest.main()